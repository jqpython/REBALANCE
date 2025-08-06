"""
Employment-Optimized Fair SMOTE Implementation

This module extends FairSMOTE with employment-specific optimizations
that consider job categories, experience patterns, and career progression
when generating synthetic samples.

Key Employment Optimizations:
1. Job Category Awareness: Ensures synthetic samples respect job category boundaries
2. Experience Progression: Synthetic samples follow realistic career development patterns
3. Education-Role Matching: Maintains appropriate education-occupation relationships
4. Sector-Specific Features: Preserves industry-specific characteristics
5. Legal Compliance: Optimized for employment discrimination detection
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple, Any, List
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import warnings

from .fair_smote import FairSMOTE


class EmploymentFairSMOTE(FairSMOTE):
    """
    Employment-Optimized Fairness-Aware SMOTE.
    
    Extends FairSMOTE with employment-specific intelligence:
    - Respects job category boundaries when generating synthetic samples
    - Follows realistic experience progression patterns
    - Maintains education-occupation relationships
    - Preserves sector-specific characteristics
    
    This ensures that synthetic employment data is not only fair
    but also realistic and professionally coherent.
    
    Parameters
    ----------
    protected_attribute : str
        Column name of the protected attribute (e.g., 'sex', 'race')
    
    job_category_column : str, optional
        Column name containing job categories/occupations
        
    experience_column : str, optional  
        Column name containing years of experience
        
    education_column : str, optional
        Column name containing education level
        
    sector_column : str, optional
        Column name containing industry/sector information
        
    preserve_job_boundaries : bool, default=True
        Whether to respect job category boundaries during synthesis
        
    experience_variance_threshold : float, default=0.3
        Maximum relative variance allowed in experience for neighbors
        
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE
        
    employment_realism_weight : float, default=0.7
        Weight for employment realism vs pure distance in neighbor selection
        (0.0 = pure distance, 1.0 = pure employment logic)
    """
    
    def __init__(self,
                 protected_attribute: str,
                 job_category_column: Optional[str] = None,
                 experience_column: Optional[str] = None,
                 education_column: Optional[str] = None,
                 sector_column: Optional[str] = None,
                 preserve_job_boundaries: bool = True,
                 experience_variance_threshold: float = 0.3,
                 k_neighbors: int = 5,
                 employment_realism_weight: float = 0.7,
                 sampling_strategy: Union[str, dict] = 'auto',
                 fairness_strategy: str = 'equal_opportunity',
                 random_state: Optional[int] = None):
        
        super().__init__(
            protected_attribute=protected_attribute,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            fairness_strategy=fairness_strategy,
            random_state=random_state
        )
        
        # Employment-specific parameters
        self.job_category_column = job_category_column
        self.experience_column = experience_column
        self.education_column = education_column
        self.sector_column = sector_column
        self.preserve_job_boundaries = preserve_job_boundaries
        self.experience_variance_threshold = experience_variance_threshold
        self.employment_realism_weight = employment_realism_weight
        
        # Employment intelligence
        self.job_profiles_ = {}
        self.education_job_mapping_ = {}
        self.experience_distributions_ = {}
        self.sector_characteristics_ = {}
    
    def _validate_inputs(self, X, y):
        """
        Enhanced input validation for employment contexts.
        """
        super()._validate_inputs(X, y)
        
        if isinstance(X, pd.DataFrame):
            # Validate employment columns exist
            employment_columns = [
                self.job_category_column, self.experience_column,
                self.education_column, self.sector_column
            ]
            
            for col in employment_columns:
                if col is not None and col not in X.columns:
                    warnings.warn(
                        f"Employment column '{col}' not found in dataset. "
                        f"Employment optimization for this feature will be disabled."
                    )
    
    def _analyze_employment_patterns(self, X, y):
        """
        Analyze employment patterns to inform intelligent synthesis.
        
        This creates a comprehensive profile of how employment features
        interact, enabling realistic synthetic sample generation.
        """
        if not isinstance(X, pd.DataFrame):
            return {}
        
        patterns = {}
        
        # 1. Job Category Profiles
        if self.job_category_column and self.job_category_column in X.columns:
            patterns['job_profiles'] = self._analyze_job_profiles(X, y)
        
        # 2. Education-Job Relationships
        if (self.education_column and self.job_category_column and 
            self.education_column in X.columns and self.job_category_column in X.columns):
            patterns['education_job_mapping'] = self._analyze_education_job_mapping(X)
        
        # 3. Experience Distributions
        if self.experience_column and self.experience_column in X.columns:
            patterns['experience_distributions'] = self._analyze_experience_patterns(X, y)
        
        # 4. Sector Characteristics
        if self.sector_column and self.sector_column in X.columns:
            patterns['sector_characteristics'] = self._analyze_sector_patterns(X, y)
        
        return patterns
    
    def _analyze_job_profiles(self, X, y):
        """
        Create profiles for each job category showing typical characteristics.
        """
        profiles = {}
        
        for job in X[self.job_category_column].unique():
            job_mask = X[self.job_category_column] == job
            job_data = X[job_mask]
            job_outcomes = y[job_mask]
            
            profile = {
                'sample_count': len(job_data),
                'positive_rate': job_outcomes.mean() if len(job_outcomes) > 0 else 0,
                'typical_features': {}
            }
            
            # Analyze typical features for this job
            for col in X.columns:
                if col != self.job_category_column and pd.api.types.is_numeric_dtype(X[col]):
                    profile['typical_features'][col] = {
                        'mean': job_data[col].mean(),
                        'std': job_data[col].std(),
                        'median': job_data[col].median()
                    }
            
            profiles[job] = profile
        
        return profiles
    
    def _analyze_education_job_mapping(self, X):
        """
        Understand which education levels are common for each job category.
        """
        mapping = defaultdict(lambda: defaultdict(int))
        
        for _, row in X.iterrows():
            job = row[self.job_category_column]
            education = row[self.education_column]
            mapping[job][education] += 1
        
        # Convert to probabilities
        for job in mapping:
            total = sum(mapping[job].values())
            for education in mapping[job]:
                mapping[job][education] = mapping[job][education] / total
        
        return dict(mapping)
    
    def _analyze_experience_patterns(self, X, y):
        """
        Analyze experience patterns across different groups and outcomes.
        """
        patterns = {}
        
        # Overall experience distribution
        patterns['overall'] = {
            'mean': X[self.experience_column].mean(),
            'std': X[self.experience_column].std(),
            'quartiles': X[self.experience_column].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # Experience by protected group
        if self.protected_attribute in X.columns:
            patterns['by_protected_group'] = {}
            for group in X[self.protected_attribute].unique():
                group_mask = X[self.protected_attribute] == group
                group_experience = X.loc[group_mask, self.experience_column]
                
                patterns['by_protected_group'][group] = {
                    'mean': group_experience.mean(),
                    'std': group_experience.std(),
                    'median': group_experience.median()
                }
        
        # Experience by outcome
        patterns['by_outcome'] = {}
        for outcome in [0, 1]:
            outcome_mask = y == outcome
            outcome_experience = X.loc[outcome_mask, self.experience_column]
            
            patterns['by_outcome'][outcome] = {
                'mean': outcome_experience.mean(),
                'std': outcome_experience.std(),
                'median': outcome_experience.median()
            }
        
        return patterns
    
    def _analyze_sector_patterns(self, X, y):
        """
        Analyze sector-specific patterns and characteristics.
        """
        patterns = {}
        
        for sector in X[self.sector_column].unique():
            sector_mask = X[self.sector_column] == sector
            sector_data = X[sector_mask]
            sector_outcomes = y[sector_mask]
            
            patterns[sector] = {
                'size': len(sector_data),
                'positive_rate': sector_outcomes.mean() if len(sector_outcomes) > 0 else 0,
                'protected_group_distribution': (
                    sector_data[self.protected_attribute].value_counts(normalize=True).to_dict()
                    if self.protected_attribute in X.columns else {}
                )
            }
        
        return patterns
    
    def _smote_for_group_with_employment_intelligence(self, group_samples, n_samples, 
                                                      employment_patterns, protected_group):
        """
        Enhanced SMOTE that considers employment relationships.
        
        This method generates synthetic samples that are not only
        mathematically sound but also professionally realistic.
        """
        if len(group_samples) < 2:
            return []
        
        # Use original DataFrame for employment logic
        if hasattr(self, '_original_X_format') and isinstance(self._original_X_format, pd.DataFrame):
            # Map encoded samples back to original indices for employment analysis
            original_df = self._original_X_format
            protected_mask = original_df[self.protected_attribute] == protected_group
            positive_mask = self._original_y_format == 1
            group_indices = original_df[protected_mask & positive_mask].index
            
            if len(group_indices) == 0:
                return self._fallback_smote(group_samples, n_samples)
        else:
            return self._fallback_smote(group_samples, n_samples)
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Select base sample
            base_idx = np.random.randint(0, len(group_samples))
            base_sample = group_samples[base_idx]
            
            # Find employment-aware neighbors
            if self.preserve_job_boundaries and len(group_indices) > 1:
                neighbor_sample = self._find_employment_aware_neighbor(
                    base_idx, group_samples, group_indices, employment_patterns
                )
            else:
                neighbor_sample = self._find_traditional_neighbor(base_sample, group_samples)
            
            # Generate synthetic sample with employment constraints
            synthetic = self._generate_employment_realistic_sample(
                base_sample, neighbor_sample, employment_patterns
            )
            
            synthetic_samples.append(synthetic)
        
        return synthetic_samples
    
    def _find_employment_aware_neighbor(self, base_idx, group_samples, group_indices, patterns):
        """
        Find a neighbor that makes employment sense.
        """
        base_sample = group_samples[base_idx]
        
        # Get corresponding original data row
        original_idx = list(group_indices)[base_idx] if base_idx < len(group_indices) else group_indices[0]
        base_original = self._original_X_format.loc[original_idx]
        
        # Calculate employment-informed distances
        distances = []
        for i, candidate_sample in enumerate(group_samples):
            if i == base_idx:
                distances.append(float('inf'))  # Exclude self
                continue
            
            candidate_idx = list(group_indices)[i] if i < len(group_indices) else group_indices[0]
            candidate_original = self._original_X_format.loc[candidate_idx]
            
            # Combine euclidean distance with employment logic
            euclidean_dist = np.linalg.norm(base_sample - candidate_sample)
            employment_dist = self._calculate_employment_distance(
                base_original, candidate_original, patterns
            )
            
            # Weighted combination
            combined_dist = (
                (1 - self.employment_realism_weight) * euclidean_dist +
                self.employment_realism_weight * employment_dist
            )
            distances.append(combined_dist)
        
        # Select from k nearest employment-aware neighbors
        k = min(self.k_neighbors, len([d for d in distances if d != float('inf')]))
        if k == 0:
            return self._find_traditional_neighbor(base_sample, group_samples)
        
        nearest_indices = np.argsort(distances)[:k]
        selected_idx = np.random.choice(nearest_indices)
        
        return group_samples[selected_idx]
    
    def _calculate_employment_distance(self, base_original, candidate_original, patterns):
        """
        Calculate distance based on employment logic.
        
        Samples that violate employment logic get higher distances.
        """
        distance = 0.0
        
        # Job category boundary penalty
        if (self.job_category_column and 
            self.job_category_column in base_original.index and
            self.preserve_job_boundaries):
            
            if base_original[self.job_category_column] != candidate_original[self.job_category_column]:
                distance += 10.0  # Heavy penalty for crossing job boundaries
        
        # Experience variance penalty
        if (self.experience_column and 
            self.experience_column in base_original.index):
            
            base_exp = base_original[self.experience_column]
            candidate_exp = candidate_original[self.experience_column]
            
            if base_exp > 0:  # Avoid division by zero
                exp_variance = abs(candidate_exp - base_exp) / base_exp
                if exp_variance > self.experience_variance_threshold:
                    distance += exp_variance * 5.0
        
        # Education-job mismatch penalty
        if (self.education_column and self.job_category_column and
            self.education_column in base_original.index and
            self.job_category_column in base_original.index):
            
            job = base_original[self.job_category_column]
            education = candidate_original[self.education_column]
            
            # Check if this education-job combination is realistic
            if (job in patterns.get('education_job_mapping', {}) and
                education in patterns['education_job_mapping'][job]):
                # Reward common combinations
                probability = patterns['education_job_mapping'][job][education]
                distance += (1 - probability) * 2.0
            else:
                # Penalize unusual combinations
                distance += 3.0
        
        return distance
    
    def _find_traditional_neighbor(self, base_sample, group_samples):
        """
        Fallback to traditional nearest neighbor selection.
        """
        distances = [
            np.linalg.norm(base_sample - sample) 
            for sample in group_samples
        ]
        
        # Exclude self (distance = 0)
        distances = [d if d > 0 else float('inf') for d in distances]
        
        if all(d == float('inf') for d in distances):
            # If somehow all distances are inf, just pick a random different sample
            available_indices = [i for i, d in enumerate(distances) if d != float('inf')]
            if not available_indices:
                available_indices = list(range(len(group_samples)))
            
            selected_idx = np.random.choice(available_indices)
        else:
            # Select from k nearest neighbors
            k = min(self.k_neighbors, len([d for d in distances if d != float('inf')]))
            nearest_indices = np.argsort(distances)[:k]
            selected_idx = np.random.choice(nearest_indices)
        
        return group_samples[selected_idx]
    
    def _generate_employment_realistic_sample(self, base_sample, neighbor_sample, patterns):
        """
        Generate a synthetic sample with employment constraints.
        """
        # Standard SMOTE interpolation
        lambda_param = np.random.random()
        synthetic = base_sample + lambda_param * (neighbor_sample - base_sample)
        
        # Apply employment-specific constraints
        synthetic = self._apply_employment_constraints(synthetic, patterns)
        
        return synthetic
    
    def _apply_employment_constraints(self, synthetic_sample, patterns):
        """
        Apply employment-specific constraints to ensure realism.
        """
        # This is a simplified version - in practice, you might have
        # more sophisticated constraints based on your specific domain
        
        # Example: Ensure experience is non-negative and reasonable
        if (self.experience_column and 
            hasattr(self, 'feature_indices_') and 
            self.experience_column in self.feature_indices_):
            
            exp_idx = self.feature_indices_[self.experience_column]
            synthetic_sample[exp_idx] = max(0, synthetic_sample[exp_idx])
            synthetic_sample[exp_idx] = min(50, synthetic_sample[exp_idx])  # Cap at 50 years
        
        return synthetic_sample
    
    def _fallback_smote(self, group_samples, n_samples):
        """
        Fallback to standard SMOTE when employment intelligence isn't available.
        """
        return super()._smote_for_group(group_samples, n_samples)
    
    def _fit_resample(self, X, y):
        """
        Enhanced fit_resample with employment intelligence.
        """
        # Analyze employment patterns first
        employment_patterns = self._analyze_employment_patterns(self._original_X_format, self._original_y_format)
        
        # Store patterns for use in synthesis
        self.employment_patterns_ = employment_patterns
        
        # Use parent class logic with our enhanced SMOTE
        return super()._fit_resample(X, y)
    
    def _generate_synthetic_samples(self, X_encoded, y, sampling_plan):
        """
        Override to use employment-aware SMOTE.
        """
        # Use original approach but with employment intelligence
        protected_values = X_encoded[:, self.protected_attribute_idx_]
        
        synthetic_samples = []
        synthetic_labels = []
        
        for group, plan in sampling_plan.items():
            if plan['positive_samples_needed'] > 0:
                # Convert group name to encoded value
                encoded_group = self.encoders_[self.protected_attribute].transform([group])[0]
                
                # Get positive samples from this protected group
                group_mask = (protected_values == encoded_group) & (y == 1)
                group_samples = X_encoded[group_mask]
                
                if len(group_samples) < 2:
                    warnings.warn(
                        f"Not enough positive samples in group {group} "
                        f"for SMOTE (found {len(group_samples)}). Skipping."
                    )
                    continue
                
                # Generate employment-aware synthetic samples
                synthetic = self._smote_for_group_with_employment_intelligence(
                    group_samples,
                    n_samples=plan['positive_samples_needed'],
                    employment_patterns=getattr(self, 'employment_patterns_', {}),
                    protected_group=group
                )
                
                synthetic_samples.extend(synthetic)
                synthetic_labels.extend([1] * len(synthetic))
        
        # Convert to arrays
        if synthetic_samples:
            X_synthetic = np.array(synthetic_samples)
            y_synthetic = np.array(synthetic_labels)
        else:
            X_synthetic = np.empty((0, X_encoded.shape[1]))
            y_synthetic = np.empty(0)
        
        return X_synthetic, y_synthetic
    
    def get_employment_insights(self):
        """
        Get insights about employment patterns discovered during fitting.
        
        Returns
        -------
        dict
            Dictionary containing employment insights and patterns
        """
        if not hasattr(self, 'employment_patterns_'):
            return {"message": "Fit the model first to get employment insights"}
        
        insights = {
            "employment_patterns_analyzed": bool(self.employment_patterns_),
            "job_categories_found": len(self.employment_patterns_.get('job_profiles', {})),
            "education_job_combinations": len(self.employment_patterns_.get('education_job_mapping', {})),
            "sector_patterns": len(self.employment_patterns_.get('sector_characteristics', {})),
            "experience_analysis_available": 'experience_distributions' in self.employment_patterns_
        }
        
        return insights