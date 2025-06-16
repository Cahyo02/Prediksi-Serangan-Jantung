import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from mealpy.swarm_based.GWO import OriginalGWO as GWO
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar

# SMOTE imports
from imblearn.over_sampling import SMOTE
from collections import Counter

# Model saving imports
import pickle
import joblib
from datetime import datetime
import json

import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class HeartAttackPredictionPipeline:
    """
    Complete pipeline for heart attack prediction using:
    1. SMOTE for data balancing
    2. Gray Wolf Optimization for feature selection
    3. StandardScaler applied ONLY to GWO-selected features
    4. XGBoost for classification
    5. Model persistence for deployment
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None  # Will only scale selected features
        self.encoders = {}
        self.selected_features = []
        self.selected_indices = []
        self.original_feature_count = 0 
        self.smote_info = {}
        self.performance_metrics = {}
        
    def load_and_validate_data(self, filename):
        """Load dataset with comprehensive validation"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset file '{filename}' not found!")
        
        df = pd.read_csv(filename)
        print(f"📊 Dataset loaded successfully with shape: {df.shape}")
        
        # Display basic info
        print(f"📈 Dataset Info:")
        print(f"   - Rows: {df.shape[0]:,}")
        print(f"   - Columns: {df.shape[1]}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"⚠️  Missing values detected:")
            for col, missing in missing_values[missing_values > 0].items():
                print(f"   - {col}: {missing} ({missing/len(df)*100:.1f}%)")
            df = self._handle_missing_values(df)
        else:
            print("✅ No missing values found")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values appropriately"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
                else:
                    # Fill numerical with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        print("✅ Missing values handled successfully")
        return df_clean
    
    def preprocess_data(self, df, target_column="heart_attack", features_to_drop=None):
        """Enhanced preprocessing with feature dropping capability"""
        df_processed = df.copy()
        
        # Drop unwanted features if specified
        if features_to_drop:
            original_cols = df_processed.shape[1]
            df_processed = df_processed.drop(columns=features_to_drop, errors='ignore')
            dropped_count = original_cols - df_processed.shape[1]
            print(f"🗑️  Dropped {dropped_count} features: {features_to_drop}")
        
        # Encode categorical variables
        self.encoders = {}
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and col != target_column:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.encoders[col] = le
                print(f"🔢 Encoded categorical feature: {col}")
        
        # Handle target variable if it's categorical
        if df_processed[target_column].dtype == 'object':
            le_target = LabelEncoder()
            df_processed[target_column] = le_target.fit_transform(df_processed[target_column])
            self.encoders[target_column] = le_target
            print(f"🎯 Encoded target variable: {target_column}")
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column]).values
        y = df_processed[target_column].values
        feature_names = df_processed.drop(columns=[target_column]).columns.tolist()

        # STORE ORIGINAL FEATURE COUNT
        self.original_feature_count = len(feature_names)
        
        print(f"✅ Preprocessing complete:")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Samples: {X.shape[0]:,}")
        print(f"   - Target classes: {len(np.unique(y))}")
        
        return X, y, feature_names
    
    def analyze_class_distribution(self, y, dataset_name="Dataset"):
        """Analyze and visualize class distribution"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print(f"\n📊 {dataset_name} Class Distribution:")
        print("-" * 50)
        for class_label, count in zip(unique, counts):
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)  # Visual bar
            print(f"Class {class_label}: {count:5d} samples ({percentage:5.1f}%) {bar}")
        
        # Calculate imbalance ratio
        minority_count = min(counts)
        majority_count = max(counts)
        imbalance_ratio = majority_count / minority_count
        
        print(f"⚖️  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        # Determine severity
        if imbalance_ratio <= 2:
            severity = "Mild"
        elif imbalance_ratio <= 5:
            severity = "Moderate"
        else:
            severity = "Severe"
        
        print(f"📈 Imbalance Severity: {severity}")
        
        return dict(zip(unique, counts)), imbalance_ratio
    
    def apply_smote_balancing(self, X_train, y_train):
        """
        Apply SMOTE with adaptive strategy based on imbalance ratio
        """
        print(f"\n🔄 Applying SMOTE for data balancing...")
        
        # Analyze original distribution
        original_dist, imbalance_ratio = self.analyze_class_distribution(y_train, "Original Training")
        
        # Adaptive SMOTE strategy
        if imbalance_ratio <= 2:
            # Mild imbalance - conservative approach
            majority_count = max(original_dist.values())
            target_count = int(majority_count * 0.8)
            sampling_strategy = {cls: target_count for cls in original_dist.keys() 
                               if original_dist[cls] < target_count}
            k_neighbors = min(3, min(original_dist.values()) - 1) if min(original_dist.values()) > 1 else 1
        elif imbalance_ratio <= 5:
            # Moderate imbalance
            majority_count = max(original_dist.values())
            target_count = int(majority_count * 0.9)
            sampling_strategy = {cls: target_count for cls in original_dist.keys() 
                               if original_dist[cls] < target_count}
            k_neighbors = min(5, min(original_dist.values()) - 1) if min(original_dist.values()) > 1 else 1
        else:
            # Severe imbalance - full balancing
            sampling_strategy = 'auto'
            k_neighbors = min(5, min(original_dist.values()) - 1) if min(original_dist.values()) > 1 else 1
        
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=max(1, k_neighbors)
            )
            
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            # Analyze new distribution
            new_dist, new_imbalance_ratio = self.analyze_class_distribution(y_resampled, "After SMOTE")
            
            # Calculate improvement metrics
            original_size = len(y_train)
            new_size = len(y_resampled)
            increase_percentage = ((new_size - original_size) / original_size) * 100
            
            self.smote_info = {
                'original_size': original_size,
                'new_size': new_size,
                'increase_percentage': increase_percentage,
                'original_imbalance_ratio': imbalance_ratio,
                'new_imbalance_ratio': new_imbalance_ratio,
                'strategy_used': 'adaptive',
                'success': True
            }
            
            print(f"✅ SMOTE completed successfully!")
            print(f"📈 Dataset size: {original_size:,} → {new_size:,} (+{increase_percentage:.1f}%)")
            print(f"⚖️  Imbalance ratio: {imbalance_ratio:.2f}:1 → {new_imbalance_ratio:.2f}:1")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"❌ SMOTE failed: {e}")
            print("📦 Proceeding with original data...")
            
            self.smote_info = {
                'original_size': len(y_train),
                'new_size': len(y_train),
                'increase_percentage': 0,
                'original_imbalance_ratio': imbalance_ratio,
                'new_imbalance_ratio': imbalance_ratio,
                'strategy_used': 'none',
                'success': False,
                'error': str(e)
            }
            
            return X_train, y_train
    
    def optimize_features_with_gwo(self, X_train_balanced, y_train_balanced, feature_names, 
                                   epoch=50, pop_size=20):
        """
        Perform feature selection using Gray Wolf Optimization on unscaled data
        """
        print(f"\n🐺 Starting Gray Wolf Optimization for feature selection...")
        print(f"🔧 Parameters: epoch={epoch}, population_size={pop_size}")
        print(f"⚠️  Note: GWO runs on unscaled data to avoid scaling bias")
        
        # Create problem instance with unscaled data
        problem = FeatureSelectionProblem(X_train_balanced, y_train_balanced, feature_names)
        
        # Initialize GWO optimizer
        optimizer = GWO(epoch=epoch, pop_size=pop_size, verbose=True)
        
        print(f"🎯 Optimizing {len(feature_names)} features...")
        print("⏳ This may take a while...")
        
        # Run optimization
        best_agent = optimizer.solve(problem)
        best_solution = best_agent.solution
        best_fitness = best_agent.target.fitness
        
        # Convert solution to feature selection
        threshold = 0.5
        selected_idx = np.where(best_solution > threshold)[0]
        
        # Ensure minimum features
        if len(selected_idx) == 0:
            selected_idx = np.argsort(best_solution)[-5:]  # Top 5 features
            print("⚠️  No features met threshold, selecting top 5")
        elif len(selected_idx) > len(feature_names) * 0.8:
            # Limit to 80% of features
            max_features = int(len(feature_names) * 0.8)
            selected_idx = np.argsort(best_solution)[-max_features:]
            print(f"⚠️  Too many features selected, limiting to top {max_features}")
        
        self.selected_indices = selected_idx
        self.selected_features = [feature_names[i] for i in selected_idx]
        
        # Results summary
        reduction_percentage = (1 - len(self.selected_features) / len(feature_names)) * 100
        optimization_score = (1 - best_fitness) * 100
        
        print(f"\n🎉 Feature Selection Results:")
        print(f"🔢 Total features: {len(feature_names)}")
        print(f"✅ Selected features: {len(self.selected_features)}")
        print(f"📉 Feature reduction: {reduction_percentage:.1f}%")
        print(f"🎯 Optimization F1-Score: {optimization_score:.2f}%")
        
        print(f"\n📋 Selected Features:")
        for i, feature in enumerate(self.selected_features, 1):
            score = best_solution[self.selected_indices[i-1]]
            print(f"   {i:2d}. {feature:<25} (score: {score:.3f})")
        
        return best_solution, best_fitness
    
    def apply_scaling_to_selected_features(self, X_train_balanced, X_test):
        """
        Apply StandardScaler ONLY to the features selected by GWO
        """
        print(f"\n⚖️  Applying StandardScaler to selected features only...")
        
        # Extract selected features from training data
        X_train_selected = X_train_balanced[:, self.selected_indices]
        X_test_selected = X_test[:, self.selected_indices]
        
        print(f"🔧 Scaling {len(self.selected_indices)} selected features:")
        for i, feature in enumerate(self.selected_features):
            print(f"   {i+1:2d}. {feature}")
        
        # Initialize and fit scaler only on selected features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        print(f"✅ StandardScaler fitted and applied to selected features")
        print(f"📊 Scaled training data shape: {X_train_scaled.shape}")
        print(f"📊 Scaled test data shape: {X_test_scaled.shape}")
        
        # Display scaling statistics
        print(f"\n📈 Scaling Statistics:")
        for i, feature in enumerate(self.selected_features):
            mean_val = self.scaler.mean_[i]
            std_val = self.scaler.scale_[i]
            print(f"   {feature:<25}: mean={mean_val:8.3f}, std={std_val:8.3f}")
        
        return X_train_scaled, X_test_scaled
    
    def train_final_model(self, X_train_scaled, y_train_balanced, X_test_scaled, y_test):
        """
        Train final XGBoost model with GWO-selected and scaled features
        """
        print(f"\n🚀 Training final XGBoost model...")
        print(f"📊 Training data shape: {X_train_scaled.shape}")
        print(f"📊 Test data shape: {X_test_scaled.shape}")
        print(f"🎯 Using {len(self.selected_features)} GWO-selected and scaled features")
        
        # Initialize XGBoost with optimized parameters
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state,
            n_estimators=600,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.15,
            scale_pos_weight=1.22,
            n_jobs=-1
            )
            
        
        # Train model on scaled selected features
        print("⏳ Training in progress...")
        self.model.fit(X_train_scaled, y_train_balanced)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate comprehensive metrics
        self.performance_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'test_recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'test_f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'features_selected': len(self.selected_features),
            'feature_reduction_percentage': (1 - len(self.selected_features) / (self.original_feature_count)) * 100
        }
        
        # Add AUC if possible
        try:
            self.performance_metrics['test_auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            self.performance_metrics['test_auc_roc'] = None
        
        # Display results
        print(f"\n🎯 Model Performance on Test Set:")
        print(f"   📊 Accuracy:  {self.performance_metrics['test_accuracy']:.4f}")
        print(f"   🎯 Precision: {self.performance_metrics['test_precision']:.4f}")
        print(f"   🔍 Recall:    {self.performance_metrics['test_recall']:.4f}")
        print(f"   ⚖️  F1-Score:  {self.performance_metrics['test_f1_score']:.4f}")
        
        if self.performance_metrics['test_auc_roc']:
            print(f"   📈 AUC-ROC:   {self.performance_metrics['test_auc_roc']:.4f}")
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n🔍 Feature Importance Rankings (Scaled Selected Features):")
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.selected_features, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance, 1):
                bar = "█" * int(importance * 50)  # Visual bar
                print(f"   {i:2d}. {feature:<25}: {importance:.4f} {bar}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(f"   True Negative:  {cm[0,0]:4d}  |  False Positive: {cm[0,1]:4d}")
        print(f"   False Negative: {cm[1,0]:4d}  |  True Positive:  {cm[1,1]:4d}")
        
        return y_pred, y_pred_proba
    
    def save_model(self, save_dir="saved_models_new_4"):
        """
        Save complete model pipeline for deployment
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"heart_attack_gwo_scaled_xgb_{timestamp}"
        
        # Create comprehensive model package
        model_package = {
            'model': self.model,
            'selected_features': self.selected_features,
            'selected_indices': self.selected_indices,
            'scaler': self.scaler,  # Scaler fitted only on selected features
            'encoders': self.encoders,
            'smote_info': self.smote_info,
            'performance_metrics': self.performance_metrics,
            'timestamp': timestamp,
            'model_type': 'GWO_Selected_Scaled_XGBoost_SMOTE_Pipeline',
            'pipeline_version': '2.1',
            'scaling_applied_to': 'GWO_selected_features_only'
        }
        
        # Save main package
        main_path = os.path.join(save_dir, f"{model_name}.joblib")
        joblib.dump(model_package, main_path)
        
        # Save XGBoost model separately
        xgb_path = os.path.join(save_dir, f"{model_name}_xgb.json")
        self.model.save_model(xgb_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'selected_features': self.selected_features,
            'feature_count': len(self.selected_features),
            'performance_metrics': {k: float(v) if v is not None else None 
                                   for k, v in self.performance_metrics.items()},
            'smote_applied': self.smote_info.get('success', False),
            'model_type': 'GWO_Selected_Scaled_XGBoost_SMOTE_Pipeline',
            'scaling_approach': 'GWO_selected_features_only'
        }
        
        metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"\n💾 Model Saved Successfully!")
        print(f"📁 Location: {save_dir}")
        print(f"📝 Model name: {model_name}")
        print(f"📊 Performance: F1={self.performance_metrics['test_f1_score']:.4f}")
        print(f"🔧 Features: {len(self.selected_features)} (GWO-selected and scaled)")
        print(f"⚖️  Scaling: Applied to selected features only")
        
        return main_path, metadata
    
    def run_complete_pipeline(self, data_file, target_column="heart_attack", 
                             features_to_drop=None, test_size=0.2, 
                             gwo_epoch=50, gwo_pop_size=20):
        """
        Execute the complete ML pipeline with GWO-selected feature scaling
        """
        print("🚀 Starting Complete Heart Attack Prediction Pipeline")
        print("🔧 Modified Version: StandardScaler applied to GWO-selected features only")
        print("=" * 70)
        
        # Step 1: Load and preprocess data
        print("\n📂 STEP 1: Data Loading and Preprocessing")
        df = self.load_and_validate_data(data_file)
        X, y, feature_names = self.preprocess_data(df, target_column, features_to_drop)
        
        # Step 2: Initial analysis
        print("\n📊 STEP 2: Initial Data Analysis")
        self.analyze_class_distribution(y, "Complete Dataset")
        
        # Step 3: Train-test split
        print(f"\n✂️  STEP 3: Train-Test Split ({int((1-test_size)*100)}%-{int(test_size*100)}%)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Step 4: SMOTE balancing (on unscaled data)
        print("\n🔄 STEP 4: SMOTE Data Balancing (on unscaled data)")
        X_train_balanced, y_train_balanced = self.apply_smote_balancing(X_train, y_train)
        
        # Step 5: Gray Wolf Optimization (on unscaled data)
        print("\n🐺 STEP 5: Gray Wolf Optimization Feature Selection (on unscaled data)")
        best_solution, best_fitness = self.optimize_features_with_gwo(
            X_train_balanced, y_train_balanced, feature_names, gwo_epoch, gwo_pop_size
        )
        
        # Step 6: Apply scaling ONLY to selected features
        print("\n⚖️  STEP 6: StandardScaler on GWO-Selected Features")
        X_train_scaled, X_test_scaled = self.apply_scaling_to_selected_features(
            X_train_balanced, X_test
        )
        
        # Step 7: Final model training
        print("\n🎯 STEP 7: Final Model Training and Evaluation")
        y_pred, y_pred_proba = self.train_final_model(
            X_train_scaled, y_train_balanced, X_test_scaled, y_test
        )
        
        # Step 8: Save model
        print("\n💾 STEP 8: Model Persistence")
        save_path, metadata = self.save_model()
        
        # Pipeline summary
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Final Results:")
        print(f"   🎯 F1-Score: {self.performance_metrics['test_f1_score']:.4f}")
        print(f"   🔢 Features: {len(feature_names)} → {len(self.selected_features)}")
        print(f"   📉 Reduction: {(1-len(self.selected_features)/(self.original_feature_count))*100:.1f}%")
        print(f"   ⚖️  Scaling: Applied to {len(self.selected_features)} selected features only")
        print(f"   💾 Model saved: {os.path.basename(save_path)}")
        
        return self, save_path


class FeatureSelectionProblem(Problem):
    """
    Problem definition for Gray Wolf Optimization
    Operates on unscaled data to avoid scaling bias in feature selection
    """
    def __init__(self, X_train, y_train, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.n_features = X_train.shape[1]
        
        # Search space: [0, 1] for each feature
        bounds = [FloatVar(lb=0.0, ub=1.0) for _ in range(self.n_features)]
        super().__init__(bounds=bounds, minmax="min")
    
    def obj_func(self, solution):
        """
        Objective function: minimize (1 - F1_score)
        Uses unscaled data for feature selection to avoid bias
        """
        # Convert solution to feature selection
        threshold = 0.5
        selected_idx = np.where(solution > threshold)[0]
        
        # Ensure minimum features
        if len(selected_idx) == 0:
            selected_idx = np.argsort(solution)[-3:]
        elif len(selected_idx) > self.n_features * 0.8:
            selected_idx = np.argsort(solution)[-(int(self.n_features * 0.8)):]
        
        try:
            # Select features from unscaled data
            X_selected = self.X_train[:, selected_idx]
            
            # Apply basic scaling for model training within GWO
            # Note: This is temporary scaling just for GWO evaluation
            temp_scaler = StandardScaler()
            X_selected_scaled = temp_scaler.fit_transform(X_selected)
            
            # Create model
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                random_state=42,
                n_estimators=600,
                max_depth=4,
                learning_rate=0.1,
                n_jobs=-1
            )
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            f1_scores = cross_val_score(model, X_selected_scaled, self.y_train, 
                                       cv=skf, scoring='f1', n_jobs=-1)
            
            # Return negative F1 for minimization
            return 1.0 - np.mean(f1_scores)
            
        except Exception as e:
            return 1.0  # Worst possible score


def load_and_predict(model_path, new_data):
    """
    Load saved model and make predictions
    Updated to handle GWO-selected feature scaling
    """
    try:
        # Load model package
        model_package = joblib.load(model_path)
        
        # Extract components
        model = model_package['model']
        selected_idx = model_package['selected_indices']
        scaler = model_package['scaler']  # Fitted only on selected features
        encoders = model_package['encoders']
        
        print(f"📊 Model uses {len(selected_idx)} GWO-selected features")
        print(f"⚖️  Scaler applied to selected features only")
        
        # Preprocess new data
        new_data_processed = new_data.copy()
        
        # Apply encoders
        for col, encoder in encoders.items():
            if col in new_data_processed.columns and col != 'heart_attack':
                new_data_processed[col] = encoder.transform(new_data_processed[col])
        
        # Select features first, then scale only the selected ones
        X_new = new_data_processed.values
        X_new_selected = X_new[:, selected_idx]
        X_new_scaled = scaler.transform(X_new_selected)
        
        # Make predictions
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled)
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = HeartAttackPredictionPipeline(random_state=42)
    
    # Define features to drop (adjust based on your dataset)
    features_to_drop = [
        'region',
        'income_level',
        'alcohol_consumption',
        'dietary_habits',
        'participated_in_free_screening'
    ]
    
    # Run complete pipeline
    try:
        trained_pipeline, model_path = pipeline.run_complete_pipeline(
            data_file="heart_attack_prediction_indonesia.csv",
            target_column="heart_attack",
            features_to_drop=features_to_drop,
            test_size=0.2,
            gwo_epoch=50,
            gwo_pop_size=20
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Model saved at: {model_path}")
        
    except FileNotFoundError:
        print("❌ Dataset file not found. Please ensure 'heart_attack_prediction_indonesia.csv' exists.")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")