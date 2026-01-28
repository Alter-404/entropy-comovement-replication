#!/usr/bin/env python3
"""
Extension 3: Walk-Forward XGBoost Model Training

Implements a strictly out-of-sample walk-forward validation:
- Initial training: 1963-1989
- First prediction: 1990
- Expanding window: add 1 year, retrain, predict next year

CRITICAL: No information leakage - only uses data from t-1 or earlier to predict t.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

# Try to import XGBoost, fall back to sklearn if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  Warning: XGBoost not installed, using RandomForest fallback")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Global demo mode flag
DEMO_MODE = False


def load_ml_data(data_dir: Path) -> pd.DataFrame:
    """Load the prepared ML dataset."""
    print("  Loading ML dataset...")
    
    ml_path = data_dir / "ML_Features_Target.parquet"
    
    if ml_path.exists():
        df = pd.read_parquet(ml_path)
    else:
        raise FileNotFoundError(f"ML dataset not found at {ml_path}. Run extension3_ml_prep.py first.")
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    print(f"    Loaded {len(df):,} observations")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Identify feature columns for modeling."""
    exclude_cols = {'PERMNO', 'DATE', 'YEAR_MONTH', 'TARGET_RANK_NEXT', 'TARGET_RANK', 'RET'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def create_model(use_xgboost: bool = True, demo: bool = False):
    """Create the ML model."""
    if demo:
        # Fast model for demo mode - use small GradientBoosting for feature importance
        model = GradientBoostingRegressor(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        return model
    
    if use_xgboost and HAS_XGBOOST:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        # Fallback to GradientBoosting (similar to XGBoost)
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    return model


def walk_forward_train(df: pd.DataFrame, 
                       initial_train_end: str = "1989-12-31",
                       step_months: int = 12,
                       demo: bool = False) -> pd.DataFrame:
    """
    Walk-forward training with expanding window.
    
    Parameters
    ----------
    df : DataFrame
        ML dataset with features and target
    initial_train_end : str
        End date for initial training period
    step_months : int
        Number of months to predict before retraining
    
    Returns
    -------
    DataFrame
        Predictions with columns: DATE, PERMNO, Predicted_Rank, Actual_Rank
    """
    print("  Starting walk-forward training...")
    
    feature_cols = get_feature_columns(df)
    print(f"    Features: {feature_cols}")
    
    df = df.sort_values('DATE')
    df = df.dropna(subset=feature_cols + ['TARGET_RANK_NEXT'])
    
    # Define time periods
    initial_end = pd.Timestamp(initial_train_end)
    all_dates = df['DATE'].unique()
    all_dates = np.sort(all_dates)
    
    # Filter to dates after initial period
    prediction_dates = all_dates[all_dates > initial_end]
    
    if len(prediction_dates) == 0:
        print("    Warning: No prediction dates available. Adjusting initial period.")
        # Use first 60% for training
        split_idx = int(len(all_dates) * 0.6)
        initial_end = all_dates[split_idx]
        prediction_dates = all_dates[all_dates > initial_end]
    
    print(f"    Initial training ends: {initial_end}")
    print(f"    Prediction period: {prediction_dates[0]} to {prediction_dates[-1]}")
    print(f"    Total prediction months: {len(prediction_dates)}")
    
    all_predictions = []
    scaler = StandardScaler()
    
    # Group prediction dates into windows
    n_windows = max(1, len(prediction_dates) // step_months)
    
    for window_idx in range(n_windows):
        start_idx = window_idx * step_months
        end_idx = min((window_idx + 1) * step_months, len(prediction_dates))
        
        if start_idx >= len(prediction_dates):
            break
        
        window_dates = prediction_dates[start_idx:end_idx]
        train_end = window_dates[0] - pd.DateOffset(months=1)
        
        # Training data: all data up to train_end
        train_mask = df['DATE'] <= train_end
        train_df = df[train_mask]
        
        if len(train_df) < 100:
            print(f"    Window {window_idx + 1}: Insufficient training data ({len(train_df)}), skipping")
            continue
        
        # Prepare training data
        X_train = train_df[feature_cols].values
        y_train = train_df['TARGET_RANK_NEXT'].values
        
        # Handle missing values
        X_train = np.nan_to_num(X_train, nan=0.0)
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = create_model(use_xgboost=HAS_XGBOOST, demo=demo)
        model.fit(X_train_scaled, y_train)
        
        # Predict for window
        for pred_date in window_dates:
            pred_mask = df['DATE'] == pred_date
            pred_df = df[pred_mask]
            
            if len(pred_df) == 0:
                continue
            
            X_pred = pred_df[feature_cols].values
            X_pred = np.nan_to_num(X_pred, nan=0.0)
            X_pred_scaled = scaler.transform(X_pred)
            
            predictions = model.predict(X_pred_scaled)
            
            # Clip predictions to [0, 1]
            predictions = np.clip(predictions, 0, 1)
            
            for idx, (permno, actual) in enumerate(zip(pred_df['PERMNO'], pred_df['TARGET_RANK_NEXT'])):
                all_predictions.append({
                    'DATE': pred_date,
                    'PERMNO': permno,
                    'PREDICTED_RANK': predictions[idx],
                    'ACTUAL_RANK': actual
                })
        
        if (window_idx + 1) % 5 == 0:
            print(f"    Completed window {window_idx + 1}/{n_windows}")
    
    predictions_df = pd.DataFrame(all_predictions)
    
    print(f"\n    Total predictions: {len(predictions_df):,}")
    
    return predictions_df, model, feature_cols


def evaluate_predictions(predictions_df: pd.DataFrame) -> dict:
    """Evaluate prediction quality."""
    print("\n  Evaluating predictions...")
    
    df = predictions_df.dropna()
    
    # Overall metrics
    mse = mean_squared_error(df['ACTUAL_RANK'], df['PREDICTED_RANK'])
    r2 = r2_score(df['ACTUAL_RANK'], df['PREDICTED_RANK'])
    
    # Rank correlation (more relevant for portfolio sorting)
    rank_corr = df['PREDICTED_RANK'].corr(df['ACTUAL_RANK'], method='spearman')
    
    # Hit rate: Did we correctly identify top/bottom decile?
    df['PRED_DECILE'] = pd.qcut(df['PREDICTED_RANK'], 10, labels=False, duplicates='drop') + 1
    df['ACTUAL_DECILE'] = pd.qcut(df['ACTUAL_RANK'], 10, labels=False, duplicates='drop') + 1
    
    # Extreme decile accuracy
    extreme_mask = (df['ACTUAL_DECILE'] == 1) | (df['ACTUAL_DECILE'] == 10)
    extreme_correct = ((df['PRED_DECILE'] == df['ACTUAL_DECILE']) & extreme_mask).sum()
    extreme_total = extreme_mask.sum()
    extreme_accuracy = extreme_correct / extreme_total if extreme_total > 0 else 0
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'rank_correlation': rank_corr,
        'extreme_decile_accuracy': extreme_accuracy
    }
    
    print(f"    MSE: {mse:.4f}")
    print(f"    R²: {r2:.4f}")
    print(f"    Rank Correlation (Spearman): {rank_corr:.4f}")
    print(f"    Extreme Decile Accuracy: {extreme_accuracy:.1%}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Walk-forward XGBoost training")
    parser.add_argument("--demo", action="store_true", help="Use demo settings")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--initial-end", type=str, default="1989-12-31",
                        help="End of initial training period")
    parser.add_argument("--step-months", type=int, default=12,
                        help="Months between retraining")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 3: WALK-FORWARD XGBOOST TRAINING")
    print("=" * 70)
    
    if HAS_XGBOOST:
        print("\n  Using XGBoost")
    else:
        print("\n  Using GradientBoosting (XGBoost not installed)")
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    
    # Adjust for demo mode
    if args.demo:
        args.step_months = 24  # Larger steps for speed
    
    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    
    try:
        ml_df = load_ml_data(data_dir)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Running ML prep first...")
        import subprocess
        demo_flag = ["--demo"] if args.demo else []
        subprocess.run([sys.executable, "scripts/extension3_ml_prep.py"] + demo_flag,
                       cwd=str(project_root))
        ml_df = load_ml_data(data_dir)
    
    # Walk-forward training
    print("\n" + "-" * 70)
    print("WALK-FORWARD TRAINING")
    print("-" * 70)
    
    predictions_df, model, feature_cols = walk_forward_train(
        ml_df,
        initial_train_end=args.initial_end,
        step_months=args.step_months,
        demo=args.demo
    )
    
    # Evaluate
    print("\n" + "-" * 70)
    print("PREDICTION EVALUATION")
    print("-" * 70)
    
    metrics = evaluate_predictions(predictions_df)
    
    # Save predictions
    print("\n" + "-" * 70)
    print("SAVING OUTPUTS")
    print("-" * 70)
    
    pred_path = data_dir / "ML_Predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"  Saved predictions: {pred_path}")
    
    # Save feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n  Feature Importance (last model):")
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['Feature']}: {row['Importance']:.4f}")
        
        importance_path = data_dir / "ML_Feature_Importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"\n  Saved feature importance: {importance_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = data_dir / "ML_Metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\n" + "=" * 70)
    print("WALK-FORWARD TRAINING COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
