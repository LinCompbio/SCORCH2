"""
SCORCH2 SHAP Explanation Tool

This module provides SHAP-based model interpretation for SCORCH2 rescoring results.
It helps understand which molecular features contribute most to the model predictions
for specific compounds or datasets.

Usage:
    python shap_explanation.py --model_path path/to/model.xgb --data_path path/to/data.csv
    
Features:
    - Generate SHAP waterfall plots for individual compounds
    - Rank features by importance
    - Support for both SC2-PS and SC2-PB models
    - Configurable visualization parameters
"""

import argparse
import os
import sys
from typing import Optional, Tuple, Dict, List
import warnings

import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体支持和样式
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

warnings.filterwarnings('ignore')


class SCORCH2ShapExplainer:
    """
    SHAP explainer for SCORCH2 models.
    
    This class provides methods to interpret SCORCH2 model predictions using
    SHAP (SHapley Additive exPlanations) values.
    """
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the SHAP explainer.
        
        Args:
            model_path: Path to the XGBoost model file
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.explainer = None
        
        self._load_model()
        self._setup_explainer()
    
    def _load_model(self) -> None:
        """Load the XGBoost model."""
        try:
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            
            # 设置GPU参数
            if self.use_gpu:
                params = {'tree_method': 'hist', 'device': 'cuda'}
                self.model.set_param(params)
                
            print(f"✓ Model loaded successfully from {self.model_path}")
            print(f"✓ Model expects {self.model.num_features()} features")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def _setup_explainer(self) -> None:
        """Setup the SHAP explainer."""
        try:
            self.explainer = shap.TreeExplainer(model=self.model)
            print("✓ SHAP explainer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SHAP explainer: {e}")
    
    def load_data(self, data_path: str, compound_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare data for SHAP analysis.
        
        Args:
            data_path: Path to the CSV file containing features
            compound_id: Specific compound ID to analyze (optional)
            
        Returns:
            Prepared DataFrame for SHAP analysis
        """
        try:
            # 加载数据
            data = pd.read_csv(data_path)
            print(f"✓ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # 过滤特定化合物
            if compound_id:
                if 'Id' in data.columns:
                    original_size = len(data)
                    data = data[data['Id'] == compound_id]
                    if data.empty:
                        raise ValueError(f"Compound ID '{compound_id}' not found in data")
                    print(f"✓ Filtered to compound '{compound_id}': {len(data)} rows")
                else:
                    print("Warning: 'Id' column not found, using all data")
            
            # 移除ID列（如果存在）
            if 'Id' in data.columns:
                data = data.drop('Id', axis=1)
            
            # 验证特征数量
            if data.shape[1] != self.model.num_features():
                raise ValueError(
                    f"Feature count mismatch: model expects {self.model.num_features()}, "
                    f"data has {data.shape[1]} features"
                )
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {data_path}: {e}")
    
    def verify_model_output(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Verify model output and check if it returns probabilities or raw scores.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dictionary with prediction statistics
        """
        try:
            raw_prediction = self.model.predict(xgb.DMatrix(data))
            
            stats = {
                'min': float(raw_prediction.min()),
                'max': float(raw_prediction.max()),
                'mean': float(raw_prediction.mean()),
                'std': float(raw_prediction.std())
            }
            
            print("=== MODEL OUTPUT VERIFICATION ===")
            print(f"Prediction range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")

            if stats['max'] <= 1.0 and stats['min'] >= 0.0:
                print("✓ Model outputs PROBABILITIES (0-1 range)")
            else:
                print("✗ Model outputs RAW SCORES/LOGITS (outside 0-1 range)")

            return stats
            
        except Exception as e:
            raise RuntimeError(f"Failed to verify model output: {e}")
    
    def calculate_shap_values(self, data: pd.DataFrame) -> shap.Explanation:
        """
        Calculate SHAP values for the given data.
        
        Args:
            data: Input data for SHAP analysis
            
        Returns:
            SHAP explanation object
        """
        try:
            print(f"Calculating SHAP values for {data.shape[0]} samples...")
            shap_values = self.explainer(data, check_additivity=False)
            print("✓ SHAP values calculated successfully")
            return shap_values
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate SHAP values: {e}")
    
    def rank_features(self, shap_values: shap.Explanation, data: pd.DataFrame) -> pd.DataFrame:
        """
        Rank features by their SHAP importance.
        
        Args:
            shap_values: SHAP explanation object
            data: Original data with feature names
            
        Returns:
            DataFrame with ranked features
        """
        try:
            # 计算特征重要性（绝对值平均）
            shap_mean_abs = np.abs(shap_values.values).mean(axis=0)

            # 创建排序DataFrame
            feature_ranking = pd.DataFrame({
                'feature': data.columns,
                'importance': shap_mean_abs
            }).sort_values(by='importance', ascending=False)
            
            print(f"✓ Features ranked by SHAP importance")
            return feature_ranking
            
        except Exception as e:
            raise RuntimeError(f"Failed to rank features: {e}")
    
    def plot_waterfall(self, 
                      shap_values: shap.Explanation, 
                      sample_idx: int = 0,
                      max_display: int = 20,
                      compound_id: str = "compound",
                      save_path: Optional[str] = None,
                      show_plot: bool = True) -> None:
        """
        Create a SHAP waterfall plot.
        
        Args:
            shap_values: SHAP explanation object
            sample_idx: Index of the sample to plot
            max_display: Maximum number of features to display
            compound_id: Compound identifier for the plot title
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
        """
        try:
            # 创建waterfall图
            shap.plots.waterfall(shap_values[sample_idx], max_display=max_display, show=False)

            # 获取当前图形和轴
            fig, ax = plt.gcf(), plt.gca()

            # 设置图形参数
            fig.set_size_inches(16, 10)
            fig.set_dpi(300)

            # 调整字体大小
            ax.tick_params(labelsize=14)
            ax.set_xlabel(f"SHAP value - {compound_id}", fontsize=16)
            
            # 调整布局
            fig.tight_layout()
            
            # 保存图片
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"✓ Plot saved to {save_path}")
            
            # 显示图片
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            raise RuntimeError(f"Failed to create waterfall plot: {e}")
    
    def save_feature_ranking(self, feature_ranking: pd.DataFrame, output_path: str) -> None:
        """
        Save feature ranking to CSV file.
        
        Args:
            feature_ranking: DataFrame with ranked features
            output_path: Path to save the ranking file
        """
        try:
            feature_ranking.to_csv(output_path, index=False)
            print(f"✓ Feature ranking saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save feature ranking: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="SCORCH2 SHAP Explanation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze specific compound
    python shap_explanation.py --model models/sc2_pb.xgb --data features.csv --compound CHEMBL123456
    
    # Generate feature ranking
    python shap_explanation.py --model models/sc2_ps.xgb --data features.csv --ranking-only
    
    # Save plots and rankings
    python shap_explanation.py --model models/sc2_pb.xgb --data features.csv \\
        --compound CHEMBL123456 --output-dir results/
        """
    )
    
    parser.add_argument(
        '--model', 
        required=True,
        help='Path to XGBoost model file (.xgb)'
    )
    parser.add_argument(
        '--data', 
        required=True,
        help='Path to CSV file with normalized features'
    )
    parser.add_argument(
        '--compound',
        help='Specific compound ID to analyze'
    )
    parser.add_argument(
        '--max-display',
        type=int,
        default=20,
        help='Maximum number of features to display in waterfall plot (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--ranking-only',
        action='store_true',
        help='Only generate feature ranking, skip waterfall plot'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化解释器
        explainer = SCORCH2ShapExplainer(
            model_path=args.model,
            use_gpu=not args.no_gpu
        )
        
        # 加载数据
        data = explainer.load_data(args.data, args.compound)
        
        # 验证模型输出
        explainer.verify_model_output(data)
        
        # 计算SHAP值
        shap_values = explainer.calculate_shap_values(data)
        
        # 特征排序
        feature_ranking = explainer.rank_features(shap_values, data)
        
        # 保存特征排序
        ranking_path = os.path.join(args.output_dir, 'shap_feature_ranking.csv') if args.output_dir else 'shap_feature_ranking.csv'
        explainer.save_feature_ranking(feature_ranking, ranking_path)
        
        # 显示top特征
        print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
        for i, (_, row) in enumerate(feature_ranking.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.6f}")
        
        # 生成waterfall图（如果不是仅排序模式）
        if not args.ranking_only:
            compound_name = args.compound if args.compound else f"sample_0"
            plot_path = None
            if args.output_dir:
                plot_path = os.path.join(args.output_dir, f'shap_waterfall_{compound_name}.png')
            
            explainer.plot_waterfall(
                shap_values=shap_values,
                sample_idx=0,
                max_display=args.max_display,
                compound_id=compound_name,
                save_path=plot_path,
                show_plot=True
            )
        
        print(f"\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()