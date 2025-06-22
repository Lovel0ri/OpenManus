# app/extension/data_analysis.py
import os
import sys
import io
import base64
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from app.tool.base import BaseTool


class EnhancedDataAnalysis(BaseTool):
    """增强的数据分析工具

    提供高级数据分析功能，包括环境设置、数据读取、统计分析、可视化和预测模型构建。"""

    name: str = "enhanced_data_analysis"
    description: str = "提供高级数据分析功能，包括数据可视化、统计分析和预测模型"

    def _call(self, action: str, **kwargs) -> Any:
        """分发不同的数据分析操作"""
        if action == "setup":
            return self._setup_environment()
        elif action == "analyze":
            return self._analyze_data(kwargs.get("file_path"))
        elif action == "visualize":
            return self._create_visualization(
                kwargs.get("data"), kwargs.get("chart_type"), kwargs.get("columns")
            )
        elif action == "predict":
            return self._build_prediction_model(
                kwargs.get("data"),
                kwargs.get("target_variable"),
                kwargs.get("model_type", "linear")
            )
        else:
            return f"未知的数据分析操作: {action}"

    def _setup_environment(self) -> str:
        """安装数据分析所需的Python包"""
        try:
            import pip
            packages = [
                "pandas", "numpy", "matplotlib", "scikit-learn"
            ]
            for pkg in packages:
                try:
                    __import__(pkg)
                except ImportError:
                    pip.main(["install", pkg])
            return "数据分析环境已成功设置"
        except Exception as e:
            return f"设置数据分析环境时出错: {e}"

    def _analyze_data(self, file_path: str) -> Dict[str, Any]:
        """
        加载数据并执行统计分析
        返回：
          - 基本信息（行数、列数、数据类型）
          - 描述性统计
          - 缺失值概览
        """
        df = pd.read_csv(file_path)
        info = {
            "shape": df.shape,
            "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "describe": df.describe(include='all').to_dict()
        }
        return info

    def _create_visualization(
        self,
        data: Any,
        chart_type: str,
        columns: Optional[list] = None
    ) -> str:
        """
        创建图表并返回Base64编码的图片字符串
        chart_type 支持: 'line', 'bar', 'hist'
        columns: 指定绘图字段
        """
        if isinstance(data, str):  # 如果传入文件路径
            df = pd.read_csv(data)
        else:
            df = pd.DataFrame(data)

        cols = columns or df.columns.tolist()
        plt.figure()
        if chart_type == 'line':
            df[cols].plot.line()
        elif chart_type == 'bar':
            df[cols].plot.bar()
        elif chart_type == 'hist':
            df[cols].hist()
        else:
            raise ValueError(f"不支持的图表类型: {chart_type}")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

    def _build_prediction_model(
        self,
        data: Any,
        target_variable: str,
        model_type: str = 'linear'
    ) -> Dict[str, Any]:
        """
        构建预测模型并返回性能评估结果
        model_type 支持: 'linear'
        返回：
          - 训练和测试的MSE和R2
          - 模型系数（如果适用）
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = pd.DataFrame(data)

        if target_variable not in df.columns:
            raise ValueError(f"目标变量 {target_variable} 不在数据中")

        X = df.drop(columns=[target_variable]).select_dtypes(include=[np.number])
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_type == 'linear':
            model = LinearRegression()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        result = {
            'mse': mse,
            'r2': r2,
        }
        if hasattr(model, 'coef_'):
            result['coefficients'] = model.coef_.tolist()
            result['intercept'] = float(model.intercept_)
        return result
