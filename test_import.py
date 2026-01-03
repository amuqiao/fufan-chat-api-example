#!/usr/bin/env python3
"""
测试脚本：直接测试 document_loaders 模块的导入
"""

import sys
import os

# 打印当前工作目录和 Python 路径
print(f"当前工作目录: {os.getcwd()}")
print(f"Python 路径: {sys.path}")

# 尝试添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
print(f"添加到 Python 路径: {project_root}")

# 再次尝试导入
try:
    from document_loaders.pdfloader import UnstructuredLightPipeline

    print("✓ 成功导入 UnstructuredLightPipeline")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback

    traceback.print_exc()
