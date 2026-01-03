#!/usr/bin/env python3
"""
测试脚本：直接测试 open_filename 函数的导入
"""

import sys
import os

print(f"当前 Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

# 尝试导入 open_filename 函数
try:
    from pdfminer.utils import open_filename

    print("✓ 成功导入 open_filename 函数")
    print(f"open_filename 函数类型: {type(open_filename)}")
    print(f"open_filename 函数定义: {open_filename.__code__}")
    print("\n✅ 最初的 ImportError 问题已经解决！")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback

    traceback.print_exc()
    print("\n❌ 最初的 ImportError 问题仍然存在！")

# 尝试使用 pdfminer.high_level 模块
try:
    from pdfminer.high_level import extract_text, open_filename as open_filename_high

    print("\n✓ 成功导入 pdfminer.high_level 模块")
    print(f"extract_text 函数可用: {hasattr(extract_text, '__call__')}")
    print(f"high_level.open_filename 可用: {hasattr(open_filename_high, '__call__')}")
except Exception as e:
    print(f"\n✗ 导入 pdfminer.high_level 失败: {e}")
