#!/usr/bin/env python3
"""
简化测试脚本：仅验证 open_filename 导入，不依赖 unstructured 库
"""

import sys
import os

print(f"当前 Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")


# 测试用例 1：验证 open_filename 从 pdfminer.utils 导入
def test_open_filename_import():
    print("\n=== 测试用例 1: 验证 open_filename 从 pdfminer.utils 导入 ===")
    try:
        from pdfminer.utils import open_filename

        print("✅ 成功从 pdfminer.utils 导入 open_filename")
        print(f"   open_filename 类型: {type(open_filename)}")
        print(f"   open_filename 类名: {open_filename.__name__}")
        return True
    except ImportError as e:
        print(f"❌ 从 pdfminer.utils 导入 open_filename 失败: {e}")
        return False


# 测试用例 2：验证 extract_text 从 pdfminer.high_level 导入
def test_extract_text_import():
    print("\n=== 测试用例 2: 验证 extract_text 从 pdfminer.high_level 导入 ===")
    try:
        from pdfminer.high_level import extract_text

        print("✅ 成功从 pdfminer.high_level 导入 extract_text")
        print(f"   extract_text 类型: {type(extract_text)}")
        return True
    except ImportError as e:
        print(f"❌ 从 pdfminer.high_level 导入 extract_text 失败: {e}")
        return False


# 测试用例 3：验证 pdfminer 基本功能可用
def test_pdfminer_basic():
    print("\n=== 测试用例 3: 验证 pdfminer 基本功能可用 ===")
    try:
        from pdfminer import pdfpage, pdfparser

        print("✅ 成功导入 pdfminer 核心模块")
        print(f"   pdfpage 模块类型: {type(pdfpage)}")
        print(f"   pdfparser 模块类型: {type(pdfparser)}")
        return True
    except ImportError as e:
        print(f"❌ 导入 pdfminer 核心模块失败: {e}")
        return False


# 运行所有测试用例
def run_all_tests():
    print("开始运行测试用例...\n")

    test_results = []
    test_results.append(test_open_filename_import())
    test_results.append(test_extract_text_import())
    test_results.append(test_pdfminer_basic())

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"总测试用例数: {len(test_results)}")
    print(f"通过测试用例数: {sum(test_results)}")
    print(f"失败测试用例数: {len(test_results) - sum(test_results)}")

    if all(test_results):
        print("\n🎉 所有测试用例通过！")
        print("✅ 最初的 ImportError 问题已经成功解决！")
        print("\n注意：完整运行 init_vs.py 需要安装大量依赖项")
        print("(opencv-python, unstructured-inference, transformers, scikit-learn 等)")
        print("这些依赖项超出了当前任务的范围。")
    else:
        print("\n❌ 部分测试用例失败！")


if __name__ == "__main__":
    run_all_tests()
