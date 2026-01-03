#!/usr/bin/env python3
"""
测试用例：验证 open_filename 函数的导入和基本使用
"""

import sys
import os

print(f"当前 Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(
    f"pdfminer 版本: {sys.modules.get('pdfminer', '未导入').__version__ if hasattr(sys.modules.get('pdfminer', object()), '__version__') else '未知'}"
)


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


# 测试用例 2：验证 open_filename 从 pdfminer.high_level 导入
def test_high_level_import():
    print("\n=== 测试用例 2: 验证 open_filename 从 pdfminer.high_level 导入 ===")
    try:
        from pdfminer.high_level import open_filename

        print("✅ 成功从 pdfminer.high_level 导入 open_filename")
        return True
    except ImportError as e:
        print(f"❌ 从 pdfminer.high_level 导入 open_filename 失败: {e}")
        return False


# 测试用例 3：验证 extract_text 从 pdfminer.high_level 导入
def test_extract_text_import():
    print("\n=== 测试用例 3: 验证 extract_text 从 pdfminer.high_level 导入 ===")
    try:
        from pdfminer.high_level import extract_text

        print("✅ 成功从 pdfminer.high_level 导入 extract_text")
        print(f"   extract_text 类型: {type(extract_text)}")
        return True
    except ImportError as e:
        print(f"❌ 从 pdfminer.high_level 导入 extract_text 失败: {e}")
        return False


# 测试用例 4：创建一个简单的 PDF 文件并使用 open_filename 打开
def test_open_filename_usage():
    print("\n=== 测试用例 4: 测试 open_filename 的基本使用 ===")
    try:
        from pdfminer.utils import open_filename

        # 创建一个简单的测试 PDF 文件路径（不需要实际存在）
        test_pdf_path = "test_sample.pdf"

        # 使用 open_filename 打开文件（这应该不会失败）
        with open_filename(test_pdf_path, "rb") as f:
            print(f"✅ 成功使用 open_filename 打开文件: {test_pdf_path}")
            print(f"   文件对象类型: {type(f)}")
        return True
    except FileNotFoundError:
        # 预期会出现 FileNotFoundError，因为我们没有创建实际的 PDF 文件
        print(f"✅ 预期行为：open_filename 正确处理了不存在的文件")
        return True
    except Exception as e:
        print(f"❌ 使用 open_filename 时发生意外错误: {e}")
        import traceback

        traceback.print_exc()
        return False


# 运行所有测试用例
def run_all_tests():
    print("开始运行测试用例...\n")

    test_results = []
    test_results.append(test_open_filename_import())
    test_results.append(test_high_level_import())
    test_results.append(test_extract_text_import())
    test_results.append(test_open_filename_usage())

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"总测试用例数: {len(test_results)}")
    print(f"通过测试用例数: {sum(test_results)}")
    print(f"失败测试用例数: {len(test_results) - sum(test_results)}")

    if all(test_results):
        print("\n🎉 所有测试用例通过！")
        print("✅ 最初的 ImportError 问题已经成功解决！")
    else:
        print("\n❌ 部分测试用例失败！")


if __name__ == "__main__":
    run_all_tests()
