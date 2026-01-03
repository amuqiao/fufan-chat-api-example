#!/usr/bin/env python3
"""
测试脚本：验证 pdfminer.utils 模块中是否存在 open_filename 函数
"""

try:
    # 尝试直接导入
    from pdfminer.utils import open_filename
    print("✓ 成功导入 open_filename 函数")
except ImportError as e:
    print(f"✗ 导入失败：{e}")
    
    # 检查 pdfminer 版本
    import pdfminer
    print(f"当前 pdfminer 版本：{pdfminer.__version__ if hasattr(pdfminer, '__version__') else '未知'}")
    
    # 检查 pdfminer 模块结构
    import os
    pdfminer_path = os.path.dirname(pdfminer.__file__)
    print(f"pdfminer 安装路径：{pdfminer_path}")
    
    # 列出所有 pdfminer 模块
    print("\npdfminer 包中的模块：")
    for file in os.listdir(pdfminer_path):
        if file.endswith('.py') and not file.startswith('_'):
            module_name = file[:-3]
            print(f"  - {module_name}")
    
    # 检查是否有 pdfminer.high_level
    print("\n检查 pdfminer.high_level：")
try:
    from pdfminer import high_level
    print(f"✓ 存在 pdfminer.high_level 模块")
    print(f"  包含的函数：{[func for func in dir(high_level) if not func.startswith('_')]}")
except Exception as e:
    print(f"✗ 不存在 pdfminer.high_level：{e}")

# 检查 pdfminer.pdfinterp
print("\n检查 pdfminer.pdfinterp：")
try:
    from pdfminer import pdfinterp
    print(f"✓ 存在 pdfminer.pdfinterp 模块")
    print(f"  包含的函数：{[func for func in dir(pdfinterp) if not func.startswith('_')]}")
except Exception as e:
    print(f"✗ 不存在 pdfminer.pdfinterp：{e}")

# 检查 pdfminer.pdfpage
print("\n检查 pdfminer.pdfpage：")
try:
    from pdfminer import pdfpage
    print(f"✓ 存在 pdfminer.pdfpage 模块")
    print(f"  包含的函数：{[func for func in dir(pdfminer.pdfpage) if not func.startswith('_')]}")
except Exception as e:
    print(f"✗ 不存在 pdfminer.pdfpage：{e}")

# 检查 pdfminer.pdftypes
print("\n检查 pdfminer.pdftypes：")
try:
    from pdfminer import pdftypes
    print(f"✓ 存在 pdfminer.pdftypes 模块")
    print(f"  包含的函数：{[func for func in dir(pdftypes) if not func.startswith('_')]}")
except Exception as e:
    print(f"✗ 不存在 pdfminer.pdftypes：{e}")

# 检查 unstructured 版本
print("\n检查 unstructured 版本：")
try:
    import unstructured
    print(f"✓ 成功导入 unstructured")
    print(f"  版本：{unstructured.__version__}")
except Exception as e:
    print(f"✗ 导入 unstructured 失败：{e}")
