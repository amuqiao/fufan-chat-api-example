import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Virtual environment: {'Yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No'}")
print(f"sys.prefix: {sys.prefix}")

# 测试基本导入
print("\nTesting basic imports...")
try:
    import sqlalchemy
    print("✓ sqlalchemy imported successfully")
except ImportError as e:
    print(f"✗ sqlalchemy import failed: {e}")

try:
    from PyPDF2 import PdfReader
    print("✓ PyPDF2 imported successfully")
except ImportError as e:
    print(f"✗ PyPDF2 import failed: {e}")

try:
    # 尝试新的导入路径（LangChain 1.0+）
    from langchain_core.documents.base import Document
    print("✓ langchain imported successfully (using langchain_core)")
except ImportError:
    try:
        # 尝试旧的导入路径（LangChain < 1.0）
        from langchain.docstore.document import Document
        print("✓ langchain imported successfully (using langchain.docstore)")
    except ImportError as e:
        print(f"✗ langchain import failed: {e}")

print("\nSimple test completed!")
