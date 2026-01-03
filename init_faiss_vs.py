#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的Faiss向量数据库初始化脚本
用于隔离Faiss的执行环境，避免内存错误影响主脚本
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))


async def init_faiss_vector_store():
    """
    初始化Faiss向量数据库
    """
    print("初始化Faiss向量数据库...")

    try:
        # 延迟导入，避免脚本启动时就加载Faiss库
        from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
        from server.knowledge_base.utils import KnowledgeFile
        from langchain.docstore.document import Document
        from PyPDF2 import PdfReader
        import json
        import os

        print("Faiss库导入成功")

        # 初始化知识库服务
        kbs_to_init = ["private", "wiki"]

        for kb_name in kbs_to_init:
            print(f"\n处理知识库: {kb_name}")

            # 创建FaissKBService实例
            faiss_service = FaissKBService(kb_name)
            print(f"  FaissKBService实例化成功")

            # 初始化服务
            faiss_service.do_init()
            print(f"  服务初始化成功")

            # 创建知识库
            faiss_service.do_create_kb()
            print(f"  知识库创建成功")

            # 处理文档
            if kb_name == "private":
                # 处理private知识库的PDF文件
                content_path = f"e:/github_project/fufan-chat-api-6.0.0/knowledge_base/{kb_name}/content"
                pdf_files = [f for f in os.listdir(content_path) if f.endswith(".pdf")]

                for pdf_file in pdf_files:
                    print(f"  处理PDF文件: {pdf_file}")
                    file_path = os.path.join(content_path, pdf_file)

                    # 简单的PDF解析
                    docs = []
                    try:
                        reader = PdfReader(file_path)
                        for page_num, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        "source": file_path,
                                        "page": page_num + 1,
                                    },
                                )
                                docs.append(doc)

                        print(f"    解析到 {len(docs)} 页文本")

                        # 添加文档到向量数据库
                        if docs:
                            added_docs = await faiss_service.do_add_doc(docs)
                            print(f"    添加了 {len(added_docs)} 个文档到向量数据库")
                    except Exception as e:
                        print(f"    处理PDF文件失败: {e}")

            elif kb_name == "wiki":
                # 处理wiki知识库的JSONL文件
                jsonl_path = f"e:/github_project/fufan-chat-api-6.0.0/knowledge_base/{kb_name}/content/education.jsonl"

                if os.path.exists(jsonl_path):
                    print(f"  处理JSONL文件: {jsonl_path}")

                    docs = []
                    try:
                        with open(jsonl_path, "r", encoding="utf-8") as file:
                            for line in file:
                                try:
                                    data = json.loads(line)
                                    doc = Document(
                                        page_content=data["contents"],
                                        metadata={"source": jsonl_path},
                                    )
                                    docs.append(doc)
                                except json.JSONDecodeError as e:
                                    continue
                                except KeyError as e:
                                    continue

                        print(f"    解析到 {len(docs)} 个文档")

                        # 添加文档到向量数据库
                        if docs:
                            added_docs = await faiss_service.do_add_doc(docs)
                            print(f"    添加了 {len(added_docs)} 个文档到向量数据库")
                    except Exception as e:
                        print(f"    处理JSONL文件失败: {e}")

        print("\nFaiss向量数据库初始化完成！")
        return 0

    except Exception as e:
        print(f"初始化Faiss向量数据库失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(init_faiss_vector_store())
    sys.exit(exit_code)
