import os
import sys
import json
import uuid
import bcrypt
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数"""
    print("初始化 Faiss 向量数据库...")

    # 延迟导入，避免在脚本开始时就出现错误
    from configs.kb_config import username, password, hostname, database_name
    from server.db.models.user_model import UserModel
    from server.db.models.knowledge_base_model import KnowledgeBaseModel

    # 创建 SQLAlchemy 同步引擎和会话
    DATABASE_URL = f"mysql+pymysql://{username}:{password}@{hostname}/{database_name}?charset=utf8mb4"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def get_db():
        """获取数据库会话"""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def get_user_by_username(username: str):
        """根据用户名获取用户"""
        with next(get_db()) as db:
            user = db.execute(select(UserModel).filter_by(username=username)).scalar()
            return user

    def register_user(username: str, password: str):
        """注册新用户"""
        with next(get_db()) as db:
            # 密码加密
            hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

            # 创建新用户
            new_user = UserModel(
                id=str(uuid.uuid4()),
                username=username,
                password_hash=hashed_password.decode(),
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return {"id": new_user.id, "username": new_user.username}

    def get_kb_by_name(kb_name: str):
        """根据知识库名称获取知识库"""
        with next(get_db()) as db:
            kb = db.execute(
                select(KnowledgeBaseModel).filter_by(kb_name=kb_name)
            ).scalar()
            return kb

    def add_kb_to_db(
        kb_name: str, kb_info: str, vs_type: str, embed_model: str, user_id: str
    ):
        """添加知识库到数据库"""
        with next(get_db()) as db:
            new_kb = KnowledgeBaseModel(
                kb_name=kb_name,
                kb_info=kb_info,
                vs_type=vs_type,
                embed_model=embed_model,
                user_id=user_id,
            )
            db.add(new_kb)
            db.commit()
            db.refresh(new_kb)
            return new_kb

    # 检查用户是否存在
    admin_user = get_user_by_username(username="admin")

    if admin_user is None:
        # 用户不存在，创建用户
        print("创建管理员用户...")
        admin_user = register_user(username="admin", password="admin")
        user_id = admin_user["id"]
    else:
        user_id = admin_user.id

    print(f"使用用户 ID: {user_id}")

    # 检查并创建private知识库
    private_kb = get_kb_by_name("private")
    if private_kb is None:
        print("创建private知识库...")
        private_kb = add_kb_to_db(
            kb_name="private",
            kb_info="个人/公司私有知识库数据",
            vs_type="faiss",
            embed_model="bge-large-zh-v1.5",
            user_id=user_id,
        )

    # 检查并创建wiki知识库
    wiki_kb = get_kb_by_name("wiki")
    if wiki_kb is None:
        print("创建wiki知识库...")
        wiki_kb = add_kb_to_db(
            kb_name="wiki",
            kb_info="wiki公共数据信息",
            vs_type="faiss",
            embed_model="bge-large-zh-v1.5",
            user_id=user_id,
        )

    print("知识库创建完成")

    # 导入知识库服务相关模块
    from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
    from server.knowledge_base.utils import KnowledgeFile

    def simple_pdf_parser(file_path):
        """简单的PDF解析器"""
        docs = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num + 1},
                    )
                    docs.append(doc)
            print(f"解析PDF文件 {file_path}，共 {len(docs)} 页")
        except Exception as e:
            print(f"解析PDF文件 {file_path} 失败: {e}")
        return docs

    def process_wiki_documents(user_id):
        """处理wiki文档"""
        print("处理wiki文档...")

        # 文件夹路径
        file_path = "e:/github_project/fufan-chat-api-6.0.0/knowledge_base/wiki/content/education.jsonl"

        # 确保文件夹存在
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建文件夹: {folder_path}")

        # 读取jsonl文件
        docs = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        document = Document(
                            page_content=data["contents"],
                            metadata={"source": file_path},
                        )
                        docs.append(document)
                    except json.JSONDecodeError as e:
                        print(f"解析JSON错误: {e}")
                    except KeyError as e:
                        print(f"JSON数据缺少键: {e}")
            print(f"读取到 {len(docs)} 个wiki文档")
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

        # 实例化 FaissKBService
        faiss_service = FaissKBService("wiki")

        # 创建 KnowledgeFile 对象
        kb_file = KnowledgeFile("education.jsonl", "wiki")

        # 添加文档到 FAISS 服务
        try:
            added_docs_info = faiss_service.add_doc(kb_file, docs=docs)
            print(f"添加wiki文档成功: {added_docs_info}")
        except Exception as e:
            print(f"添加wiki文档失败: {e}")

    def process_private_documents(user_id):
        """处理private文档"""
        print("处理private文档...")

        # 文件夹路径，包含所有PDF文件
        folder_path = (
            "e:/github_project/fufan-chat-api-6.0.0/knowledge_base/private/content"
        )

        # 确保文件夹存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建文件夹: {folder_path}")
            return

        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        print(f"找到 {len(pdf_files)} 个PDF文件")

        if not pdf_files:
            print("没有找到PDF文件，跳过处理")
            return

        # 实例化 FaissKBService
        faiss_service = FaissKBService("private")

        # 处理每一个PDF文件
        for pdf_file in pdf_files:
            full_path = os.path.join(folder_path, pdf_file)
            print(f"处理PDF文件: {pdf_file}")

            # 解析PDF文件
            docs = simple_pdf_parser(full_path)

            if docs:
                # 创建 KnowledgeFile 对象
                kb_file = KnowledgeFile(pdf_file, "private")

                # 添加文档到 FAISS 服务
                try:
                    added_docs_info = faiss_service.add_doc(kb_file, docs=docs)
                    print(f"添加PDF文档 {pdf_file} 成功: {added_docs_info}")
                except Exception as e:
                    print(f"添加PDF文档 {pdf_file} 失败: {e}")

    # 处理文档
    process_wiki_documents(user_id)
    process_private_documents(user_id)

    print("Faiss向量数据库初始化完成")


if __name__ == "__main__":
    main()

