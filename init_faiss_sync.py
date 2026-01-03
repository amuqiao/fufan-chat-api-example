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
    print("开始文档处理流程...")

    # 由于直接导入FaissKBService会导致Segmentation fault，我们简化初始化流程
    # 只确保知识库在数据库中存在，并创建必要的目录结构
    print("\n2. 初始化Faiss向量数据库目录结构...")

    # 为每个知识库创建必要的目录结构
    kbs_to_init = ["private", "wiki"]
    for kb_name in kbs_to_init:
        print(f"\n  初始化知识库: {kb_name}")

        # 知识库主目录
        kb_path = f"e:/github_project/fufan-chat-api-6.0.0/knowledge_base/{kb_name}"
        print(f"  知识库主目录: {kb_path}")

        # 内容目录
        content_path = f"{kb_path}/content"
        if not os.path.exists(content_path):
            os.makedirs(content_path)
            print(f"  创建内容目录: {content_path}")
        else:
            print(f"  内容目录已存在: {content_path}")

        # 向量数据库目录
        vs_path = f"{kb_path}/vector_stores"
        if not os.path.exists(vs_path):
            os.makedirs(vs_path)
            print(f"  创建向量数据库目录: {vs_path}")
        else:
            print(f"  向量数据库目录已存在: {vs_path}")

    # 测试PDF解析功能
    print("\n3. 测试PDF解析功能...")

    def test_pdf_parser():
        """测试PDF解析器"""
        # 检查是否有测试PDF文件
        private_content_path = (
            "e:/github_project/fufan-chat-api-6.0.0/knowledge_base/private/content"
        )
        pdf_files = [f for f in os.listdir(private_content_path) if f.endswith(".pdf")]

        if not pdf_files:
            print("  没有找到测试PDF文件，跳过PDF解析测试")
            return

        test_pdf = os.path.join(private_content_path, pdf_files[0])
        print(f"  使用测试PDF文件: {test_pdf}")

        # 解析PDF文件
        try:
            reader = PdfReader(test_pdf)
            total_pages = len(reader.pages)
            print(f"  PDF文件共 {total_pages} 页")

            # 只解析前2页进行测试
            for page_num in range(min(2, total_pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    print(f"  第 {page_num + 1} 页前100字符: {text[:100]}...")

            print("  PDF解析测试成功")
        except Exception as e:
            print(f"  PDF解析测试失败: {e}")

    test_pdf_parser()

    # 测试JSONL文件读取功能
    print("\n4. 测试JSONL文件读取功能...")

    def test_jsonl_reader():
        """测试JSONL文件读取"""
        jsonl_path = "e:/github_project/fufan-chat-api-6.0.0/knowledge_base/wiki/content/education.jsonl"

        if not os.path.exists(jsonl_path):
            print(f"  JSONL文件不存在: {jsonl_path}")
            print("  跳过JSONL文件读取测试")
            return

        try:
            with open(jsonl_path, "r", encoding="utf-8") as file:
                # 只读取前3行进行测试
                lines = [next(file) for _ in range(3)]

            print(f"  成功读取 {len(lines)} 行测试数据")

            # 解析其中一行
            for i, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    print(f"  第 {i+1} 行解析成功，包含字段: {list(data.keys())}")
                    if "contents" in data:
                        print(f"    contents前50字符: {data['contents'][:50]}...")
                except json.JSONDecodeError as e:
                    print(f"  第 {i+1} 行解析失败: {e}")

            print("  JSONL文件读取测试成功")
        except Exception as e:
            print(f"  JSONL文件读取测试失败: {e}")

    test_jsonl_reader()

    print("\n" + "=" * 50)
    print("Faiss向量数据库初始化准备工作完成！")
    print("=" * 50)
    print("注意: 由于直接导入FaissKBService会导致内存错误，")
    print("我们已完成了知识库的创建和目录结构的初始化。")
    print("您可以通过系统中已有的FaissKBService接口来添加文档和查询数据。")
    print("=" * 50)


if __name__ == "__main__":
    main()

