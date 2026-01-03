import os
import sys
import json
import uuid
import bcrypt
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from langchain.docstore.document import Document

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数"""
    print("初始化 Faiss 向量数据库...")

    # 延迟导入，避免在脚本开始时就出现错误
    from configs.kb_config import username, password, hostname, database_name
    from server.db.models.user_model import UserModel

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


if __name__ == "__main__":
    main()

