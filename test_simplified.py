import os
import sys
import logging
import asyncio

# 获取当前脚本所在目录，自动确定项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# 添加项目根目录到Python路径
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


async def test_model_config():
    """测试模型配置"""
    logger.info("=== 测试模型配置 ===")
    from configs import model_config

    logger.info(f"使用的LLM模型: {model_config.LLM_MODELS}")
    logger.info(f"使用的Embedding模型: {model_config.EMBEDDING_MODEL}")
    logger.info(f"Embedding设备: {model_config.EMBEDDING_DEVICE}")
    logger.info(f"使用的Reranker模型: {model_config.RERANKER_MODEL}")

    # 检查模型路径是否存在
    for model_type, models in model_config.MODEL_PATH.items():
        for model_name, model_path in models.items():
            if os.path.exists(model_path):
                logger.info(f"✓ 模型路径存在: {model_name} -> {model_path}")
            else:
                logger.warning(f"✗ 模型路径不存在: {model_name} -> {model_path}")


async def test_kb_folders():
    """测试知识库文件夹"""
    logger.info("\n=== 测试知识库文件夹 ===")
    # 使用PROJECT_ROOT自动获取项目根目录，避免硬编码
    kb_root = os.path.join(PROJECT_ROOT, "knowledge_base")

    # 检查知识库根目录，如果不存在则创建
    if os.path.exists(kb_root):
        logger.info(f"✓ 知识库根目录存在: {kb_root}")
    else:
        logger.warning(f"⚠️  知识库根目录不存在，正在创建: {kb_root}")
        try:
            os.makedirs(kb_root, exist_ok=True)
            logger.info(f"✓ 知识库根目录创建成功: {kb_root}")
        except Exception as e:
            logger.error(f"✗ 知识库根目录创建失败: {e}")
            return

    # 检查各个知识库文件夹
    kb_folders = ["private", "wiki", "test"]
    for folder in kb_folders:
        # 创建完整的文件夹路径
        folder_path = os.path.join(kb_root, folder)
        content_path = os.path.join(folder_path, "content")

        # 确保文件夹存在
        if not os.path.exists(content_path):
            logger.warning(f"⚠️  知识库文件夹不存在，正在创建: {content_path}")
            try:
                os.makedirs(content_path, exist_ok=True)
                logger.info(f"✓ 知识库文件夹创建成功: {content_path}")
            except Exception as e:
                logger.error(f"✗ 知识库文件夹创建失败: {e}")
                continue

        logger.info(f"✓ 知识库文件夹存在: {content_path}")
        # 列出文件夹中的文件
        try:
            files = os.listdir(content_path)
            logger.info(f"  文件夹 {folder} 中的文件: {files}")
        except Exception as e:
            logger.error(f"✗ 读取文件夹 {folder} 失败: {e}")


async def test_database_connection():
    """测试数据库连接"""
    logger.info("\n=== 测试数据库连接 ===")
    try:
        from server.db.base import async_engine
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.future import select
        from server.db.models.user_model import UserModel

        # 测试连接
        async with async_engine.begin() as conn:
            logger.info("✓ 数据库连接成功")

            # 测试查询
            async with AsyncSession(async_engine) as session:
                result = await session.execute(select(UserModel).limit(1))
                users = result.scalars().all()
                logger.info(f"✓ 数据库查询成功，用户数量: {len(users)}")
    except Exception as e:
        logger.error(f"✗ 数据库连接失败: {e}")
        import traceback

        traceback.print_exc()


async def test_embedding_model():
    """测试Embedding模型加载"""
    logger.info("\n=== 测试Embedding模型加载 ===")
    try:
        # 使用正确的导入路径
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from configs import model_config

        embed_model_name = model_config.EMBEDDING_MODEL
        embed_model_path = model_config.MODEL_PATH["embed_model"][embed_model_name]

        logger.info(
            f"尝试加载Embedding模型: {embed_model_name} 从路径: {embed_model_path}"
        )

        # 加载模型
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_path,
            model_kwargs={"device": model_config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"✓ Embedding模型加载成功")

        # 测试嵌入功能
        test_text = "测试Embedding模型"
        embedding = embeddings.embed_query(test_text)
        logger.info(f"✓ Embedding模型功能正常，嵌入向量维度: {len(embedding)}")

    except Exception as e:
        logger.error(f"✗ Embedding模型加载失败: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """主测试函数"""
    logger.info("开始简化测试...")
    logger.info(f"当前项目根目录: {PROJECT_ROOT}")

    # 执行各项测试
    await test_model_config()
    await test_kb_folders()
    await test_database_connection()
    # 可选：注释掉模型加载测试，加快测试速度
    # await test_embedding_model()

    logger.info("简化测试完成!")


if __name__ == "__main__":
    asyncio.run(main())
