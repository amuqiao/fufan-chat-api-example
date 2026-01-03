import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

logger.info("开始测试init_vs核心功能")

# 测试1: 数据库连接
try:
    logger.info("测试1: 数据库连接")
    from server.db.base import async_engine
    import asyncio
    
    async def test_db():
        """测试数据库连接"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.future import select
        from server.db.models.user_model import UserModel
        
        async with async_engine.begin() as conn:
            logger.info("✓ 数据库连接成功")
            
            async with AsyncSession(async_engine) as session:
                result = await session.execute(select(UserModel).limit(1))
                users = result.scalars().all()
                logger.info(f"✓ 数据库查询成功，用户数量: {len(users)}")
    
    asyncio.run(test_db())
except Exception as e:
    logger.error(f"✗ 数据库测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: PDF文件读取
try:
    logger.info("\n测试2: PDF文件读取")
    from PyPDF2 import PdfReader
    
    # 检查PDF文件是否存在
    pdf_path = "knowledge_base/private/content/1.pdf"
    if os.path.exists(pdf_path):
        logger.info(f"✓ PDF文件存在: {pdf_path}")
        
        # 读取PDF内容
        reader = PdfReader(pdf_path)
        logger.info(f"✓ PDF读取成功，共 {len(reader.pages)} 页")
        
        # 提取第一页文本
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text()
            logger.info(f"✓ 提取第一页文本成功，长度: {len(text)} 字符")
            logger.info(f"  第一页前100字符: {text[:100]}...")
    else:
        logger.warning(f"⚠️ PDF文件不存在: {pdf_path}")
        logger.info("创建测试PDF文件...")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "w") as f:
            f.write("%PDF-1.1\n1 0 obj<<>>stream\n测试PDF\n这是第二行\nendstream\nendobj\ntrailer<< /Root<< /Pages<< /Kids[<</MediaBox[0 0 612 792]/Contents 1 0 R>>]>> >> >>\n%%EOF")
        logger.info(f"✓ 测试PDF文件创建成功: {pdf_path}")
except Exception as e:
    logger.error(f"✗ PDF测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 文档对象创建
try:
    logger.info("\n测试3: 文档对象创建")
    
    # 尝试新的导入路径（LangChain 1.0+）
    try:
        from langchain_core.documents.base import Document
        logger.info("  使用 langchain_core.documents.base 导入 Document")
    except ImportError:
        # 尝试旧的导入路径（LangChain < 1.0）
        from langchain.docstore.document import Document
        logger.info("  使用 langchain.docstore.document 导入 Document")
    
    # 创建文档对象
    doc = Document(
        page_content="这是一个测试文档",
        metadata={"source": "test.pdf", "page": 1}
    )
    logger.info(f"✓ 文档对象创建成功")
    logger.info(f"  文档内容: {doc.page_content}")
    logger.info(f"  文档元数据: {doc.metadata}")
except Exception as e:
    logger.error(f"✗ 文档对象测试失败: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n核心功能测试完成")
