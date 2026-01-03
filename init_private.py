
import sys
sys.path.append('.')
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
import asyncio

async def init():
    service = FaissKBService(kb_name="private")
    service.do_init()
    service.do_create_kb()

if __name__ == '__main__':
    asyncio.run(init())
