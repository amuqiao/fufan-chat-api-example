#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import asyncio
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info("Starting test script...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Executable: {sys.executable}")
logger.info(f"Platform: {sys.platform}")
logger.info(f"CWD: {os.getcwd()}")

# 测试基本功能
try:
    logger.info("Testing basic imports...")
    import fastchat

    logger.info(f"FastChat version: {fastchat.__version__}")

    from fastchat.serve.controller import app, Controller

    logger.info("Successfully imported Controller")

    logger.info("Testing FastAPI app creation...")
    from fastapi import FastAPI

    test_app = FastAPI()
    logger.info("Successfully created FastAPI app")

    logger.info("Test script completed successfully!")
    print("Test script completed successfully!")

except Exception as e:
    logger.error(f"Error occurred: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
