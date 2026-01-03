import os
import sys
import importlib.util
import logging
import shutil
from modelscope import snapshot_download

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 模型配置列表
MODELS_CONFIG = [
    {
        "name": "bge-large-zh-v1.5",
        "model_id": "AI-ModelScope/bge-large-zh-v1.5",
        "local_dir": r"E:\github_project\models\bge-large-zh-v1.5",
        "revision": "master",
        "is_embedding": True,
        "need_download": False,
    },
    {
        "name": "chatglm3-6b",
        "model_id": "ZhipuAI/chatglm3-6b",
        "local_dir": r"E:\github_project\models\chatglm3-6b",
        "revision": "v1.0.0",
        "is_llm": True,
        "need_download": False,
    },
    {
        "name": "bge-reranker-large",
        "model_id": "Xorbits/bge-reranker-large",
        "local_dir": r"E:\github_project\models\bge-reranker-large",
        "revision": "master",
        "is_reranker": True,
        "need_download": False,
    },
    {
        "name": "m3e-base",
        "model_id": "AI-ModelScope/m3e-base",
        "local_dir": r"E:\github_project\models\m3e-base",
        "revision": "master",
        "is_embedding": True,
        "need_download": True,
    },
    {
        "name": "chatglm4-9b-chat",
        "model_id": "ZhipuAI/chatglm4-9b-chat",
        "local_dir": r"E:\github_project\models\chatglm4-9b-chat",
        "revision": "master",
        "is_llm": True,
        "need_download": False,
    },
]


def check_model_exists(local_dir):
    """
    检查模型文件是否存在
    """
    if not os.path.exists(local_dir):
        return False

    # 遍历目录，收集所有文件
    all_files = []
    for root, dirs, files in os.walk(local_dir):
        all_files.extend(files)

    # 基础模型文件检查

    # 至少需要config.json
    if "config.json" not in all_files:
        logger.warning(f"缺少必要文件: config.json in {local_dir}")
        return False

    # 检查模型权重文件
    has_model_weight = False
    # 情况1：单个模型文件
    single_model_files = ["model.safetensors", "pytorch_model.bin"]
    for weight_file in single_model_files:
        if weight_file in all_files:
            has_model_weight = True
            break

    # 情况2：分片模型文件（pytorch_model-*-of-*.bin）
    if not has_model_weight:
        has_sharded_model = any(
            "pytorch_model-" in file and "-of-" in file for file in all_files
        )
        has_index_file = "pytorch_model.bin.index.json" in all_files
        if has_sharded_model and has_index_file:
            has_model_weight = True

    if not has_model_weight:
        logger.warning(f"缺少模型权重文件 in {local_dir}")
        logger.warning(f"当前目录文件: {all_files}")
        return False

    # 检查是否包含tokenizer相关文件（至少一个）
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    ]
    has_tokenizer = any(
        tokenizer_file in all_files for tokenizer_file in tokenizer_files
    )
    if not has_tokenizer:
        logger.warning(f"缺少tokenizer文件: {tokenizer_files} in {local_dir}")

    logger.info(f"模型文件检查通过: {local_dir}")
    return True


def test_model_import(model_config, verify_level="full"):
    """
    测试模型是否能够导入和正常工作

    Args:
        model_config (dict): 模型配置
        verify_level (str): 验证级别，可选值：
            - "basic": 仅测试导入
            - "full": 测试导入和功能

    Returns:
        tuple: (import_success, func_test_success)
            - import_success: bool, 模型是否成功导入
            - func_test_success: bool, 功能测试是否成功
    """
    import time

    try:
        import torch
        from transformers import (
            AutoModel,
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        logger.info(f"测试导入模型: {model_config['name']}")

        # 通用参数
        model_kwargs = {"trust_remote_code": True}

        # 检查accelerate库是否可用，如果可用则使用device_map="auto"优化大模型加载
        try:
            import accelerate

            model_kwargs["device_map"] = "auto"
            logger.info(f"✓ accelerate库可用，使用device_map='auto'优化模型加载")
        except ImportError:
            logger.info(
                f"⚠️ accelerate库不可用，不使用device_map='auto' (可通过 'pip install accelerate' 安装以优化大模型加载)"
            )

        # 仅导入tokenizer进行基本测试
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["local_dir"], trust_remote_code=True
        )
        logger.info(f"✓ 成功导入tokenizer: {model_config['name']}")

        # 尝试导入模型
        if model_config.get("is_llm"):
            # 测试大语言模型
            model = AutoModel.from_pretrained(model_config["local_dir"], **model_kwargs)
            logger.info(f"✓ 成功导入LLM模型: {model_config['name']}")

            # 功能测试
            func_test_success = True
            if verify_level == "full":
                logger.info(f"开始测试LLM生成功能: {model_config['name']}")
                test_input = "你好，这是一个测试"
                start_time = time.time()

                try:
                    # 使用模型生成文本
                    inputs = tokenizer(test_input, return_tensors="pt")
                    outputs = model.generate(
                        **inputs, max_new_tokens=20, temperature=0.7, top_p=0.9
                    )
                    generated_text = tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )

                    end_time = time.time()
                    gen_time = end_time - start_time

                    logger.info(f"✓ LLM生成功能测试通过: {model_config['name']}")
                    logger.info(f"  测试输入: {test_input}")
                    logger.info(f"  生成输出: {generated_text}")
                    logger.info(f"  生成时间: {gen_time:.2f} 秒")
                except Exception as func_error:
                    func_test_success = False
                    logger.error(
                        f"⚠️ LLM生成功能测试失败: {model_config['name']}, 错误: {func_error}"
                    )
                    logger.error(
                        f"  💡 提示: 功能测试失败不影响模型使用，可能是环境或参数问题"
                    )

            return True, func_test_success

        elif model_config.get("is_embedding"):
            # 测试嵌入模型
            model = AutoModel.from_pretrained(model_config["local_dir"], **model_kwargs)
            logger.info(f"✓ 成功导入嵌入模型: {model_config['name']}")

            # 功能测试
            func_test_success = True
            if verify_level == "full":
                logger.info(f"开始测试嵌入生成功能: {model_config['name']}")
                test_sentence = "这是一个嵌入测试句子"
                start_time = time.time()

                try:
                    # 生成嵌入
                    inputs = tokenizer(
                        test_sentence,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = (
                            outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                        )

                    end_time = time.time()
                    embed_time = end_time - start_time

                    logger.info(f"✓ 嵌入生成功能测试通过: {model_config['name']}")
                    logger.info(f"  测试句子: {test_sentence}")
                    logger.info(f"  嵌入维度: {len(embedding)}")
                    logger.info(f"  生成时间: {embed_time:.2f} 秒")
                    logger.info(f"  嵌入示例: {embedding[:5]}...")
                except Exception as func_error:
                    func_test_success = False
                    logger.error(
                        f"⚠️ 嵌入生成功能测试失败: {model_config['name']}, 错误: {func_error}"
                    )
                    logger.error(
                        f"  💡 提示: 功能测试失败不影响模型使用，可能是环境或参数问题"
                    )

            return True, func_test_success

        elif model_config.get("is_reranker"):
            # 测试重排序模型
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config["local_dir"], **model_kwargs
            )
            logger.info(f"✓ 成功导入重排序模型: {model_config['name']}")

            # 功能测试
            func_test_success = True
            if verify_level == "full":
                logger.info(f"开始测试重排序功能: {model_config['name']}")
                query = "介绍一下人工智能"
                docs = [
                    "人工智能是计算机科学的一个分支。",
                    "机器学习是人工智能的一个子领域。",
                    "深度学习是机器学习的一个方法。",
                    "计算机视觉是人工智能的应用领域。",
                ]
                start_time = time.time()

                try:
                    # 生成排序分数
                    inputs = tokenizer(
                        [query] * len(docs),
                        docs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                        scores = outputs.logits.squeeze().tolist()

                    end_time = time.time()
                    rank_time = end_time - start_time

                    # 排序结果
                    ranked_docs = sorted(
                        zip(docs, scores), key=lambda x: x[1], reverse=True
                    )

                    logger.info(f"✓ 重排序功能测试通过: {model_config['name']}")
                    logger.info(f"  查询: {query}")
                    logger.info(f"  测试文档数: {len(docs)}")
                    logger.info(f"  处理时间: {rank_time:.2f} 秒")
                    logger.info(f"  排序结果:")
                    for i, (doc, score) in enumerate(ranked_docs, 1):
                        logger.info(f"    {i}. 分数: {score:.4f} | {doc}")
                except Exception as func_error:
                    func_test_success = False
                    logger.error(
                        f"⚠️ 重排序功能测试失败: {model_config['name']}, 错误: {func_error}"
                    )
                    logger.error(
                        f"  💡 提示: 功能测试失败不影响模型使用，可能是环境或参数问题"
                    )

            return True, func_test_success

        else:
            logger.warning(f"⚠️  未知模型类型: {model_config['name']}")
            return True, False
    except Exception as e:
        logger.error(f"✗ 导入模型失败: {model_config['name']}, 错误: {e}")
        return False, False


def download_model(model_config):
    """
    下载模型
    """
    model_dir = model_config["local_dir"]

    try:
        logger.info(f"开始下载模型: {model_config['name']}")
        logger.info(f"模型ID: {model_config['model_id']}")
        logger.info(f"保存路径: {model_dir}")

        # 调用snapshot_download下载模型
        downloaded_dir = snapshot_download(
            model_config["model_id"],
            revision=model_config["revision"],
            cache_dir=r"E:\github_project\models",
            local_dir=model_dir,
        )

        logger.info(
            f"✓ 模型下载完成: {model_config['name']}, 保存路径: {downloaded_dir}"
        )
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ 模型下载失败: {model_config['name']}, 错误: {e}")

        # 特殊处理ModelScope平台404错误
        if "not exists on either" in error_msg or "<Response [404]" in error_msg:
            logger.error(
                f"  💡 失败原因: 模型 {model_config['model_id']} 不在ModelScope平台上"
            )
            logger.error(f"  💡 解决方案:")
            logger.error(f"    1. 检查模型ID是否正确")
            logger.error(f"    2. 确认模型是否已发布到ModelScope")
            logger.error(f"    3. 考虑从其他平台（如Hugging Face）手动下载")
            logger.error(f"    4. 将该模型的need_download参数设置为False以跳过下载")

        # 清理已创建的模型文件夹
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                logger.info(f"  ✓ 已清理下载失败的模型文件夹: {model_dir}")
            except Exception as cleanup_error:
                logger.error(
                    f"  ✗ 清理模型文件夹失败: {model_dir}, 错误: {cleanup_error}"
                )

        return False


def main(verify_level="full"):
    """
    主函数，遍历所有模型，检查并下载

    Args:
        verify_level (str): 验证级别，可选值：
            - "basic": 仅测试导入
            - "full": 测试导入和功能
    """
    logger.info("开始处理模型下载")
    logger.info(f"共需处理 {len(MODELS_CONFIG)} 个模型")
    logger.info(f"验证级别: {verify_level}")

    # 统计信息
    total_models = len(MODELS_CONFIG)
    skip_count = 0
    skip_download_disabled_count = 0
    download_count = 0
    success_count = 0
    fail_count = 0

    for i, model_config in enumerate(MODELS_CONFIG, 1):
        logger.info(f"\n=== 处理模型 {i}/{total_models}: {model_config['name']} ===")

        # 检查是否需要下载
        if not model_config.get("need_download", True):
            logger.info(f"⏭️  模型下载已禁用，跳过: {model_config['name']}")
            skip_download_disabled_count += 1
            continue

        # 1. 检查模型文件是否存在
        if check_model_exists(model_config["local_dir"]):
            logger.info(f"✓ 模型文件已存在: {model_config['local_dir']}")

            # 2. 测试模型是否能够导入
            try:
                import_success, func_test_success = test_model_import(
                    model_config, verify_level
                )

                if import_success:
                    logger.info(f"✓ 模型可以正常导入，跳过下载")
                    skip_count += 1
                    # 功能测试失败不影响跳过下载
                    if not func_test_success:
                        logger.warning(
                            f"⚠️  模型导入成功，但功能测试失败 (不影响模型使用)"
                        )
                    continue
                else:
                    logger.warning(f"⚠️  模型存在但导入失败，重新下载")
            except Exception as e:
                # 检查是否是依赖缺失导致的错误
                error_msg = str(e).lower()
                if any(
                    keyword in error_msg
                    for keyword in [
                        "not found in your environment",
                        "requires the",
                        "missing",
                        "no module named",
                    ]
                ):
                    logger.error(f"✗ 模型导入失败，原因: 缺少依赖库")
                    logger.error(f"  错误信息: {e}")
                    logger.error(f"  解决方法: 安装缺少的依赖库后重新运行")
                    fail_count += 1
                    continue  # 跳过重新下载，因为依赖问题重新下载也无法解决
                else:
                    logger.warning(f"⚠️  模型存在但导入失败，重新下载")
        else:
            logger.info(f"⚠️  模型文件不存在，开始下载")

        # 3. 下载模型
        download_count += 1
        if download_model(model_config):
            success_count += 1
        else:
            fail_count += 1

    # 打印统计信息
    logger.info("\n=== 模型处理完成 ===")
    logger.info(f"总模型数: {total_models}")
    logger.info(f"跳过下载(已禁用): {skip_download_disabled_count}")
    logger.info(f"跳过下载(已存在且可用): {skip_count}")
    logger.info(f"下载尝试: {download_count}")
    logger.info(f"下载成功: {success_count}")
    logger.info(f"下载失败: {fail_count}")
    logger.info("模型处理完成!")


if __name__ == "__main__":
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="模型下载和验证脚本")
    parser.add_argument(
        "--verify-level",
        type=str,
        choices=["basic", "full"],
        default="full",
        help="验证级别: basic(仅测试导入), full(测试导入和功能)",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(verify_level=args.verify_level)
