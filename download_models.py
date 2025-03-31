#!/usr/bin/env python3
"""
模型批量下载脚本
此脚本用于提前下载模型，避免每次运行代码时重新下载
支持并行下载多个模型
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import yaml
from huggingface_hub import snapshot_download
import concurrent.futures
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_name, quantization=None, output_dir=None):
    """
    下载指定的模型和分词器
    
    参数:
        model_name: 模型名称或路径
        quantization: 量化位数（4或8）
        output_dir: 输出目录，如果不指定则使用默认的缓存目录
    """
    try:
        logger.info(f"开始下载模型: {model_name}")
        
        # 为每个模型创建单独的输出目录（如果指定了output_dir）
        model_output_dir = None
        if output_dir:
            model_output_dir = os.path.join(output_dir, model_name.split("/")[-1])
            os.makedirs(model_output_dir, exist_ok=True)
        
        # 下载分词器
        logger.info(f"下载分词器: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True,
            local_files_only=False
        )
        
        # 如果指定了量化，则设置量化配置
        if quantization:
            logger.info(f"使用 {quantization} 位量化下载模型: {model_name}")
            if quantization == 4:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"量化位数必须是4或8，不能是: {quantization}")
            
            # 下载模型（只下载，不加载到内存）
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                local_files_only=False
            )
        else:
            # 如果不需要量化，则直接下载完整模型
            logger.info(f"下载完整模型: {model_name}")
            # 使用snapshot_download只下载文件而不加载模型
            snapshot_download(
                repo_id=model_name,
                local_dir=model_output_dir if model_output_dir else None,
                local_dir_use_symlinks=False
            )
        
        logger.info(f"模型 {model_name} 下载完成!")
        return True, model_name, None
    except Exception as e:
        logger.error(f"下载模型 {model_name} 时出错: {str(e)}")
        return False, model_name, str(e)


def download_from_config(config_file, max_workers=4):
    """
    从配置文件中批量下载模型
    
    参数:
        config_file: 配置文件路径
        max_workers: 最大并行下载数量
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    models_to_download = []
    for model_config in config['models']:
        model_name = model_config['name']
        quantization = model_config.get('quantization', None)
        output_dir = model_config.get('output_dir', None)
        
        models_to_download.append((model_name, quantization, output_dir))
    
    logger.info(f"将并行下载 {len(models_to_download)} 个模型，最大并行数: {max_workers}")
    
    # 使用ThreadPoolExecutor进行并行下载
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        future_to_model = {
            executor.submit(download_model, model_name, quantization, output_dir): model_name
            for model_name, quantization, output_dir in models_to_download
        }
        
        # 处理任务完成情况
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                success, name, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    logger.error(f"模型 {name} 下载失败: {error}")
            except Exception as e:
                failed += 1
                logger.error(f"处理模型 {model_name} 的下载结果时出错: {str(e)}")
    
    logger.info(f"下载完成! 成功: {successful}, 失败: {failed}")
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="批量下载Hugging Face模型")
    parser.add_argument("--model", type=str, help="要下载的模型名称")
    parser.add_argument("--quantization", type=int, choices=[4, 8], help="量化位数（4或8）")
    parser.add_argument("--output-dir", type=str, help="模型保存目录")
    parser.add_argument("--config", type=str, help="包含多个模型的配置文件路径")
    parser.add_argument("--workers", type=int, default=4, help="最大并行下载数量（默认为4）")
    
    args = parser.parse_args()
    
    if args.config:
        # 如果提供了配置文件，从配置文件中下载多个模型
        download_from_config(args.config, max_workers=args.workers)
    elif args.model:
        # 如果提供了模型名称，下载单个模型
        success, name, error = download_model(args.model, args.quantization, args.output_dir)
        if not success:
            logger.error(f"下载失败: {error}")
    else:
        # 如果没有提供模型名称或配置文件，打印帮助信息
        parser.print_help()


if __name__ == "__main__":
    main() 