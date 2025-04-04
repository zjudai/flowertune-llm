"""flwr-nlp: A Flower / FlowerTune app."""
# flwr-nlp: 基于Flower框架的联邦学习自然语言处理应用

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
import os

FDS = None  # 缓存联邦数据集对象，避免重复初始化


def formatting_prompts_func(example):
    """Construct prompts."""
    # 构建提示函数：将指令和响应格式化为模型可接受的输入格式
    output_texts = []
    # 构建标准的Alpaca格式提示
    # (https://github.com/tatsu-lab/stanford_alpaca#data-release)
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for i in range(len(example["instruction"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n"
            f"### Response: {example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    # 获取分词器、数据整理器和提示格式化函数
    # 参数：
    #   model_name: 预训练模型名称
    # 返回：
    #   tokenizer: 分词器对象
    #   data_collator: 数据整理器对象
    #   formatting_prompts_func: 提示格式化函数
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记为结束标记
    response_template_with_context = "\n### Response:"  # alpaca响应标签
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]  # 获取响应模板对应的token ID
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )  # 创建仅完成任务的数据整理器

    return tokenizer, data_collator, formatting_prompts_func


def formatting(dataset):
    """Format dataset."""
    # 将输入与指令合并的格式化函数
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets."""
    # 根据不同的LLM任务重新格式化数据集
    # 参数：
    #   dataset: 原始数据集
    #   llm_task: LLM任务类型，如"finance", "code", "medical"等
    # 返回：
    #   重新格式化后的数据集
    dataset = dataset.rename_column("output", "response")  # 将"output"列重命名为"response"
    if llm_task in ["finance", "code"]:
        dataset = dataset.map(formatting, remove_columns=["input"])  # 针对金融和代码任务的特殊处理
    if llm_task == "medical":
        dataset = dataset.remove_columns(["instruction"])  # 移除原始指令列
        dataset = dataset.rename_column("input", "instruction")  # 将输入列重命名为指令列
    return dataset


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # 加载特定分区的数据
    # 参数：
    #   partition_id: 分区ID
    #   num_partitions: 总分区数
    #   dataset_name: 数据集名称
    # 返回：
    #   客户端训练数据集
    global FDS  # 使用全局变量FDS
    if FDS is None:
        # 仅在首次调用时初始化FederatedDataset
        partitioner = IidPartitioner(num_partitions=num_partitions)  # 创建IID分区器
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},  # 为训练集应用分区器
        )
    client_trainset = FDS.load_partition(partition_id, "train")  # 加载指定分区的训练数据
    
    # Get task type from environment variable, default to "generalnlp" if not set
    llm_task = os.environ.get("FLWR_LLM_TASK", "generalnlp")
    print(f"Using LLM task type: {llm_task} for dataset formatting")
    
    client_trainset = reformat(client_trainset, llm_task=llm_task)  # 使用环境变量中的任务类型
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    # 递归替换字典键中的特定字符
    # 参数：
    #   input_dict: 输入字典
    #   match: 要替换的字符，默认为"-"
    #   target: 替换成的字符，默认为"_"
    # 返回：
    #   处理后的新字典
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)  # 替换键中的特定字符
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)  # 递归处理嵌套字典
        else:
            new_dict[new_key] = value  # 保留非字典值不变
    return new_dict
