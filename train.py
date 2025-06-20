import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizerFast, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from model import T5ForGenerate
from dataset import ASTE_dataset, my_collate_fn
import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import logging


# 禁用 transformers和pytorch lightning的日志
logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def print_author_info():
    author_info = """
    ***************************************************
    *               Author Information                *
    ***************************************************
    * Name         : Guangtao Xu                      *
    * Affiliation  : Dalian University of Technology  *
    * Contact      : Guangtao_xu@mail.dlut.edu.cn     *
    * GitHub       : https://github.com/XGT-DLUT      *
    * Paper Title  :                                  *
    ***************************************************
    """
    print(author_info)

def set_random_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)

    # 如果使用 GPU，设置每个 GPU 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU

    # 确保 PyTorch 的操作是可重复的（可选，可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/sda/xuguangtao/llm/flan-t5-base")
    parser.add_argument('--out_path', type=str, default="checkpoints/")
    parser.add_argument('--data_name', type=str, default="16res", help="[14lap, 14res, 15res, 16res]")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--max_new_length", type=int, default=128)
    parser.add_argument("--t5_lr", type=float, default=8e-5)
    parser.add_argument("--t5_l2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 打印作者信息
    print_author_info()

    # 设置随机种子
    set_random_seed(args.seed)

    # 加载切词器
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    tokenizer.add_tokens(["[SEP]"])
    tokenizer.add_tokens(["[SSEP]"])
    # 加载数据
    print('加载训练和验证数据...')
    train_dataset = ASTE_dataset(args.data_name, 'train', tokenizer)
    val_dataset = ASTE_dataset(args.data_name, 'dev', tokenizer)
    args.num_training_samples = len(train_dataset)
    # 数据并行
    collate_fn = my_collate_fn(tokenizer, args)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn, num_workers=10, pin_memory=True)

    # 加载base模型
    print('加载base模型...')
    t5_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    t5_model.resize_token_embeddings(len(tokenizer))

    # 添加训练验证测试函数等
    model = T5ForGenerate(t5_model, args, tokenizer)
    model.train()
    # 设置检查点回调函数
    checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",  # 根据验证集的F1值保存最优模型
    mode="max",  # 当指标最大时保存
    save_top_k=1,  # 只保存最好的一个模型
    dirpath=args.out_path + args.data_name,  # 设置保存检查点的目录
    filename="best_model",  # 模型保存的文件名，包含 epoch 信息
    save_weights_only=True        # 只保存权重
    )

    # 创建Trainer
    trainer = pl.Trainer(
    enable_progress_bar=True,
    logger=False,
    accumulate_grad_batches=args.accumulate_grad_batches,
    max_epochs=args.max_epochs, 
    num_nodes=1,  # 使用一个GPU
    enable_checkpointing=True,
    callbacks=[checkpoint_callback], # 添加检查点回调
    val_check_interval=1.0,
    num_sanity_val_steps=0,   # 直接开始训练，而非先走验证集测试一下验证步骤是否正常
    gradient_clip_val=1.0       # 设置最大梯度值，防止梯度爆炸使模型异常
    )

    # 训练模型
    print('开始训练...')
    trainer.fit(model, train_loader, val_loader)

    # 加载最优模型
    best_model = T5ForGenerate.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    t5_model = t5_model,
    args=args,
    tokenizer = tokenizer,
    strict=False        # 忽略checkpoints中不存在的网络层
    )

    # 在测试集上进行测试
    print('开始测试...')
    test_dataset = ASTE_dataset(args.data_name, 'test', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn, num_workers=10, pin_memory=True)
    trainer.test(best_model, test_loader)
    