import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
import time
import math
import json
import functools    

class T5ForGenerate(pl.LightningModule):
    def __init__(self, t5_model, args, tokenizer):
        super(T5ForGenerate, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.training_step_outputs = []
        self.paser_error = 0
        self.pred_nums = 0
        self.label_nums = 0
        self.correct_nums = 0
        self.best_f1 = 0
        self.training_start_time = None
        # 加载预训练的 Llama 模型
        self.t5_model = t5_model
        self.test_results = []

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.t5_model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels, 
                                decoder_attention_mask=decoder_attention_mask)
        return outputs
    
    
    def on_train_start(self):
        # 记录训练开始时间
        self.training_start_time = time.time()

    def training_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids_prompt = batch['input_ids_prompt']
        attention_mask_prompt = batch['attention_mask_prompt']
        labels = batch['output_ids']
        decoder_attention_mask = batch['decoder_attention_mask']
        
        # 不带 prompt 的模型输出
        outputs_no_prompt = self(input_ids, attention_mask, labels, decoder_attention_mask)
        logits_no_prompt = outputs_no_prompt.logits
        # 带 prompt 的模型输出
        outputs_with_prompt = self(input_ids_prompt, attention_mask_prompt, labels, decoder_attention_mask)
        logits_with_prompt = outputs_with_prompt.logits
        
        # 计算每个样本的损失
        losses_no_prompt = self.compute_sample_losses(logits_no_prompt, labels)
        losses_with_prompt = self.compute_sample_losses(logits_with_prompt, labels)

        # 比较每个样本的损失，选择较大者作为最终损失
        final_losses = torch.where(losses_with_prompt > losses_no_prompt, losses_with_prompt, losses_no_prompt)

        # 计算 batch 的平均损失用于梯度更新
        loss = final_losses.mean()

        # 使用 self.log 打印损失
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.training_step_outputs.append(loss)

        return loss
        

    def compute_sample_losses(self, logits, labels):
        """
        计算每个样本的损失。
        logits: 模型的输出，形状为 (batch_size, seq_len, vocab_size)
        labels: 目标输出，形状为 (batch_size, seq_len)
        """
        # 使用交叉熵计算逐 token 的损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        # 将 logits 从 (batch_size, seq_len, vocab_size) 转换为 (batch_size * seq_len, vocab_size)
        # 将 labels 从 (batch_size, seq_len) 转换为 (batch_size * seq_len)
        loss_per_token = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        # 将逐 token 损失重塑回 (batch_size, seq_len)
        loss_per_token = loss_per_token.view(labels.size())

        # 按 token mask 求和，得到每个样本的总损失
        sample_losses = loss_per_token.sum(dim=1) / labels.ne(-100).sum(dim=1)
        return sample_losses

    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids_prompt = batch['input_ids_prompt']
        attention_mask_prompt = batch['attention_mask_prompt']
        triplets = batch['triplets']

        outputs = self.t5_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.args.max_new_length, do_sample=False,
                                        eos_token_id=self.tokenizer.eos_token_id)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        outputs_prompt = self.t5_model.generate(input_ids=input_ids_prompt, attention_mask=attention_mask_prompt, max_new_tokens=self.args.max_new_length, do_sample=False,
                                        eos_token_id=self.tokenizer.eos_token_id)
        responses_prompt = self.tokenizer.batch_decode(outputs_prompt, skip_special_tokens=True)

        for i in range(len(input_ids)):
            triplet = triplets[i]  
            response = responses[i]
            response_prompt = responses_prompt[i]
            pred_triplet = []
            pred_triplet_prompt = []

            response_segs = response.split("[SSEP]")
            for response_seg in response_segs:
                try:
                    aspect_text, opinion_text, sentiment_text = response_seg.split(",")
                    aspect = aspect_text.split(":")[1].strip()
                    opinion = opinion_text.split(":")[1].strip()
                    sentiment = sentiment_text.split(":")[1].strip()
                    pred_triplet.append([aspect.lower(), opinion.lower(), sentiment.lower()])
                except:
                    continue
            
            response_segs_prompt = response_prompt.split("[SSEP]")
            for response_seg_prompt in response_segs_prompt:
                try:
                    aspect_text, opinion_text, sentiment_text = response_seg_prompt.split(",")
                    aspect = aspect_text.split(":")[1].strip()
                    opinion = opinion_text.split(":")[1].strip()
                    sentiment = sentiment_text.split(":")[1].strip()
                    pred_triplet_prompt.append([aspect.lower(), opinion.lower(), sentiment.lower()])
                except:
                    continue

            triplet = [[item.lower() for item in t] for t in triplet]
            triplet = set(tuple(t) for t in triplet)
            pred = pred_triplet
            pred = set(tuple(t) for t in pred)

            pred_prompt = pred_triplet_prompt
            pred_prompt = set(tuple(t) for t in pred_prompt)

            intersection = pred & pred_prompt
            pred_final = tuple(intersection if intersection else pred_prompt)   

            self.pred_nums += len(pred_final)
            self.label_nums += len(triplet) 

            for t in pred_final:
                if t in triplet:
                    self.correct_nums += 1

    def on_validation_epoch_end(self):
        micro_p = float(self.correct_nums/self.pred_nums) if self.pred_nums else 0 
        micro_r = float(self.correct_nums/self.label_nums) if self.label_nums else 0 
        micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0   

        # 重置统计变量
        self.pred_nums = 0
        self.label_nums = 0
        self.correct_nums = 0
        self.best_f1 = max(self.best_f1, (micro_f1*100))

        # 使用 self.log() 记录验证损失和PRF1指标
        self.log('val_p', micro_p, on_epoch=True, prog_bar=False)
        self.log('val_r', micro_r, on_epoch=True, prog_bar=False)
        self.log('val_f1', micro_f1, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        train_epoch_loss = torch.stack(self.training_step_outputs).mean()
        val_metrics = self.trainer.callback_metrics  # 获取所有指标
        # 获取验证集PRF1
        val_p = val_metrics.get('val_p', 0)
        val_r = val_metrics.get('val_r', 0)
        val_f1 = val_metrics.get('val_f1', 0)

        # 输出验证指标
        print('Epoch: {}\nTrain epoch loss: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Best F1: {:.2f}'.format(epoch, train_epoch_loss, val_p * 100, val_r * 100, val_f1 * 100, self.best_f1))
        self.training_step_outputs.clear()  # free memory

    def on_train_end(self):
        # 获取训练时长
        if self.training_start_time is not None:
            total_training_time = time.time() - self.training_start_time
        print(f"Best F1: {self.best_f1:.2f}  Training completed in: {total_training_time:.2f} seconds")

    def test_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids_prompt = batch['input_ids_prompt']
        attention_mask_prompt = batch['attention_mask_prompt']
        triplets = batch['triplets']

        outputs = self.t5_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.args.max_new_length, do_sample=False,
                                         eos_token_id=self.tokenizer.eos_token_id)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        outputs_prompt = self.t5_model.generate(input_ids=input_ids_prompt, attention_mask=attention_mask_prompt, max_new_tokens=self.args.max_new_length, do_sample=False,
                                        eos_token_id=self.tokenizer.eos_token_id)
        responses_prompt = self.tokenizer.batch_decode(outputs_prompt, skip_special_tokens=True)

        for i in range(len(input_ids)):
            triplet = triplets[i]  
            response = responses[i]
            response_prompt = responses_prompt[i]
            pred_triplet = []
            pred_triplet_prompt = []
            
            response_segs = response.split("[SSEP]")
            for response_seg in response_segs:
                try:
                    aspect_text, opinion_text, sentiment_text = response_seg.split(",")
                    aspect = aspect_text.split(":")[1].strip()
                    opinion = opinion_text.split(":")[1].strip()
                    sentiment = sentiment_text.split(":")[1].strip()
                    pred_triplet.append([aspect.lower(), opinion.lower(), sentiment.lower()])
                except:
                    self.paser_error += 1 
                    continue
            
            response_segs_prompt = response_prompt.split("[SSEP]")
            for response_seg_prompt in response_segs_prompt:
                try:
                    aspect_text, opinion_text, sentiment_text = response_seg_prompt.split(",")
                    aspect = aspect_text.split(":")[1].strip()
                    opinion = opinion_text.split(":")[1].strip()
                    sentiment = sentiment_text.split(":")[1].strip()
                    pred_triplet_prompt.append([aspect.lower(), opinion.lower(), sentiment.lower()])
                except:
                    self.paser_error += 1 
                    continue

            triplet = [[item.lower() for item in t] for t in triplet]
         
            triplet = set(tuple(t) for t in triplet)
            pred = pred_triplet
            pred = set(tuple(t) for t in pred)

            pred_prompt = pred_triplet_prompt
            pred_prompt = set(tuple(t) for t in pred_prompt)

            intersection = pred & pred_prompt
            pred_final = tuple(intersection if intersection else pred_prompt)   
            
            self.test_results.append({
                "predict       ": str(pred_triplet),
                "predict_prompt": str(pred_triplet_prompt),
                "predict_final ": str(pred_final),
                "label         ": str(triplet)
            })

            self.pred_nums += len(pred_final)
            self.label_nums += len(triplet) 

            for t in pred_final:
                if t in triplet:
                    self.correct_nums += 1

    def on_test_epoch_end(self):
        micro_p = float(self.correct_nums/self.pred_nums) if self.pred_nums else 0 
        micro_r = float(self.correct_nums/self.label_nums) if self.label_nums else 0 
        micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0   

        # 输出训练损失和验证指标
        print('Dataset: {}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(self.args.data_name, micro_p * 100, micro_r * 100, micro_f1 * 100))
        print('无法解析的response数量：{}'.format(self.paser_error))
        with open('mydataset/ASTE_G/' + self.args.data_name + '/predict_label.json', 'w', encoding='utf8') as f:
            json.dump(self.test_results, f, ensure_ascii=False)


    def configure_optimizers(self):

        t5_params_decay = []
        t5_params_no_decay = []
        t5_params_names = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                if param.ndim == 1 or "bias" in name.lower():
                    t5_params_no_decay.append(param)
                else:
                    t5_params_decay.append(param)

        print('t5训练参数包括：')
        for name in t5_params_names:
            print(name)
        param_groups = [
            # llama 模型需要 weight decay 的参数
            {
                "params": t5_params_decay,
                "lr": self.args.t5_lr,
                "weight_decay": self.args.t5_l2
            },
            # llama 模型不需要 weight decay 的参数（如偏置、LayerNorm）
            {
                "params": t5_params_no_decay,
                "lr": self.args.t5_lr,
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer