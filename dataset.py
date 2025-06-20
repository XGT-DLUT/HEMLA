import os
import json
import ast
import torch
from torch.utils.data import Dataset

class ASTE_dataset(Dataset):
    def __init__(self, data_name, data_type, tokenizer):
        data_path = os.path.join('mydataset/ASTE_G', data_name, 'gpt4o_'+data_type+'.jsonl')
        self.tokenizer = tokenizer
        self.data_type = data_type
        # 读取 JSON 数据
        with open(data_path, 'r') as f:
            self.datas = json.load(f)
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        """
        获取一个样本的数据
        :param idx: 数据索引
        :return: 一个字典，包含输入和目标文本的张量
        """
        data = self.datas[idx]
        text = data['input']
        label = data['label']
        pseudo_label = data['response']
        triplets = json.loads(label)
        # 重新构造label为自然语言格式
        new_label = []
        for triplet in triplets:
            aspect, opinion, sentiment = triplet
            new_label.append('aspect: '+ aspect + ', opinion: ' + opinion + ', sentiment: ' + sentiment)
        new_label = ' [SSEP] '.join(new_label)

        new_pseudo_label = []
        try:
            pseudo_label = ast.literal_eval(pseudo_label)
            for triplet in pseudo_label:
                aspect, opinion, sentiment = triplet
                new_pseudo_label.append('aspect: '+ aspect + ', opinion: ' + opinion + ', sentiment: ' + sentiment)
            new_pseudo_label = ' [SSEP] '.join(new_pseudo_label)
        except:
            new_pseudo_label = ''

        new_text = new_pseudo_label  + ' [SEP] ' + text

        inputs= self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        input_attention_mask = inputs.attention_mask

        inputs_prompt= self.tokenizer(new_text, return_tensors="pt")
        input_ids_prompt = inputs_prompt.input_ids
        input_attention_mask_prompt = inputs_prompt.attention_mask

        outputs= self.tokenizer(new_label, return_tensors="pt")
        output_ids = outputs.input_ids
        output_attention_mask = outputs.attention_mask

        prompt_len = self.tokenizer(new_pseudo_label, return_tensors="pt", add_special_tokens=False).input_ids.size(1)

        input = {
            'input_ids': input_ids,
            'input_attention_mask': input_attention_mask,
            'input_ids_prompt': input_ids_prompt,
            'input_attention_mask_prompt': input_attention_mask_prompt,
            'output_ids': output_ids,
            'output_attention_mask': output_attention_mask,
            'triplets': triplets,
            'prompt_len': prompt_len
        }

        return input
    
class my_collate_fn(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        # 对输入序列padding
        input_lengths = [x['input_ids'].shape[1] for x in batch]
        max_input_len = max(input_lengths)

        input_lengths_prompt = [x['input_ids_prompt'].shape[1] for x in batch]
        max_input_len_prompt = max(input_lengths_prompt)
        # 对 label 序列padding
        output_lengths = [x['output_ids'].shape[1] for x in batch]
        max_output_len = max(output_lengths)
        
        # 准备存放处理后的数据列表
        input_ids_batch = []
        input_attention_mask_batch = []
        input_ids_batch_prompt = []
        input_attention_mask_batch_prompt = []
        output_ids_batch = []
        output_attention_mask_batch = []
        triplets = []
        prompt_lens = []

        for x in batch:
            prompt_lens.append(x['prompt_len'])
            # x['input_ids'] 等是 (1, seq_len) 的张量, 需要 squeeze
            input_ids = x['input_ids'].squeeze(0)  # shape: [seq_len]
            input_attention_mask = x['input_attention_mask'].squeeze(0) # shape: [seq_len]

            input_ids_prompt = x['input_ids_prompt'].squeeze(0)  # shape: [seq_len]
            input_attention_mask_prompt = x['input_attention_mask_prompt'].squeeze(0) # shape: [seq_len]

            output_ids = x['output_ids'].squeeze(0) # shape: [seq_len]
            output_attention_mask = x['output_attention_mask'].squeeze(0) # shape: [seq_len]

            # 无prompt输入pad
            pad_length_input = max_input_len - len(input_ids)
            if pad_length_input > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_length_input,), self.tokenizer.pad_token_id, dtype=torch.long)]
                )
                input_attention_mask = torch.cat(
                    [input_attention_mask, torch.zeros(pad_length_input, dtype=torch.long)]
                )
            
            # prompt输入pad
            pad_length_input_prompt = max_input_len_prompt - len(input_ids_prompt)
            if pad_length_input_prompt > 0:
                input_ids_prompt = torch.cat(
                    [input_ids_prompt, torch.full((pad_length_input_prompt,), self.tokenizer.pad_token_id, dtype=torch.long)]
                )
                input_attention_mask_prompt = torch.cat(
                    [input_attention_mask_prompt, torch.zeros(pad_length_input_prompt, dtype=torch.long)]
                )

            pad_length_output = max_output_len - len(output_ids)
            if pad_length_output > 0:
                output_ids = torch.cat(
                    [output_ids, torch.full((pad_length_output,), -100, dtype=torch.long)]
                )
                output_attention_mask = torch.cat(
                    [output_attention_mask, torch.zeros(pad_length_output, dtype=torch.long)]
                )
            
            
            input_ids_batch.append(input_ids.unsqueeze(0))  # (1, seq_len)
            input_attention_mask_batch.append(input_attention_mask.unsqueeze(0))
            input_ids_batch_prompt.append(input_ids_prompt.unsqueeze(0))  # (1, seq_len)
            input_attention_mask_batch_prompt.append(input_attention_mask_prompt.unsqueeze(0))
            output_ids_batch.append(output_ids.unsqueeze(0))
            output_attention_mask_batch.append(output_attention_mask.unsqueeze(0))
            triplets.append(x['triplets'])

        # 将list中单个tensor堆叠为batch tensor
        input_ids_batch = torch.cat(input_ids_batch, dim=0)  # (batch_size, seq_len)
        input_attention_mask_batch = torch.cat(input_attention_mask_batch, dim=0)  # (batch_size, seq_len)
        input_ids_batch_prompt = torch.cat(input_ids_batch_prompt, dim=0)  # (batch_size, seq_len)
        input_attention_mask_batch_prompt = torch.cat(input_attention_mask_batch_prompt, dim=0)  # (batch_size, seq_len)
        output_ids_batch = torch.cat(output_ids_batch, dim=0)  # (batch_size, seq_len)
        output_attention_mask_batch = torch.cat(output_attention_mask_batch, dim=0)  # (batch_size, seq_len)

        # 返回的数据字典
        return {
            'input_ids': input_ids_batch,
            'attention_mask': input_attention_mask_batch,
            'input_ids_prompt': input_ids_batch_prompt,
            'attention_mask_prompt': input_attention_mask_batch_prompt,
            'output_ids': output_ids_batch,
            'decoder_attention_mask': output_attention_mask_batch,
            'triplets': triplets,
            'prompt_lens': prompt_lens
        }