from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_config import cfg, get_cfg
import torch

class ModelBase:
    def __init__(self, model_nickname: str):
        self.llm_cfg = get_cfg(model_nickname)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(self.llm_cfg.model_name, torch_dtype=torch.float16).to(self.device) #l
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_cfg.model_name, torch_dtype=torch.float16) #l
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def apply_sft_template(self, instruction, system_message=None):
        if system_message is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            
        return messages
    
    def apply_inst_template(self, text):
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        return messages