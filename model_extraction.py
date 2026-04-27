from model_base import ModelBase
from embedding_manager import EmbeddingManager
from tqdm import tqdm
import torch

class ModelExtraction(ModelBase):
    def __init__(self, model_nickname: str):
        super().__init__(model_nickname)

    def extract_embds(self, inputs: dict[str,list[str]], system_message: str=None, message: str=None) -> EmbeddingManager:
        embds_manager = EmbeddingManager(self.llm_cfg, message)
        embds_manager.layers = {
            label_name:[torch.zeros(len(texts), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)] for label_name, texts in inputs.items()
        }

        for label_name, texts in inputs.items():
            for i, txt in tqdm(enumerate(texts), desc="Extracting embeddings"):

                if self.tokenizer.chat_template is not None:
                    txt = self.apply_sft_template(instruction=txt, system_message=system_message)
                    input_ids = self.tokenizer.apply_chat_template(txt, add_generation_prompt=True, return_tensors="pt").to(self.device)
                    model_inputs = {"input_ids": input_ids}
                else:
                    model_inputs = self.tokenizer(txt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**model_inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states

                for j in range(self.llm_cfg.n_layer):
                    embds_manager.layers[label_name][j][i, :] = hidden_states[j][:, -1, :].detach().cpu()

        return embds_manager

