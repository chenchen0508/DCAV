__cfg = {
    'llama2-7b': {
        'model_nickname': 'llama2-7b',
        'model_name': 'meta-llama/Llama-2-7b-chat-hf', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'llama3-8b': {
        'model_nickname': 'llama3-8b',
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'llama3-8b-base': {
        'model_nickname': 'llama3-8b-base',
        'model_name': 'meta-llama/Llama-3.1-8B', 
        'n_layer': 32, 
        'n_dimension': 4096
    },
    'gemma-7b-base':{
        'model_nickname': 'gemma-7b-base',
        'model_name': 'google/gemma-7b', 
        'n_layer': 28, 
        'n_dimension': 3072
    },
     'qwen-7b-base': {
        'model_nickname': 'qwen-7b-base',
        'model_name': 'Qwen/Qwen2.5-7B', 
        'n_layer': 28, 
        'n_dimension': 3584
    },
    'mistral-7b': {
        'model_nickname': 'mistral-7b',
        'model_name': 'mistralai/Mistral-7B-Instruct-v0.1', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'mistral-7b-base': {
        'model_nickname': 'mistral-7b-base',
        'model_name': 'mistralai/Mistral-7B-v0.1', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
}

class cfg:
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)

def get_cfg(model_nickname: str):
    assert model_nickname in __cfg, f"{model_nickname} not found in config"
    return cfg(__cfg[model_nickname])