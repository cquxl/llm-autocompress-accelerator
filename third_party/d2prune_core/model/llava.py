
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch



class LLaVA:
    def __init__(self, model_path, device_type, model_name=None):
        self.model_path = model_path
        self.model_name = model_name.lower() if model_name else model_path.split("/")[-1]
        self.device_type = device_type

    def load_model(self, seq_len=None):
        model = LlavaForConditionalGeneration.from_pretrained(self.model_path,
                                                              device_map=self.device_type,
                                                              torch_dtype=torch.float16)
        # model.seq_len = model.config.max_position_embeddings
        model.seq_len = model.config.text_config.max_position_embeddings if not seq_len else seq_len# 4096
        model.language_model.seq_len = model.config.text_config.max_position_embeddings if not seq_len else seq_len# 4096
        return model

    def load_processer(self):
        processor = AutoProcessor.from_pretrained(self.model_path)
        return processor