from transformers import DeiTForImageClassification, DeiTConfig, \
    AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification


class DeiT:
    def __init__(self, model_path, device_type, model_name=None):
        self.model_path = model_path
        self.model_name = model_name.lower() if model_name else model_path.split("/")[-1]
        self.device_type = device_type

    def load_model(self, seq_len=None):
        model = AutoModelForImageClassification.from_pretrained(self.model_path,
                                                                     device_map=self.device_type,
                                                                     torch_dtype='auto',
                                                                     low_cpu_mem_usage=True)
        # model.seq_len = model.config.max_position_embeddings
        model.seq_len = seq_len
        return model

    def load_processer(self):
        processor = AutoImageProcessor.from_pretrained(self.model_path)
        return processor