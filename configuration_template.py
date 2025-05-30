from os import path as osp

config_path = osp.abspath(osp.dirname(__file__))

class Config:
    def __init__(self):

        # insert settings
        
        self.model_id = "llava-hf/llava-next-72b-hf"
        self.refresh_model_type()
        
        self.quant = None # None, '4bit' , '8bit'
        self.model_cache_dir = 'PATH'
        
        self.patch_size = 100
        self.patch_padding = 10
        self.grid_padding = 50
        self.patch_pad_color = (255, 255, 255)
        self.grid_pad_color = (255, 255, 255)

        self.max_new_tokens = 150
        self.do_sample = False

        self.colorpatch_data = osp.join(config_path, 'colorpatch_data')
        self.colorgrid_data = osp.join(config_path, 'colorgrid_data')
        self.data_dir = osp.join(config_path, 'data')
        self.output_dir = osp.join(config_path, 'results')
        
        self.limit = None
        
        
    def refresh_model_type(self):

        # process settings

        assert self.model_id in [
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-v1.6-vicuna-13b-hf",
            "llava-hf/llava-v1.6-34b-hf",
            "llava-hf/llava-next-72b-hf",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct",
            "deepseek-ai/Janus-Pro-1B",
            "deepseek-ai/Janus-Pro-7B",
        ]
        id_prefix = self.model_id.split('/')[0]
        if id_prefix == 'llava-hf':
             self.model_type = 'LLaVa'
        elif id_prefix == 'Qwen':
            self.model_type = 'Qwen'
        elif id_prefix == 'deepseek-ai':
            self.model_type = 'Janus'
        else:
            raise NotImplementedError