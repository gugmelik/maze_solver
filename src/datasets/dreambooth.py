from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import IMG_EXTENTIONS


class DreamBoothTrainDataset(BaseDataset):
    def __init__(self, data_path, placeholder_token, class_name, *args, **kwargs):
        self.placeholder_token = placeholder_token
        self.class_name = class_name
        index = []
        
        for file_path in Path(data_path).iterdir():
            if file_path.suffix in IMG_EXTENTIONS:
                index.append(file_path)
        index = index  * 100
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        file_path = self._index[ind]
        
        instance_data  = {
            "original_sizes": [1024, 1024],
            "crop_top_lefts": [0, 0],
            "target_sizes": [1024, 1024],
            "pixel_values": self.load_object(file_path),
            "prompt": f"a photo of a {self.placeholder_token} {self.class_name}",
            "class_prompt": f"a photo of a {self.placeholder_token} {self.class_name}",
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data


class DreamBoothLivingSmall(BaseDataset):
    def __init__(self, data_path, placeholder_token, class_name, *args, **kwargs):
        self.placeholder_token = placeholder_token
        self.class_name = class_name
        index = [
            "a photo of a {0}",
            "a {0} in a purple wizard outfit",
            "a {0} in a watercolor painting style",
            "a {0} with the Eiffel Tower in the background",
            "a {0} riding a bike",
        ]

        self.images = []
        for file_path in Path(data_path).iterdir():
            if file_path.suffix in IMG_EXTENTIONS:
                self.images.append(self.load_object(file_path))
            
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        prompt = self._index[ind]
        instance_data  = {
            "concept": self.images,
            "prompt": prompt.format(self.placeholder_token + " " + self.class_name),
            "class_prompt": prompt.format(self.placeholder_token)
        }
        return instance_data
                
            
        