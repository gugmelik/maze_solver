import torch
import numpy as np
from copy import deepcopy
from src.metrics.base_metric import BaseMetric
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim
from src.datasets.data_utils import get_bigger_crop
import clip

class TextSimMetric(BaseMetric):
    def __init__(self, model_name, device, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to(self.device)

    def __call__(self, **batch):
        prompt = batch['prompt']
        tokenized_prompt = clip.tokenize([prompt]).to(self.device)
        generated = batch['generated']
        assert type(generated) is list, type(generated) 
        assert type(prompt) is str, type(prompt)

            
        preprecessed = [self.preprocess(img) for img in generated]
        images = torch.stack(preprecessed).to(self.device) 
        _, logits_per_text = self.model(images, tokenized_prompt)
        result = {"text_sim": logits_per_text.mean().item()}
        return result


class TrippleTextSimMetric(TextSimMetric):
    def __init__(self, id_embeds_pth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligner = Aligner()
        self.id_embeds = torch.load(id_embeds_pth)
    
    def calc_best_bbox(self, embeds, bboxes, person_id):
        best_score = -np.inf
        best_bbox = None
        for bbox, embed in zip(bboxes, embeds):
            score = cos_sim(embed, self.id_embeds[person_id])
            if score > best_score:
                best_score = score
                best_bbox = bbox
                
        return best_bbox
    
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        results = {}
        
        generated = batch['generated']
        
        if "facial_caption" in batch:
            batch_bboxes, batch_embeds = self.aligner(deepcopy(generated))
            generated_faces = []
            for orig_img, faces_bboxes, faces_embeds in zip(generated, batch_bboxes, batch_embeds):
                if faces_embeds is None:
                    continue
                best_bbox = self.calc_best_bbox(faces_embeds, faces_bboxes, batch["id"])
                        
                if best_bbox is not None:
                    generated_faces.append(get_bigger_crop(deepcopy(orig_img), face_crop=best_bbox))
                else:
                    print("Cant find bbox for image")
    

        assert type(generated) is list, type(generated) # list of PIL.Image
        preprecessed = [self.preprocess(img) for img in generated]
        images = torch.stack(preprecessed).to(self.device) 

        if "facial_caption" in batch:
            preprecessed_faces = [self.preprocess(img) for img in generated_faces]
            if len(preprecessed_faces) == 0:
                results["facial_ts"] = 0
            else:
                faces_images = torch.stack(preprecessed_faces).to(self.device) 
        
                
                prompt = batch['facial_caption']
                assert type(prompt) is str, type(prompt)
                tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self.device)
        
                logits_per_image, logits_per_text = self.model(faces_images, tokenized_prompt)
                results["facial_ts"] = logits_per_text.mean().item()


        if "pose_caption" in batch:
            prompt = batch['pose_caption']
            assert type(prompt) is str, type(prompt)
            tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self.device)
            
            logits_per_image, logits_per_text = self.model(images, tokenized_prompt)
            results["pose_ts"] = logits_per_text.mean().item()


        if "background_caption" in batch:
            prompt = batch['background_caption']
            assert type(prompt) is str, type(prompt)
            tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self.device)
            
            logits_per_image, logits_per_text = self.model(images, tokenized_prompt)
            results["backgraund_ts"] = logits_per_text.mean().item()
        
        return results
        

class DoubleTextSimMetric(TextSimMetric):
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        results = {}
        
        generated = batch['generated']
        
        assert type(generated) is list, type(generated) # list of PIL.Image
        preprecessed = [self.preprocess(img) for img in generated]
        images = torch.stack(preprecessed).to(self.device) 

        if "pose_caption" in batch:
            prompt = batch['pose_caption']
            assert type(prompt) is str, type(prompt)
            tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self.device)
            
            logits_per_image, logits_per_text = self.model(images, tokenized_prompt)
            results["pose_ts"] = logits_per_text.mean().item()


        if "background_caption" in batch:
            prompt = batch['background_caption']
            assert type(prompt) is str, type(prompt)
            tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self.device)
            
            logits_per_image, logits_per_text = self.model(images, tokenized_prompt)
            results["backgraund_ts"] = logits_per_text.mean().item()
        
        return results