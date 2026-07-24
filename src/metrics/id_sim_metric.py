import torch
import numpy as np
from copy import deepcopy
from src.metrics.base_metric import BaseMetric
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim
from collections import defaultdict
from src.datasets.data_utils import get_bigger_crop
import clip


class IDSimBest(BaseMetric):
    def __init__(self, id_embeds_pth, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.aligner = Aligner()
        self.id_embeds = torch.load(id_embeds_pth)
        
    def to_cpu(self):
        pass

    def to_cuda(self):
        pass

    def __call__(self, **batch):
        batch_bboxes, batch_embeds = self.aligner(batch['generated'])

        result = 0
        for face_bboxes, face_embeds in zip(batch_bboxes, batch_embeds):
            if face_embeds is None:
                continue
            score = self.calc_embeds_score(face_embeds, face_bboxes, batch["id"])
            result +=  score
        result = result / len(batch_embeds)
        return {"id_sim": result}

    def calc_embeds_score(self, embeds, bboxes, person_id):
        best_score = -np.inf
        for embed in embeds:
            best_score = max(cos_sim(embed, self.id_embeds[person_id]), best_score)
        return best_score


class IDSimMax(IDSimBest):
    def calc_embeds_score(self, embeds, bboxes, person_id):
        best_score = -np.inf
        pairs = list(zip(embeds, bboxes))
        pairs = sorted(pairs, key=lambda x: -(x[1][3] - x[1][1]) * (x[1][2] - x[1][0]))
        best_embed = pairs[0][0]
        best_score = cos_sim(best_embed, self.id_embeds[person_id])
        return best_score

class BoxSize(IDSimBest):
    def calc_embeds_score(self, embeds, bboxes, person_id):
        best_score = -np.inf
        best_bbox = None
        for embed, bbox in zip(embeds, bboxes):
            current_score = cos_sim(embed, self.id_embeds[person_id])
            if current_score > best_score:
                best_score = current_score
                best_bbox = bbox
        best_box_size = (best_bbox[3] - best_bbox[1]) * (best_bbox[2] - best_bbox[0])
        return best_box_size
    
    def __call__(self, **batch):
        res = super().__call__(**batch)
        return {"bbox_size": res["id_sim"]}


class FaceSimAll(IDSimBest):
    def __init__(self, clip_model_name, bins, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = [0] + bins + [np.inf]
        self.results = defaultdict(list)

        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()

    def to_cpu(self):
        self.clip_model = self.clip_model.to("cpu")

    def to_cuda(self):
        self.clip_model = self.clip_model.to(self.device)
    
    def calc_text_sim(self, faces, chosen_bins, prompt):
        preprecessed_faces = [self.preprocess(img) for img in faces]
        faces_images = torch.stack(preprecessed_faces).to(self.device)

        assert type(prompt) is str, type(prompt)
        tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self.device)

        logits_per_image, logits_per_text = self.clip_model(faces_images, tokenized_prompt)

        for logit, chosen_bin in zip(logits_per_image, chosen_bins):
            self.results["facial_ts"].append(logit.item())
            self.results[f"facial_ts_{chosen_bin}"].append(logit.item())


    def __call__(self, **batch):
        batch_bboxes, batch_embeds = self.aligner(batch['generated'])
        generated_faces = []
        chosen_bins = []

        for generated_img, face_bboxes, face_embeds in zip(batch['generated'], batch_bboxes, batch_embeds):
            if face_embeds is None:
                print("WARNING, CANT FIND FACE")
                self.results["id_sim"].append(0)
                self.results["facial_ts"].append(0)
                continue
            score, bbox_size, best_bbox = self.calc_embeds_score(face_embeds, face_bboxes, batch["id"])
            generated_faces.append(get_bigger_crop(deepcopy(generated_img), face_crop=best_bbox))

            self.results["id_sim"].append(score)
            self.results["bbox_size"].append(bbox_size)
            sq_size = np.sqrt(bbox_size)

            for bin_num in range(len(self.bins) - 1):
                if self.bins[bin_num] <= sq_size < self.bins[bin_num + 1]:
                    chosen_bin = self.bins[bin_num]
                    chosen_bins.append(chosen_bin)
                    self.results[f"id_sim_{chosen_bin}"].append(score)
                    break
        
        self.calc_text_sim(generated_faces, chosen_bins, batch['facial_caption'])

        
        if batch["is_last_batch"]:
            result = {k: np.mean(v) for k, v in self.results.items()}
            for bin in self.bins[:-1]:
                result[f"bin_size_{bin}"] = len(self.results[f"id_sim_{bin}"])
            self.results = defaultdict(list)
        else:
            result = {}
        return result

    def calc_embeds_score(self, embeds, bboxes, person_id):
        best_score = -np.inf
        best_bbox = None
        for embed, bbox in zip(embeds, bboxes):
            current_score = cos_sim(embed, self.id_embeds[person_id])
            if current_score > best_score:
                best_score = current_score
                best_bbox = bbox
        best_box_size = (best_bbox[3] - best_bbox[1]) * (best_bbox[2] - best_bbox[0])
        return best_score, best_box_size, best_bbox