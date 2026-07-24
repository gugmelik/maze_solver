import torch
from src.metrics.base_metric import BaseMetric
from torchvision import transforms

class DinoMetric(BaseMetric):
    def __init__(self, model_name, model_path, source, to_norm, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.to_norm = to_norm
        self.model = torch.hub.load(model_path, model_name, source=source)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to(self.device)

    def get_features(self, images):
        preprecessed = [self.preprocess(img) for img in images]
        preprecessed = torch.stack(preprecessed).to(self.device)
        features = self.model(preprecessed)
        if self.to_norm:
            features = features / features.clone().norm(dim=-1, keepdim=True)
        return features

    def __call__(self, **batch):
        concept = batch['concept']
        generated = batch['generated']
        assert type(generated) is list, type(generated) 
        assert type(concept) is list, type(concept)

        gen_features = self.get_features(generated)
        concept_features = self.get_features(concept)

        similarity_matrix = gen_features @ concept_features.T
        
        result = {"dino_score": similarity_matrix.mean().item()}
        return result
