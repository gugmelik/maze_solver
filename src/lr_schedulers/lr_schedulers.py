from torch.optim.lr_scheduler import LambdaLR

class CustomLinearLR(LambdaLR):
    def __init__(self, warmup_steps, *args, **kwargs):
        super().__init__(
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0), 
            *args, 
            **kwargs
        )