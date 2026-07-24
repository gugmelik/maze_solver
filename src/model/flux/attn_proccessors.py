import torch
import torch.nn.functional as F
from torch import nn
from copy import deepcopy

from typing import Optional
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
from src.utils.model_utils import freeze_params
