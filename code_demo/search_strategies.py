from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import torch

# Import your existing core logic
try:
    from code_demo.CLIP_core import CLIPCoreModel
except ImportError:
    from CLIP_core import CLIPCoreModel

class SearchStrategy(ABC):
    """
    The Strategy Interface.
    Any model you want to use (CLIP, ResNet, BM25, etc.) must follow this contract.
    """
    @abstractmethod
    def predict(self, query_text: str, images: List[Image.Image]) -> torch.Tensor:
        """
        Takes a text query and a list of images, returning a tensor of probabilities 
        or scores corresponding to each image.
        """
        pass

class CLIPSearchStrategy(SearchStrategy):
    """
    Concrete Strategy: Wraps the OpenAI CLIP model.
    """
    def __init__(self, model_name: str):
        self.core_model = CLIPCoreModel(model_name)

    def predict(self, query_text: str, images: List[Image.Image]) -> torch.Tensor:
        # We adapt the specific API of CLIP_core to the generic API of the Strategy
        # Note: We pass [query_text] because CLIP_core expects a list for batch processing
        return self.core_model.predict_probabilities_based_on_text([query_text], images)