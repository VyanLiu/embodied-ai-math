from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import torch

from transformers import CLIPModel, CLIPProcessor
import torch

class CLIPCoreModel:
    def __init__(self, model_name):
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def predict_probabilities_based_on_images(self, image, labels):
        """
        Scenario 1: Image Classification / Retrieval
        Input: 1 Image, N Text Labels
        Output: Probability distribution over the N labels.
        """
        if not image:
            raise ValueError("Input test images cannot be empty!")
        if not labels:
            raise ValueError("Candidate labels cannot be empty!")

        inputs = self.processor(images=image, text=labels, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        image_embeddings = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = (image_embeddings @ text_embeddings.T) * logit_scale
        
        probabilities = logits_per_image.softmax(dim=1)

        return probabilities[0]

    def predict_probabilities_based_on_text(self, text, images):
        """
        Scenario 2: Image Search
        Input: 1 Text Query, N Candidate Images
        Output: Probability distribution over the N images.
        """
        if not text:
            raise ValueError("Input test text cannot be empty!")
        if not images:
            raise ValueError("Output images cannot be empty!")

        inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        image_embeddings = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_text = (text_embeddings @ image_embeddings.T) * logit_scale
        
        probabilities = logits_per_text.softmax(dim=1)

        return probabilities[0]

class SearchStrategy(ABC):
    """
    The Strategy Interface.
    Any model you want to use (CLIP, ResNet, BM25, etc.) must follow this rule.
    """
    @abstractmethod
    def predict(self, query_text: str, images: List[Image.Image]) -> torch.Tensor:
        pass

class CLIPSearchStrategy(SearchStrategy):
    """
    Concrete Strategy: Uses OpenAI's CLIP model.
    """
    def __init__(self, model_name: str):
        self.core_model = CLIPCoreModel(model_name)

    def predict(self, query_text: str, images: List[Image.Image]) -> torch.Tensor:
        # We adapt the specific API of CLIP_core to the generic API of the Strategy
        # Note: We pass [query_text] because CLIP_core expects a list
        return self.core_model.predict_probabilities_based_on_text([query_text], images)