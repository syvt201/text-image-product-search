from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

class CLIPEncoder:
    def __init__(self,model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name, attn_implementation="sdpa", use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True)
        
    def encode(self, text=None, image=None, normalize=True, as_numpy=True):
        with torch.inference_mode():
            if text is not None:
                inputs = self.processor.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
                text_embeds = self.model.get_text_features(**inputs)
            else:
                text_embeds = None
            
            if image is not None:
                inputs = self.processor.image_processor(image, return_tensors='pt').to(self.device)
                image_embeds = self.model.get_image_features(**inputs)
            else:
                image_embeds = None
        
        # Normalize
        if normalize:
            if text_embeds is not None:
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            if image_embeds is not None:
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            
        if as_numpy:
            if text_embeds is not None:
                text_embeds = text_embeds.detach().cpu().numpy()

            if image_embeds is not None:
                image_embeds = image_embeds.detach().cpu().numpy()
                
        
        return text_embeds, image_embeds
    
            