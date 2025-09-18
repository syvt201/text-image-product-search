from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

class CLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name, attn_implementation="sdpa", use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=True, use_fast=True)
        
    def encode(self, text: str | list[str] = None, image: str | list[str] | Image.Image | list[Image.Image] = None, normalize: bool = True, as_numpy: bool = True):
        """
        Encode text and/or image to embeddings.
        Args:
            text (str or list of str): Text input(s) to encode.
            image (str or list of str or Image or list of Image): Path(s) to image file(s) or PIL Image(s) to encode.
            normalize (bool): Whether to normalize the embeddings to unit length.
            as_numpy (bool): Whether to return embeddings as numpy arrays. If False, returns as PyTorch tensors.
        Returns:
            tuple: (text_embeddings, image_embeddings)  # each has shape (batch_size, embedding_dim=512)
        """
        
        if text is None and image is None:
            raise ValueError("At least one of `text` or `image` must be provied")
        
        with torch.inference_mode():
            text_embeds, image_embeds = None, None
            
            if text is not None:
                inputs = self.processor.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
                text_embeds = self.model.get_text_features(**inputs)
            
            if image is not None:
                if not isinstance(image, list):
                    image = [image]
                
                pil_images = []
                for img in image:
                    if isinstance(img, str):
                        with Image.open(img) as im:
                            pil_images.append(im.convert("RGB").copy())
                    elif isinstance(img, Image.Image):
                        pil_images.append(img.convert("RGB") if img.mode != "RGB" else img)
                    else:
                        raise ValueError("`image` must be a file path, a PIL Image, or a list of either")
                    
                if len(pil_images) > 0:
                    inputs = self.processor.image_processor(pil_images, return_tensors='pt').to(self.device)
                    image_embeds = self.model.get_image_features(**inputs)
        
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
    
            