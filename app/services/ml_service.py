from PIL import Image
from io import BytesIO 
import numpy as np
import time
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

class EmotionModel:

    _EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]

    def __init__(self):
        self.is_loaded = False 
        self.model = None
        self.processor = None
        self.input_size = (224, 224)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
       
    @staticmethod
    def softmax(x):
        x_shifted = x - np.max(x)  
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)   

    def load_model(self):
            
        try:    
         # Загружаем процессор (предобработчик)
            self.processor = ViTImageProcessor.from_pretrained("trpakov/vit-face-expression")
            
            # Загружаем модель
            self.model = ViTForImageClassification.from_pretrained("trpakov/vit-face-expression")
            
            # Переносим модель на GPU, если доступен
            self.model.to(self.device)
            
            # Переводим в режим инференса (отключаем dropout и т.д.)
            self.model.eval()

            print("Модель успешно загружена!")
            
            self.is_loaded = True
        except Exception as e:
            raise 


    def _validate_image(self, image_bytes: bytes) -> Image.Image:
        
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify()
            img = Image.open(BytesIO(image_bytes))
            img.load()
            return img
        except Exception as e:
            raise ValueError(f"Ошибка {e}")    


    def _preprocess(self, image : Image.Image) -> torch.tensor:

        image = image.convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
    
        # Извлекаем тензор пикселей
        pixel_values = inputs["pixel_values"]
        
        # Переносим на устройство (GPU/CPU)
        pixel_values = pixel_values.to(self.device)
        
        return pixel_values  


    def _postprocess(self, logits: torch.Tensor)-> tuple[str, float, dict[str, float]]:
        logits = logits.detach().cpu().numpy()
        logits = logits.squeeze()
        probabilities = EmotionModel.softmax(logits)
        predicted_idx = int(np.argmax(probabilities))
        emotion_predicted = self._EMOTIONS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        all_probabilities = {emotion : float(prob) for emotion, prob in zip(self._EMOTIONS, probabilities)} 
        return (emotion_predicted, confidence, all_probabilities)


    def predict(self, image_bytes: bytes) -> tuple[str, float, dict[str, float]]:

        if not self.is_loaded:
            raise RuntimeError("Модель не загружена!")

        img = self._validate_image(image_bytes = image_bytes)
        data_for_image = self._preprocess(img)
        with torch.no_grad():
            outputs = self.model(data_for_image)
            logits = outputs.logits
            

        result = self._postprocess(logits)
        return result 

ml_model = EmotionModel()            