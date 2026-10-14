from PIL import Image
from io import BytesIO 
import numpy as np
import time

class EmotionModel:

    _EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]

    def __init__(self):
        self.is_loaded = False 
        self.model = None
        self.input_size = (224, 224)
       
    @staticmethod
    def softmax(x):
        x_shifted = x - np.max(x)  
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)   

    def load_model(self):
        time.sleep(5)
        self.is_loaded = True

    def _validate_image(self, image_bytes: bytes) -> Image.Image:
        
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify()
            img = Image.open(BytesIO(image_bytes))
            img.load()
            return img
        except Exception as e:
            raise ValueError(f"Ошибка {e}")    


    def _preprocess(self, image : Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((self.input_size))
        data_for_image =  np.array(image)
        data_chw = np.transpose(data_for_image, (2, 0, 1))
        data_for_image = data_chw.astype(np.float32) / 255.0
        data_batch = data_for_image[np.newaxis, ...]
        return data_batch 


    def _postprocess(self, raw_output : np.ndarray)-> tuple[str, float, dict[str, float]]:
        raw_output = raw_output.squeeze()
        probabilities = EmotionModel.softmax(raw_output)
        predicted_idx = int(np.argmax(probabilities))
        emotion_predicted = self._EMOTIONS[predicted_idx]
        confidence = probabilities[predicted_idx]
        all_probabilities = {emotion : float(prob) for emotion, prob in zip(self._EMOTIONS, probabilities)} 
        return (emotion_predicted, confidence, all_probabilities)


    def predict(self, image_bytes: bytes) -> tuple[str, float, dict[str, float]]:

        if not self.is_loaded:
            raise RuntimeError("Модель не загружена!")

        img = self._validate_image(image_bytes = image_bytes)
        data_for_image = self._preprocess(img)
        logits = np.random.randn(1, 7) * 0.5  # случайные числа вокруг 0

        # Делаем один класс доминирующим
        dominant_idx = np.random.randint(0, 7)
        logits[0, dominant_idx] = 3.0

        result = self._postprocess(logits)
        return result 

ml_model = EmotionModel()            