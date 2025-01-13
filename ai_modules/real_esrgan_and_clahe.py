from RealESRGAN import RealESRGAN
import numpy as np
from PIL import Image
import torch
import cv2
import logging 

logging.basicConfig(filename = "logs.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VisModel():
    def __init__(self, img):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def pred_gan(self, img):
        model = RealESRGAN(device = self.device, scale = 2)
        model.load_weights('Real-ESRGAN/weights/RealESRGAN_x2.pth')
        image = Image.fromarray(img).convert('RGB')
        sr_image = model.predict(image)
        try:
            sr_image.save("upscaled/upscaled_real_esrgan.png")
            logging.info("Real-ESRGAN ciktisi upscaled dosyasina kaydedildi...")
        except Exception as e:
            logging.error(f"Real-ESRGAN ciktisi kaydedilirken hata olustu...")
        return sr_image

    def pred_gan_with_clahe(self, img):
        model = RealESRGAN(device = self.device, scale = 2)
        model.load_weights('Real-ESRGAN/weights/RealESRGAN_x2.pth')
        image = Image.fromarray(img).convert('RGB')
        sr_image = model.predict(image)        
        img_np = np.array(sr_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Her bir renk kanalı için CLAHE uygula
        channels = cv2.split(img_np)
        clahe_channels = [clahe.apply(channel) for channel in channels]
        
        # Sonuçları birleştir
        clahe_image = cv2.merge(clahe_channels)

        # Yeni görüntüyü kaydet
        try:
            output_path = 'upscaled/gan_with_clahe.png'
            Image.fromarray(clahe_image).save(output_path)
            logging.info("Real-ESRGAN + CLAHE ciktisi upscaled dosyasina kaydedildi...")
        except Exception as e:
            logging.error(f"Real-ESRGAN + CLAHE ciktisi kaydedilirken hata olustu...")
        return clahe_image

        