from abc import ABC, abstractmethod
from pytesseract import image_to_string

class Ocr(ABC):
    @abstractmethod
    def img_to_text(self, imag):
        pass

class PytesseractOcr(Ocr):
    def img_to_text(self, img):
        return image_to_string(img)
