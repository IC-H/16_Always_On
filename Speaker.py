from abc import ABC, abstractmethod
from gtts import gTTS
from playsound import playsound

class Speaker(ABC):
    @abstractmethod
    def text_to_speech(self, text):
        pass

class GttsSpeaker(Speaker):
    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save('tmp.mp3')
        playsound('tmp.mp3')
