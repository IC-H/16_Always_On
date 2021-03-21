from abc import ABC, abstractmethod
import cv2

class Camera(ABC):
    @abstractmethod
    def prepare(self):
        pass

    @property
    @abstractmethod
    def is_camera_on(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

class Cv2Cam(Camera):
    def prepare(self):
        self.cap = cv2.VideoCapture(0)

    @property
    def is_camera_on(self):
        return isinstance(self.cap, cv2.VideoCapture) and self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception('Camera can not read')
        return frame

    def turn_off(self):
        self.cap.release()
        cv2.destroyAllWindows()
