from abc import ABC, abstractmethod
from enum import Enum
import cv2
import numpy as np

class HandMode(Enum):
    STAY = 0
    SELECT = 1
    OTHER = 2

class Hand:
    def __init__(self, img):
        self.img = img
        self.con = None
        self.mode = None
        self.end_points = []

class HandDetector(ABC):
    def __init__(self):
        self.colour_filter_up = np.array([255, 173, 127]) #YCrCb Filter Up Limit
        self.colour_filter_down = np.array([0, 133, 77]) #YCrCb Filter low Limit
        self.__model = None

    def model(self):
        if self.__model is None:
            # TODO
            raise TypeError('TEST!!!!!!!!!!')
        return self.__model

    @abstractmethod
    def is_hand(self, contour):
        pass

    def get_end_points(self, contour):
        phd = cv2.convexHull(contour, returnPoints = False)


    def get_contours(self, img):
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #YCrCb Transfer
        filterted_img = cv2.inRange(YCrCb, self.colour_filter_down, self.colour_filter_up) #Color filter
        return cv2.findContours(filterted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def __call__(self, img):
        hand = Hand(img)
        cons, hirs = self.get_contours(img)
        try:
            i = iter(cons)
            while True:
                con = next(i)
                if self.is_hand(con):
                    hand.con = con
                    hand.end_points = self.get_end_points(con)
                    break
        except StopIteration:
            return None
        return hand

class TestDec(HandDetector):
    def is_hand(self, contour):
        pass
