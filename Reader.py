from enum import Enum
from time import sleep
from math import sqrt
from Camera import Camera
from HandDetector import Hand, HandDetector, HandMode
from Ocr import Ocr
from Speaker import Speaker
import logging
import cv2

logging.basicConfig(filename='logs/warnig.log',level=logging.WARNING, format='【%(levelname)s】【%(asctime)s】 %(message)s')
logging.basicConfig(filename='logs/debug.log',level=logging.DEBUG, format='【%(levelname)s】【%(asctime)s】 %(message)s')

class Key(Enum):
    NONE = -1
    ESC = 27

class ExitException(Exception):
    pass

class ReadStep(Enum):
    FIND_READ_IMG = 1
    READ = 2

class Square:
    def __init__(self):
        self.__hand1 = None
        self.__hand2 = None
        self.img = None

    @property
    def hand1(self):
        return self.__hand1

    @hand1.setter
    def hand1(self, hand):
        if not isinstance(hand, Hand):
            raise TypeError('hand has to be instance of Hand class')
        if not hand.mode is HandMode.SELECT:
            raise TypeError('hand mode has to be SELECT')
        self.__hand1 = hand

    @property
    def hand2(self):
        return self.__hand2

    @hand2.setter
    def hand2(self, hand):
        if not isinstance(hand, Hand):
            raise TypeError('hand has to be instance of Hand class')
        if not hand.mode is HandMode.SELECT:
            raise TypeError('hand mode has to be SELECT')
        self.__hand2 = hand

    def trim(self):
        if self.hand1 is None:
            return
        if self.hand2 is None:
            return
        img = self.hand2.img
        p1 = self.hand1.end_points[0]
        p2 = self.hand2.end_points[0]
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        if x1 > x2:
            x1 = x2
            x2 = p1[0]
        if y1 > y2:
            y1 = y2
            y2 = p1[1]
        self.img = img[x1:x2, y1:y2]

    def __call__(self, hand):
        if hand is None:
            return
        if not hand.mode is HandMode.SELECT:
            return
        if self.hand1 is None:
            self.hand1 = hand
        elif self.hand2 is None:
            def is_far_from_hand1(hand):
                p1 = self.hand1.end_points[0]
                p2 = hand.end_points[0]
                x, y = hand.img.shape
                dis = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                ref = sqrt(x**2 + y**2)
                return (dis / ref) > 0.3
            if is_far_from_hand1(hand):
                self.hand2 = hand
        self.trim()

    def __bool__(self):
        if self.hand1 is None:
            return False
        if self.hand2 is None:
            return False
        if self.img is None:
            return False
        return True

class Reader:
    def __init__(self):
        self.__camera = None
        self.__hand_detector = None
        self.__ocr = None
        self.__speaker = None
        self.step = ReadStep.FIND_READ_IMG
        self.square = Square()
        self.debug = False

    @property
    def camera(self):
        return self.__camera

    @camera.setter
    def camera(self, camera):
        if not isinstance(camera, Camera):
            raise TypeError('camera has to be instace of Camera class')
        camera.prepare()
        self.__camera = camera

    @property
    def hand_detector(self):
        return self.__hand_detector

    @hand_detector.setter
    def hand_detector(self, hand_detector):
        if not isinstance(hand_detector, HandDetector):
            raise TypeError('hand_detector has to be instance of HandDetector class')
        self.__hand_detector = hand_detector

    @property
    def ocr(self):
        return self.__ocr

    @ocr.setter
    def ocr(self, ocr):
        if not isinstance(ocr, Ocr):
            raise TypeError('ocr has to be instance of Ocr class')
        self.__ocr = ocr

    @property
    def speaker(self):
        return self.__speaker

    @speaker.setter
    def speaker(self, speaker):
        if not isinstance(speaker, Speaker):
            raise TypeError('speaker has to be instance of Speaker')
        self.__speaker = speaker

    @property
    def is_prepared(self):
        if self.camera is None:
            return False
        if self.hand_detector is None:
            return False
        if self.ocr is None:
            return False
        if self.speaker is None:
            return False
        return True

    def check_key_interupt(self):
        try:
            pressed_key = cv2.waitKey(10)
            key = Key._value2member_map_[pressed_key]
        except KeyError as e:
            # Ignore other key interupt
            if self.debug:
                print(f'{e}, pressed_key {pressed_key}')
                logging.debug(f'{e}')
            return
        if key is Key.ESC:
            raise ExitException('End Process')

    def step_find_read_img(self):
        if not self.camera.is_camera_on:
            raise Exception('Camera not working')
        frame = self.camera.read()
        hand = self.hand_detector(frame)
        self.square(hand)
        if self.square:
            self.step = ReadStep.READ
        cv2.imshow('Preview', frame)

    def step_read(self):
        if not self.square:
            raise Exception('Image does not exist')
        text = self.ocr.img_to_text(self.square.img)
        self.speaker.text_to_speech(text)
        self.step = ReadStep.FIND_READ_IMG

    def __call__(self):
        if self.debug:
            print('Run as debug')
        if not self.is_prepared:
            msg = 'Not prepared'
            print(msg)
            if self.debug:
                logging.debug(msg)
            return
        try:
            while True:
                self.check_key_interupt()
                if self.step is ReadStep.FIND_READ_IMG:
                    self.step_find_read_img()
                elif self.step is ReadStep.READ:
                    self.step_read()
        except ExitException as e:
            msg = f'{e}'
            print(msg)
            if self.debug:
                logging.debug(msg)
        except Exception as e:
            logging.warning(f'{e}')
        finally:
            self.camera.turn_off()
