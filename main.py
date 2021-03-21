from optparse import OptionParser
from Reader import Reader
from Camera import Cv2Cam
from HandDetector import TestDec
from Ocr import PytesseractOcr
from Speaker import GttsSpeaker

parser = OptionParser()
parser.add_option("-d", "--debug", dest="debug", help="debug flag", default=False, action="store_true")
(options, args) = parser.parse_args()

reader = Reader()
reader.debug = options.debug
reader.camera = Cv2Cam()
reader.hand_detector = TestDec()
reader.ocr = PytesseractOcr()
reader.speaker = GttsSpeaker()
reader()
