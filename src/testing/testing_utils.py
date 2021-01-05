from PIL import Image, ImageFile, ImageSequence
import subprocess
import time
import cv2
#import skvideo
# skvideo.setFFmpegPath("C:\\Python-64bit\\GoPiGo-interpeter-64bit\\Lib\\site-packages\\ffmpeg\\bin")
# import skvideo.io
import numpy as np


def loadVideo(filePath, collector=None) -> list:
    if collector is None:
        frames = []
    else:
        frames = collector

    max_attempts = 0
    video = cv2.VideoCapture(filePath)

    if not video.isOpened():
        print(f"Failed to open Video from PATH -> {filePath}")
        return frames

    while max_attempts <= 9:
        success, frame = video.read()

        if success:
            frames.append(frame)
        else:
            max_attempts += 1
    video.release()

    return frames


class ImageBuffer(list):
    def __init__(self, video=None):
        super().__init__()

        if video:
            self.loadVideo(video)

    def addImage(self, img):
        self.append(img)

    def map(self, func, *args):
        if args:
            for i, img in enumerate(self):
                self[i] = func(img, *args)
        else:
            for i, img in enumerate(self):
                self[i] = func(img)

    def loadVideo(self, filePath):
        loadVideo(filePath, self)

    def createVideo(self, name, openExplorer=False):
        skvideo.io.vwrite(name, self)

        if openExplorer:
            subprocess.Popen(r'explorer /select,' + name + r'"')

    def saveSequence(self, folder):
        for i, img in enumerate(self):
            Image.fromarray(img).save(f"{folder}\\{i}.png")

    @classmethod
    def saveImage(cls, img: np.array, filePath):
        Image.fromarray(img).save(filePath)
        print('img save')




