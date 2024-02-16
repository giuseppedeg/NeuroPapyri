from PIL import Image, ImageStat
from random import random

class BinarizeTransform(object):
    def __init__(self, p = 0.2):
        assert isinstance(p, float)
        assert p <= 1.0
        self.p = p
    
    def __call__(self, image):
        p = random()
        if p > self.p:
            return image
        stat = ImageStat.Stat(image)
        thresholds = stat.mean
        extract = Image.new(mode='RGB', size=image.size)
        data = image.getdata()
        newdata = []
        for pixel in data:
            #print(pixel)
            if pixel[0] < thresholds[0] and pixel[1] < thresholds[1] and pixel[2] < thresholds[2]:
                newdata.append(pixel)
            else:
                newdata.append((255, 255, 255))
        extract.putdata(newdata)
        return extract
