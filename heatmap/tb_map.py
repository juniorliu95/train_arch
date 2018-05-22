from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
from PIL import Image
from PIL import ImageDraw2
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import re
import cv2
import time
import cf

if sys.version > '3':
    PY3 = True
else:
    PY3 = False

_IMG_WIDTH = 1024
_IMG_HEIGHT = 1024

TOP_DIR = '../'
IMG_DIR = 'Pleural_Thickening/'
PRED_DIR = IMG_DIR[:-1] + "_result/"
MASK_DIR = IMG_DIR[:-1] + "_seg_mask/"


class HeatMap(object):
    def __init__(self,
                 data=None,
                 base=None,
                 mask=None,
                 width=0,
                 height=0
                 ):

        assert base is None or os.path.isfile(base)
        assert mask is None or os.path.isfile(mask)
        assert cf.is_num(width) and cf.is_num(height)
        assert width >= 0 and height >= 0

        self.data = data
        self.base = base
        self.mask = mask
        self.width = width
        self.height = height
        self.save_as = None

    def __heat(self, heat_data, x, y, n, template):
        l = len(heat_data)
        width = self.width
        p = width * y + x

        for ip, iv in template:
            p2 = p + ip
            if 0 <= p2 < l:
                heat_data[p2] += iv * n

    def __paint_heat(self, heat_data, colors):
        im = self.__im  # heatmap
        rr = re.compile(", (\d+)%\)")
        dr = ImageDraw2.ImageDraw.Draw(im)
        width = self.width
        height = self.height

        max_v = max(heat_data)
        if max_v <= 0:
            return

        r = 240.0 / max_v
        heat_data2 = [int(i * r) - 1 for i in heat_data]  # convert to 240 steps

        size = width * height
        for p in xrange(size):
            v = heat_data2[p]
            if v > 0:
                x, y = p % width, p // width
                color = colors[v]
                alpha = int(rr.findall(color)[0])
                alpha_put = int(200 - 2 * alpha)
                self.band_alpha.putpixel((x, y), alpha_put)
                if alpha > 50:
                    al = 255 - 255 * (alpha - 50) // 50
                    im.putpixel((x, y), (0, 0, 255, al))
                else:
                    dr.point((x, y), fill=color)

    def __add_base(self, base=None):
        if not base:
            base = self.base
        self.__im0 = None

        if base:
            str_type = (str,) if PY3 else (str, unicode)
            self.__im0 = Image.open(base) if type(base) in str_type else base
            self.__im0 = self.__im0.convert("RGBA")

        if not self.__im0:
            return

        # self.__im0.paste(self.__im, mask=self.__im)
        img_resize = self.__im.resize((_IMG_WIDTH, _IMG_HEIGHT))

        self.__im0.paste(img_resize, mask=img_resize)
        self.__im = self.__im0

    def __avgvalue(self, p, pix, template):
        l = self.width * self.height
        counter = 0
        v = 0
        for ip, _ in template:
            p2 = p + ip
            if 0 <= p2 < l:
                x, y = p2 % self.width, p2 // self.width
                v += pix[x, y] / 255.0
                counter += 1
        avg = v / counter
        return avg

    def __avgvalue_rect(self, p, pix):
        a = 4
        x, y = p % self.width, p // self.width
        y_min = max(y - a, 0)
        y_max = min(y + a, self.height)
        x_min = max(x - a, 0)
        x_max = min(x + a, self.width)
        counter = 0
        v = 0
        for x_i in range(x_min, x_max):
            for y_i in range(y_min, y_max):
                counter += 1
                v += pix[x_i, y_i] / 255.0
        avg = v / counter
        return avg

    def Gaussian(self, v, sigma):
        return np.exp(- (v - 1) ** 2 / ((sigma ** 2) * 2))

    def __filter_im(self, im, template):
        if not self.mask:
            print("No mask!")
            return
        pix_im = im.load()
        pix_mask = Image.open(self.mask).load()
        for x in xrange(self.width):
            for y in xrange(self.height):
                avg = self.__avgvalue(x, y, pix_mask)
                kernel = [self.Gaussian(avg, 1), self.Gaussian(avg, 1), self.Gaussian(avg, 1), self.Gaussian(avg, 2)]
                r = pix_im[x, y][0]
                g = pix_im[x, y][1]
                b = pix_im[x, y][2]
                alpha = pix_im[x, y][3]
                if avg <= 0.001:
                    pix_im[x, y] = (0, 0, 255, int(alpha * avg))
                elif avg <= 0.999:
                    if r == 255:
                        pix_im[x, y] = (int(r * kernel[0]), int(255 * kernel[1]), 0, int(alpha * kernel[3]))
                    # elif g:
                    # pix_im[x,y] = (int(r*avg), int(g*avg), int(b*avg), int(alpha*avg*kernel[3]))
                    else:
                        pix_im[x, y] = (int(r * kernel[0]), int(g * kernel[1]), int(b * kernel[2]), int(alpha * avg))
        return

    def __filter_data(self, data, template):
        if not self.mask:
            print("No mask!")
            return
        pix_mask = Image.open(self.mask).load()
        size = len(data)
        for p in xrange(size):
            # avg = self.__avgvalue(p, pix_mask, template)
            avg = self.__avgvalue_rect(p, pix_mask)
            if avg < 1:
                data[p] = data[p] * self.Gaussian(avg, 0.6)
        return

    def heatmap(self, save_as=None, base=None, data=None, r=1):
        if not data.any():
            data = self.data

        self.height, self.width = data.shape
        self.__im = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self.band_alpha = self.__im.split()[-1]

        circle = cf.mk_circle(r, self.width)
        heat_data = [0] * self.width * self.height

        # self.__filter_data(data)

        for x in xrange(self.width):
            for y in xrange(self.height):
                n = data[y, x]
                self.__heat(heat_data, x, y, n, circle)  # put on pixels one by one

        self.__filter_data(heat_data, circle)

        self.__paint_heat(heat_data, cf.mk_colors())
        self.__im.putalpha(self.band_alpha)

        self.__add_base(base)

        if save_as:
            self.save_as = save_as
            self.__save()

        return self.__im

    def __save(self):
        # save_as = os.path.join(os.getcwd(), self.save_as)
        save_as = os.path.join(TOP_DIR + "heatmap/", self.save_as)
        folder, fn = os.path.split(save_as)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.__im.save(save_as)
        self.__im = None


def crop(image):
    for i in range(0, 7):
        image = image[~np.all(image == i, axis=1)]
        image = image[:, ~np.all(image == i, axis=0)]
    for i in range(249, 256):
        image = image[~np.all(image == i, axis=1)]
        image = image[:, ~np.all(image == i, axis=0)]
    return image


def read_process_image(addr):
    img = cv2.imread(addr, 0)
    img_crop = crop(img)
    img_equhist = cv2.equalizeHist(img_crop)
    img_resize = cv2.resize(img_equhist, (_IMG_WIDTH, _IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_addr = TOP_DIR + "tmp.png"
    cv2.imwrite(img_addr, img_resize)
    return img_addr


def to_hm_list(img_array):
    hm_list = []
    rows, columns = img_array.shape
    for row in xrange(rows):
        for column in xrange(columns):
            hm_list.append((row, column, img_array[row, column]))
    return hm_list


def main(data_cam, img_filename):
    print 'painting', img_filename
    duration = []

    hm = HeatMap()

    # if pred_label==1 and ture_label==1:
    img_array_hm = data_cam[0, :, :,1]
    # hm_list = to_hm_list(img_array_hm)
    save_name = 'heatmap/' + img_filename
    start_time = time.time()
    hm.heatmap(data=img_array_hm, save_as=save_name)
    duration.append(time.time() - start_time)

    print("done.")


if __name__ == "__main__":
    pass