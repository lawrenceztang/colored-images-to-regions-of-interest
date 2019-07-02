#!/usr/bin/env python
from tkinter import Tk
from tqdm import trange
import argparse
from tkinter import *
from PIL import Image
import PIL


def main(args):


    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    image = Image.open(args.inFile)

    dict = {}



    width, height = image.size

    for x in trange(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            if (r, g, b) in dict:
                dict[(r, g, b)] += 1
            else:
                dict[(r, g, b)] = 1

    colors = []
    colorToNum = {}

    i = 0
    for key in dict:
        colors.append('#%02x%02x%02x' % key)
        colorToNum[key] = i
        i += 1

    window = Tk()

    for i in range(len(colors)):
        lbl = Label(window, text="Color " + str(i), bg=colors[i])
        lbl.pack()

    window.update()

    text = input("Enter colors of interest separated by commas: ")
    nums = text.split(",")
    nums = list(map(int, nums))

    interest_count = 0

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            if colorToNum[(r, g, b)] not in nums:
                image.putpixel((x, y), (0, 0, 0))
            else:
                interest_count += 1

    import os
    file = args.outFile
    image.save(file + "/" + os.path.basename(os.path.normpath(args.inFile[:]))[:-4] + "-interest-" + str(interest_count / (width * height)) + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inFile")
    parser.add_argument("outFile")
    args = parser.parse_args()
    main(args)