#!/usr/bin/env python

#input: annotation in the form of an image, with different colors representing different classes
#user selects colors of interest that should be kept in the image
#output: new image, colors of interest are kept, colors that aren't of interest turn into RGB of (0, 0, 0), black

from tqdm import trange
import argparse
from tkinter import *
from PIL import Image
import PIL


def get_color_list(image):
    PIL.Image.MAX_IMAGE_PIXELS = 933120000

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
    numToColor = {}

    i = 0
    for key in dict:
        colors.append('#%02x%02x%02x' % key)
        numToColor[i] = key
        i += 1

    window = Tk()

    for i in range(len(colors)):
        lbl = Label(window, text="Color " + str(i), bg=colors[i])
        lbl.pack()

    window.update()
    print(numToColor)
    text = input("Enter colors of interest separated by commas: ")
    nums = text.split(",")
    nums = list(map(int, nums))
    colorsInterest = [numToColor[num] for num in nums]
    return colorsInterest

def mask_image(image, colorsInterest):
    width, height = image.size
    interest_count = 0

    for x in trange(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            if (r, g, b) not in colorsInterest:
                image.putpixel((x, y), (0, 0, 0))
            else:
                interest_count += 1

    return image, interest_count / (width * height)


def main(args):

    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    image = Image.open(args.inFile)
    colorsInterest = get_color_list(image)
    image, percent = mask_image(image, colorsInterest)

    import os

    file = args.outDirectory
    image.save(file + "/" + os.path.basename(os.path.normpath(args.inFile[:]))[:-4] + "-interest-" + str(
       percent) + ".png")

def auto(file, out, colorsInterest):
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    image = Image.open(file)
    image, percent = mask_image(image, colorsInterest)

    import os

    image.save(out + "/" + os.path.basename(os.path.normpath(file))[:-4] + "-interest-" + str(
        percent) + ".png")

"""
{0: (0, 0, 0), 1: (255, 0, 10), 2: (109, 207, 246), 3: (171, 160, 0), 4: (119, 255, 189), 5: (240, 110, 170), 6: (121, 0, 0), 7: (255, 255, 0), 8: (102, 45, 145)}
       black          red               blue                yellow             green               pink              dark red      bright yellow      purple

python -c 'from convert import auto; auto("HAW_2016_48_Annotated.png", "interest", [(171, 160, 0), (119, 255, 189), (109, 207, 246)])' && 
python -c 'from convert import auto; auto("HAW_2016_48_Annotated.png", "interest", [(171, 160, 0), (255, 0, 10), (240, 110, 170), (255, 255, 0), (121, 0, 0), (102, 45, 145)])' && 
python -c 'from convert import auto; auto("HAW_2016_48_Annotated.png", "interest", [(119, 255, 189), (255, 0, 10), (240, 110, 170), (255, 255, 0), (121, 0, 0), (102, 45, 145)])' && 
python -c 'from convert import auto; auto("HAW_2016_48_Annotated.png", "interest", [(255, 0, 10), (109, 207, 246), (121, 0, 0), (255, 255, 0), (102, 45, 145)])' && 
python -c 'from convert import auto; auto("HAW_2016_48_Annotated.png", "interest", [(171, 160, 0), (255, 0, 10), (240, 110, 170), (255, 255, 0), (121, 0, 0), (109, 207, 246)])' && 
python create-dataset.py HAW_2016_48_RAW.png interest dataset /home/lawrence/PycharmProjects/rost-cli 4736 6496 18368 20224 256 256 10000 &&
python network.py

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inFile")
    parser.add_argument("outDirectory")
    args = parser.parse_args()
    main(args)

