from tkinter import Tk
from tkinter.filedialog import askopenfilename

#/home/lawrence/Documents/Summer_2019
#image pixel count of 625 million

Tk().withdraw()
filename = askopenfilename()

from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

image = Image.open(filename)


dict = {}

from tkinter import *

width, height = image.size

for x in range(width):
    for y in range(height):
        r, g, b = image.getpixel((x, y))

        if (r, g, b) in dict:
            dict[(r, g, b)] += 1
        else:
            dict[(r, g, b)] = 1

        if (x * height + y) % 1000000 == 0:
            print(x * height + y)

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


for x in range(width):
    for y in range(height):
        r, g, b = image.getpixel((x, y))

        if colorToNum[(r, g, b)] not in nums:
           image.putpixel((x, y), (0, 0, 0))


file = filename[:-4] + "-interest.jpg"
image.save(file)
