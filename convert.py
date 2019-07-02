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

for key in dict:
    colors.append('#%02x%02x%02x' % key)

window = Tk()


for i in range(len(colors)):
    lbl = Label(window, text="Color " + i, bg=colors[i])
    lbl.grid(column=0, row=0)

window.mainloop()

text = input("Enter numbers of colors separated by commas")
nums = text.split(",")


