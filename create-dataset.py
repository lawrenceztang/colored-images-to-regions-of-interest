#!/usr/bin/env python

#input: image and annotation for that image
#output: creates dataset with random chunks of the original image and annotation, also runs ROST word extraction on these images
#the area of the original image to sample from, the dimensions of the output and the size of the new dataset can be specified

import argparse
from tkinter import *
import PIL
from PIL import Image
from random import randint
import os
from tqdm import trange

#python create-dataset.py HAW_2016_48_RAW.png interest dataset /home/lawrence/PycharmProjects/rost-cli 4736 6496 18368 20224 256 256 90000

def saveOutputs(newImageInput, newImageTarget, q, i, mode):
    basePath = args.outPath + str(q) + "/" + mode + "/" + os.path.basename(os.path.normpath(args.imageInputPath))[:-4] + "-" + str(i)
    inputPath = basePath + "-input" + ".png"
    newImageInput.save(inputPath)
    newImageTarget.save(basePath + "-target" + ".png")

    # os.system(
    #     args.rostPath + "/bin/words.extract.video --image=" + inputPath + " --texton=true --color=true --orb=true --orb-vocabulary=" + args.rostPath + "/libvisualwords/data/orb_vocab/default.yml --texton-vocabulary " + args.rostPath + "/libvisualwords/data/texton.vocabulary.baraka.1000.csv --orb-out=" + basePath + "-orb.csv --texton-out=" + basePath + "-texton.csv --color-out=" + basePath + "-color.csv")
    # os.system(args.rostPath + "/bin/words.mix --timestep=1250 -o " + basePath + "-mix.csv -i 436 " + basePath + "-color.csv 1000 " + basePath + "-texton.csv 5000 " + basePath + "-orb.csv")
    # temp = open(basePath + "-mix.csv")
    # string = temp.read()
    # all_words.write(string)

def main(args):

    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    imageTargets = os.listdir(args.imageTargetPath)

    for q in range(len(imageTargets)):

        imageInput = Image.open(args.imageInputPath)
        imageTarget = Image.open(args.imageTargetPath + "/" + imageTargets[q])
        imageInput = imageInput.crop((int(args.leftBound), int(args.topBound), int(args.rightBound), int(args.bottomBound)))
        imageTarget = imageTarget.crop((int(args.leftBound), int(args.topBound), int(args.rightBound), int(args.bottomBound)))
        width, height = imageInput.size
        try:
            os.mkdir(args.outPath + str(q))
            os.mkdir(args.outPath + str(q) + "/train/")
        except:
            pass
        try:
            os.mkdir(args.outPath + str(q) + "/test/")
        except:
            pass
        try:
            os.mkdir(args.outPath + str(q) + "/val/")
        except:
            pass

        # all_words = open(args.outPath + "/words.all.csv", "w+")

        for i in trange(int(args.iterations)):
            randX = randint(0, int(width * .9) - int(args.newWidth))
            randY = randint(0, height - int(args.newHeight))

            newImageInput = imageInput.crop((randX, randY, randX + int(args.newWidth), randY + int(args.newHeight)))
            newImageTarget = imageTarget.crop((randX, randY, randX + int(args.newWidth), randY + int(args.newHeight)))
            saveOutputs(newImageInput, newImageTarget, q, i, "train")


        for i in trange(int(int(args.iterations) / 10)):
            randX = randint(int(width * .9), width - int(args.newWidth))
            randY = randint(0, height - int(args.newHeight))

            newImageInput = imageInput.crop((randX, randY, randX + int(args.newWidth), randY + int(args.newHeight)))
            newImageTarget = imageTarget.crop((randX, randY, randX + int(args.newWidth), randY + int(args.newHeight)))
            saveOutputs(newImageInput, newImageTarget, q, i, "val")

        for i in trange(int(int(args.iterations) / 10)):
            randX = randint(int(width * .8), int(width * .9 - int(args.newWidth)))
            randY = randint(0, height - int(args.newHeight))

            newImageInput = imageInput.crop((randX, randY, randX + int(args.newWidth), randY + int(args.newHeight)))
            newImageTarget = imageTarget.crop((randX, randY, randX + int(args.newWidth), randY + int(args.newHeight)))
            saveOutputs(newImageInput, newImageTarget, q, i, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imageInputPath")
    parser.add_argument("imageTargetPath")
    parser.add_argument("outPath")
    parser.add_argument("rostPath")
    parser.add_argument("leftBound")
    parser.add_argument("topBound")
    parser.add_argument("rightBound")
    parser.add_argument("bottomBound")
    parser.add_argument("newWidth")
    parser.add_argument("newHeight")
    parser.add_argument("iterations")
    args = parser.parse_args()
    main(args)


