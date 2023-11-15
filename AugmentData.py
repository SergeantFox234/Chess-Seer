import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import glob
import cv2
import random as rand

#Functions for augmenting
#Cropping Function
def crop(image):
    #To crop: do
    #newImage = image[begX:endX][begY:endY]
    #then rescale it, so make endX - begX = endY - begY be divisible into image.shape[0]
    #224x224, so either divide by 2, 4, or 8 (bigger numbers make the pixel count too small, and the stretch will likely not give good data)
    choice = rand.randint(1, 2)
    distance = 0
    if choice == 1:
        #divide by 2
        distance = 112
    elif choice == 2:
        #divide by 4
        distance = 56
    
    #randomly choose the stating indices:
    startLim = 224 - distance
    xStart = rand.randint(0, startLim - 1)
    yStart = rand.randint(0, startLim - 1)

    newImage = image[xStart: xStart + distance, yStart: yStart + distance]

    #resize the image
    newImage = cv2.resize(newImage, (224, 224))
    return newImage
#Scale the color channels of an image
def colorScale(image):
    scale = rand.uniform(0.6, 0.8)
    newImage = image.copy()
    xIndex = 0
    for xval in image:
        yIndex = 0
        for yval in xval:
            newImage[xIndex][yIndex] = yval * scale
            yIndex = yIndex + 1
        xIndex = xIndex + 1
    
    return newImage

#Some items were sampled from https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset?resource=download
#Some pictures were taken personally by me

wking_dir = 'src\\whiteKings\\*.jpg'
bking_dir = 'src\\blackKings\\*.jpg'
wqueen_dir = 'src\\whiteQueens\\*.jpg'
bqueen_dir = 'src\\blackQueens\\*.jpg'
wbishop_dir = 'src\\whiteBishops\\*.jpg'
bbishop_dir = 'src\\blackBishops\\*.jpg'
wknight_dir = 'src\\whiteKnights\\*.jpg'
bknight_dir = 'src\\blackKnights\\*.jpg'
wrook_dir = 'src\\whiteRooks\\*.jpg'
brook_dir = 'src\\blackRooks\\*.jpg'
wpawn_dir = 'src\\whitePawns\\*.jpg'
bpawn_dir = 'src\\blackPawns\\*.jpg'

wkingpics = glob.glob(wking_dir)
bkingpics = glob.glob(bking_dir)
wqueenpics = glob.glob(wqueen_dir)
bqueenpics = glob.glob(bqueen_dir)
wbishpics = glob.glob(wbishop_dir)
bbishpics = glob.glob(bbishop_dir)
wknightpics = glob.glob(wknight_dir)
bknightpics = glob.glob(bknight_dir)
wrookpics = glob.glob(wrook_dir)
brookpics = glob.glob(brook_dir)
wpawnpics = glob.glob(wpawn_dir)
bpawnpics = glob.glob(bpawn_dir)

wkings = []
bkings = []
wqueens = []
bqueens = []
wbishops = []
bbishops = []
wknights = []
bknights = []
wrooks = []
brooks = []
wpawns = []
bpawns = []

for image in wkingpics:
    img = cv2.imread(image)
    wkings.append(img)
for image in bkingpics:
    img = cv2.imread(image)
    bkings.append(img)
for image in wqueenpics:
    img = cv2.imread(image)
    wqueens.append(img)
for image in bqueenpics:
    img = cv2.imread(image)
    bqueens.append(img)
for image in wbishpics:
    img = cv2.imread(image)
    wbishops.append(img)
for image in bbishpics:
    img = cv2.imread(image)
    bbishops.append(img)
for image in wknightpics:
    img = cv2.imread(image)
    wknights.append(img)
for image in bknightpics:
    img = cv2.imread(image)
    bknights.append(img)
for image in wrookpics:
    img = cv2.imread(image)
    wrooks.append(img)
for image in brookpics:
    img = cv2.imread(image)
    brooks.append(img)
for image in wpawnpics:
    img = cv2.imread(image)
    wpawns.append(img)
for image in bpawnpics:
    img = cv2.imread(image)
    bpawns.append(img)

print("\nKings: white")
print(wkings[0].shape)
print(len(wkings))

print("\nKings: black")
print(bkings[0].shape)
print(len(bkings))

print("\nQueens: white")
print(wqueens[0].shape)
print(len(wqueens))

print("\nQueens: black")
print(bqueens[0].shape)
print(len(bqueens))

print("\nBishops: white")
print(wbishops[0].shape)
print(len(wbishops))

print("\nBishops: black")
print(bbishops[0].shape)
print(len(bbishops))

print("\nKnights: white")
print(wknights[0].shape)
print(len(wknights))

print("\nKnights: black")
print(bknights[0].shape)
print(len(bknights))

print("\nRooks: white")
print(wrooks[0].shape)
print(len(wrooks))

print("\nRooks: black")
print(brooks[0].shape)
print(len(brooks))

print("\nPawns: white")
print(wpawns[0].shape)
print(len(wpawns))

print("\nPawns: black")
print(bpawns[0].shape)
print(len(bpawns))

#Dataset is loaded in... lets augment it!!
augWKings = []
augBKings = []
augWQueens = []
augBQueens = []
augWBishops = []
augBBishops = []
augWKnights = []
augBKnights = []
augWRooks = []
augBRooks = []
augWPawns = []
augBPawns = []

print('\nAugmenting Images!')
imgIndex = 0
numImages = 8 * (len(wkings) + len(bkings) + len(wqueens) + len(bqueens) + len(wbishops) + len(bbishops) + len(wknights) + len(bknights) + len(wrooks) + len(brooks) + len(wpawns) + len(bpawns))
for image in wkings:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augWKings.append(image)
        elif choice == 1:
            augWKings.append(cv2.flip(image, 1))
        elif choice == 2:
            augWKings.append(image) #Used to be crop(image)
        elif choice == 3:
            augWKings.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augWKings.append(tempImg) #Used to be crop(tempImg)
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augWKings.append(colorScale(tempImg))
        elif choice == 6:
            #tempImg = crop(image)
            tempImg = image
            augWKings.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            #tempImg = crop(tempImg)
            augWKings.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in bkings:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augBKings.append(image)
        elif choice == 1:
            augBKings.append(cv2.flip(image, 1))
        elif choice == 2:
            augBKings.append(crop(image))
        elif choice == 3:
            augBKings.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augBKings.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augBKings.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augBKings.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augBKings.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in wqueens:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augWQueens.append(image)
        elif choice == 1:
            augWQueens.append(cv2.flip(image, 1))
        elif choice == 2:
            augWQueens.append(crop(image))
        elif choice == 3:
            augWQueens.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augWQueens.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augWQueens.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augWQueens.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augWQueens.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in bqueens:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augBQueens.append(image)
        elif choice == 1:
            augBQueens.append(cv2.flip(image, 1))
        elif choice == 2:
            augBQueens.append(crop(image))
        elif choice == 3:
            augBQueens.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augBQueens.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augBQueens.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augBQueens.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augBQueens.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in wbishops:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augWBishops.append(image)
        elif choice == 1:
            augWBishops.append(cv2.flip(image, 1))
        elif choice == 2:
            augWBishops.append(crop(image))
        elif choice == 3:
            augWBishops.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augWBishops.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augWBishops.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augWBishops.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augWBishops.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in bbishops:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augBBishops.append(image)
        elif choice == 1:
            augBBishops.append(cv2.flip(image, 1))
        elif choice == 2:
            augBBishops.append(crop(image))
        elif choice == 3:
            augBBishops.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augBBishops.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augBBishops.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augBBishops.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augBBishops.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in wknights:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augWKnights.append(image)
        elif choice == 1:
            augWKnights.append(cv2.flip(image, 1))
        elif choice == 2:
            augWKnights.append(crop(image))
        elif choice == 3:
            augWKnights.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augWKnights.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augWKnights.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augWKnights.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augWKnights.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in bknights:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augBKnights.append(image)
        elif choice == 1:
            augBKnights.append(cv2.flip(image, 1))
        elif choice == 2:
            augBKnights.append(crop(image))
        elif choice == 3:
            augBKnights.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augBKnights.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augBKnights.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augBKnights.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augBKnights.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in wrooks:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augWRooks.append(image)
        elif choice == 1:
            augWRooks.append(cv2.flip(image, 1))
        elif choice == 2:
            augWRooks.append(crop(image))
        elif choice == 3:
            augWRooks.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augWRooks.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augWRooks.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augWRooks.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augWRooks.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in brooks:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augBRooks.append(image)
        elif choice == 1:
            augBRooks.append(cv2.flip(image, 1))
        elif choice == 2:
            augBRooks.append(crop(image))
        elif choice == 3:
            augBRooks.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augBRooks.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augBRooks.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augBRooks.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augBRooks.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in wpawns:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augWPawns.append(image)
        elif choice == 1:
            augWPawns.append(cv2.flip(image, 1))
        elif choice == 2:
            augWPawns.append(crop(image))
        elif choice == 3:
            augWPawns.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augWPawns.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augWPawns.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augWPawns.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augWPawns.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')
for image in bpawns:
    imgIndex += 1
    for random in range(8):
        choice = rand.randint(0,7)
        if choice == 0:
            augBPawns.append(image)
        elif choice == 1:
            augBPawns.append(cv2.flip(image, 1))
        elif choice == 2:
            augBPawns.append(crop(image))
        elif choice == 3:
            augBPawns.append(colorScale(image))
        elif choice == 4:
            tempImg = cv2.flip(image, 1)
            augBPawns.append(crop(tempImg))
        elif choice == 5:
            tempImg = cv2.flip(image, 1)
            augBPawns.append(colorScale(tempImg))
        elif choice == 6:
            tempImg = crop(image)
            augBPawns.append(colorScale(tempImg))
        elif choice == 7:
            tempImg = cv2.flip(image, 1)
            tempImg = crop(tempImg)
            augBPawns.append(colorScale(tempImg))
print(f'{8*imgIndex} / {numImages}')

awkings = np.array(augWKings)
abkings = np.array(augBKings)
awqueens = np.array(augWQueens)
abqueens = np.array(augBQueens)
awbishops = np.array(augWBishops)
abbishops = np.array(augBBishops)
awknights = np.array(augWKnights)
abknights = np.array(augBKnights)
awrooks = np.array(augWRooks)
abrooks = np.array(augBRooks)
awpawns = np.array(augWPawns)
abpawns = np.array(augBPawns)

print("\nAugmented white kings: ", awkings.shape)
print("Augmented black kings: ", abkings.shape)
print("\nAugmented white Queens: ", awqueens.shape)
print("Augmented black Queens: ", abqueens.shape)
print("\nAugmented white Bishops: ", awbishops.shape)
print("Augmented black Bishops: ", abbishops.shape)
print("\nAugmented white Knights: ", awknights.shape)
print("Augmented black Knights: ", abknights.shape)
print("\nAugmented white Rooks: ", awrooks.shape)
print("Augmented black Rooks: ", abrooks.shape)
print("\nAugmented white Pawns: ", awpawns.shape)
print("Augmented black Pawns: ", abpawns.shape)

with open('src/augmentedLabels', 'w') as labelsFile:
    index = 0
    for image in awkings:
        pathAug = "src/augmented/white_king_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "white_king_" + str(index) + ".jpg, 0\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in abkings:
        pathAug = "src/augmented/black_king_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "black_king_" + str(index) + ".jpg, 1\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in awqueens:
        pathAug = "src/augmented/white_Queen_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "white_Queen_" + str(index) + ".jpg, 2\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in abqueens:
        pathAug = "src/augmented/black_Queen_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "black_Queen_" + str(index) + ".jpg, 3\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in awbishops:
        pathAug = "src/augmented/white_Bishop_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "white_Bishop_" + str(index) + ".jpg, 4\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in abbishops:
        pathAug = "src/augmented/black_Bishop_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "black_Bishop_" + str(index) + ".jpg, 5\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in awknights:
        pathAug = "src/augmented/white_Knights_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "white_Knights_" + str(index) + ".jpg, 6\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in abknights:
        pathAug = "src/augmented/black_Knights_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "black_Knights_" + str(index) + ".jpg, 7\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in awrooks:
        pathAug = "src/augmented/white_Rooks_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "white_Rooks_" + str(index) + ".jpg, 8\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in abrooks:
        pathAug = "src/augmented/black_Rooks_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "black_Rooks_" + str(index) + ".jpg, 9\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in awpawns:
        pathAug = "src/augmented/white_pawns_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "white_pawns_" + str(index) + ".jpg, 10\n"
        index = index + 1
        labelsFile.write(labelAug)
    index = 0
    for image in abpawns:
        pathAug = "src/augmented/black_pawns_" + str(index) + ".jpg"
        cv2.imwrite(pathAug, image)
        labelAug = "black_pawns_" + str(index) + ".jpg, 11\n"
        index = index + 1
        labelsFile.write(labelAug)

#Then, we are going to want to train a logistic regression
#There are 12 classes:: White and black versions of 
# -King -Queen -Bishop -Knight -Rook -Pawn