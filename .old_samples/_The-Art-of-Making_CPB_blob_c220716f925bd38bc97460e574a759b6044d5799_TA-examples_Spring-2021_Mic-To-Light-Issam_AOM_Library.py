# AOM Library

import math
import time
from rainbowio import colorwheel

# Basic Math

'''This Function Needs To Be Defined'''
def mean(values):
    return sum(values) / len(values)

'''This Function Needs To Be Defined'''
def normalized_rms(values):
    minbuf = int(mean(values))
    sum_of_samples = sum(
        float(sample - minbuf) * (sample - minbuf)
        for sample in values
    )

    return math.sqrt(sum_of_samples / len(values))

# Translation
'''This Function Needs To Be Defined'''
def mapToRange(value, leftMin, leftMax, rightMin, rightMax):
# This Function From https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

'''This Function Needs To Be Defined'''
def getRGBFromSound(volume):
    r = mapToRange(volume*3,0,30000,0,255)
    g = mapToRange(volume*2,0,20000,0,255)
    b = mapToRange(volume ,0,120000,0,255)
    return (r,g,b)

# Procedures

'''This Function Needs To Be Defined'''
def rainbow(pixels,wait):
    for j in range(255):
        for i in range(len(pixels)):
            idx = int(i + j)
            pixels[i] = colorwheel(idx & 255)
        time.sleep(wait)

'''This Function Needs To Be Defined'''
def colorWipe(pixels,color,wait):
    for i in range(len(pixels)):
        pixels[i]=color
        time.sleep(wait)

'''This Function Needs To Be Defined'''
def colorWipeX(pixels,color,x,wait):
    if (x > len(pixels)):
        x = len(pixels)
    elif ((x < 0)):
        x = 0
    for i in range(x):
        pixels[i]=color
        time.sleep(wait)

'''This Function Needs To Be Defined'''
def theaterChase(pixels,color,wait):
    for q in range(0,2,1):
        for i in range(0,len(pixels),2):
            pixels[i+q]=color
            time.sleep(wait)
    time.sleep(wait)
    for k in range(0,2,1):
        for i in range(0,len(pixels),2):
            pixels[i+k]=(0,0,0)
            time.sleep(wait)
    time.sleep(wait)

'''This Function Needs To Be Defined'''
def theaterChaseX(pixels,color,x):
    y=0
    while y<x:
        theaterChase(pixels,color,.25)
        y+=1

'''This Function Needs To Be Defined'''
def theaterChaseRainbow(pixels):
    for j in range(0,255,10):
        for i in range(len(pixels)):
            idx = int(i + j)
            color=colorwheel(idx & 255)
            theaterChase(pixels,color,.05)