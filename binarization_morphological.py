import sys
import os
import logging
import re
from datetime import datetime

import cv2
import math
import numpy

import sauvola   # For Sauvola Binarization

from PIL import Image # For save binary image

import ntpath


##############
# Set logger #
##############
syslogger = logging.getLogger("sl")
reslogger = logging.getLogger("rl")
#syslogger.propagate = False # now logger will not log to console.
syslogger.setLevel(logging.INFO)
reslogger.setLevel(logging.INFO)
# Formatting log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Log to file

file_handler = logging.FileHandler(datetime.now().strftime('Binarization_%Y-%m-%d-%H%M%S.log'))
file_handler.setFormatter(formatter)
syslogger.addHandler(file_handler)
res_file_handler = logging.FileHandler(datetime.now().strftime('Binarization_result_%Y-%m-%d-%H%M%S.log'))
reslogger.addHandler(res_file_handler)

def IsFileExist(filepath):
    if not os.path.isfile(filepath):
        syslogger.error("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

def IsFileEmpty(filepath):
    if os.stat(filepath).st_size == 0:
        syslogger.error("File {} is empty. Exiting...".format(filepath))
        sys.exit()
        
def findfiles(path, regex):
    regObj = re.compile(regex)
    res = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if regObj.match(fname):
                res.append(os.path.join(root, fname))
    if len(res)==0:
        syslogger.error("No image is found in \"{}\". Exiting...".format(path))
        sys.exit()
    return res

def ReadImageLists(filepath):
    with open(filepath) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def main():
    ##############
    # Get images #
    ##############
    filePath = sys.argv[1]
    imagePaths = findfiles(filePath, r'.*.tif')
    #imagePaths = ReadImageLists('image_list.txt')

    outputPath = os.path.abspath("./outputs")
    
    #################
    # Start process #
    #################
    syslogger.info("Starting Binarization...")
    cnt = 0
    for imagePath in imagePaths:
        syslogger.info("Processing image {}, \"{}\"".format(cnt,imagePath))
        cnt += 1
        # call processing
        try:
            greyscaleImage = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            t_sauvola = sauvola.threshold_sauvola(greyscaleImage, window_size=15, k=0.2)
            binaryImage = greyscaleImage > t_sauvola
            binaryImage = (binaryImage*255).astype(numpy.uint8)
            binaryImage = cv2.bitwise_not(binaryImage)

            # Morphological Transformation
            bw = binaryImage
            kernel = numpy.ones((3,1), numpy.uint8)  # note this is a horizontal kernel
            d_im = cv2.dilate(bw, kernel, iterations=1)
            e_im = cv2.erode(d_im, kernel, iterations=1) 
            binaryImage = e_im

            #save
            mode='1'
            binaryImage_ = Image.fromarray(cv2.bitwise_not(binaryImage))
            binaryImage_ = binaryImage_.convert('1')
            output_filename = "[BI]"
            #cv2.imwrite(os.path.join(os.path.abspath("./output"), output_filename + ntpath.basename(outputPath)), binaryImage_)
            #binaryImage_.save('[BI]1.tif',resolution=dpi)
            binaryImage_.save(os.path.join(outputPath, output_filename + ntpath.basename(imagePath)), format='TIFF', dpi=(300.,300.), compression='tiff_lzw')

            
        except Exception as e:
            syslogger.error(e)
            continue
    syslogger.info("End of file. Exiting...")

main()