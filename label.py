import sys, json, copy
import cv2
import numpy as np

# For the classifier to work the matrices with train data have to have 
#  exactly these dimensions. The pictures need to be of a size so that
#  the characters are "about" that size.
CHAR_WIDTH = 60
CHAR_HEIGHT = 80


def preprocess(img):
    """ Preprocess the image to black and white. Makes it easier to find the shapes.
        Returns: The processed image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #stackImageWindow('hsv', hsv)
    h,s,v = cv2.split(hsv)
    v = cv2.normalize(v, v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #stackImageWindow('v', v)
    retval, threshold = cv2.threshold(v, 0.95, 1.0, cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    eroded = cv2.erode(threshold,kernel,iterations = 2)
    processed = cv2.dilate(eroded,kernel,iterations = 8)
    return np.uint8(processed * 255)


def get_contours(img):
    """ Find contours of objects in the image that are within a size of a character.
        Create coordinates that defines a box, centered on this character.
        Returns: A Python list with all boxes. 
    """
    im2, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    char_contours = []
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # determine if the shape is of the right size, if so save it's limits in char_contours
        if w >= int(CHAR_WIDTH*0.6) and h >= int(CHAR_HEIGHT*0.75):
            corr_x = (CHAR_WIDTH-w)/2
            corr_y = (CHAR_HEIGHT-h)/2
            char_contours.append([y-corr_y, y+CHAR_HEIGHT-corr_y, x-corr_x, \
                          x+CHAR_WIDTH-corr_x])
    return char_contours


def store_to_json(labeled_chars, filename):
    """ Try to open an existing file and read data from there.
        Append new data to the existing and save it back to the file.
    """
    try:
        with open(filename, 'r') as file:
            train_data = json.load(file)
    except ValueError:
        print "File", filename, "is empty or does not contain valid JSON data"
        train_data = []
    except IOError:
        print "File", filename, "does not exist. Will be created."
        train_data = []

    for (label, image) in labeled_chars:
        new_item = {
            'label': label,
            'data': (image/255).flatten().tolist()
        }
        train_data.append(new_item)

    with open('data.json', 'w') as file:
        json.dump(train_data, file)


def label_image(img, contours):
    """ Use the character boxes defined in contours to show each letter in the image
         and ask the user to classify that letter.
        Returns: an array with pairs of labels and arrays of the image.
    """
    print str(len(contours)), "character boxes in this image to classify"
    labeled_chars = []
    # Reversed order gives top left to bottom right
    for contour in reversed(contours):
        char_image = img[contour[0]:contour[1], contour[2]:contour[3]]
        img2 = copy.copy(img)
        cv2.rectangle(img2,(contour[2], contour[0]), (contour[3], contour[1]), (128,128,0), 2)
        cv2.imshow("What character?", img2)
        labeled_chars.append((chr(cv2.waitKey(0) & 255), char_image))
        cv2.destroyAllWindows()
    return labeled_chars


def load_and_extract(inputfile, outputfile):
    """ * load image from source file
        * preprocess
        * find contours (boxes containing characters)
        * ask the user to label each character
        * (optional) store the labels and data to a JSON file
    """
    print "Loading", inputfile
    img = cv2.imread(inputfile)
    print "Image size is", img.shape
    preprocessed = preprocess(img)
    contours = get_contours(preprocessed)
    labeled_chars = label_image(preprocessed, contours)
    labels = [l for l, i in labeled_chars]
    print "You entered characters:", " ".join(labels)
    if (outputfile != None):
        if raw_input("Save to " + outputfile + " y/n: ") not in ['y', 'Y']:
            return
        else:
            store_to_json(labeled_chars, outputfile)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python label.py image_to_label.png [outputfile.json]"
    elif len(sys.argv) == 2:
        print "No output file given, will just show boxes in image"
        load_and_extract(sys.argv[1], None)
    else:
        load_and_extract(sys.argv[1], sys.argv[2])





