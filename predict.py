import sys, pickle
import numpy as np
import cv2
from sklearn import linear_model
import label

# Should be set to approximately the height of the image in pixels divided by number of test rows
#  Used to sort the characters according to their appearance in the image
ROW_DIVISOR = 400 / 5

def load_and_extract(inputfile):
    print "Loading", inputfile
    img = cv2.imread(inputfile)
    print "Image size is", img.shape
    preprocessed = label.preprocess(img)
    contours = label.get_contours(preprocessed)
    for contour in contours:
        cv2.rectangle(preprocessed,(contour[2], contour[0]), (contour[3], contour[1]), (128,128,0), 2)
    cv2.imshow(inputfile, preprocessed)
    print "Found", len(contours), "characters in image"
    return preprocessed, contours


def is_lower_contour(i, j):
    """ i is lower than j if it is higher up in the image and further
         to the left. Format is [y1, y2, x1, x2].
    """
    if i[0]/ROW_DIVISOR < j[0]/ROW_DIVISOR:
        return -1
    if i[2] < j[2]:
        return -1
    return 1


def classify(fileprefix, img, contours):
    """ Loads classification model and label encoder from file.
        Sorts the contours to get the character boxes approximately from top left
         to bottom right.
        Uses the model to predict each character and outputs them to the terminal. 
    """
    with open(fileprefix+"_model.pickle", 'r') as file:
        clf = pickle.load(file)
    print "Loaded classifier of type", type(clf)
    with open(fileprefix+"_le.pickle", 'r') as file:
        le = pickle.load(file)
    print "Loaded label encoder of type", type(le)
    test_data = []
    sorted_contours = sorted(contours, is_lower_contour)
    for contour in sorted_contours:
        test_data.append(np.array(img[contour[0]:contour[1], contour[2]:contour[3]]).flatten()/255)
    print le.inverse_transform(clf.predict(test_data))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python predict.py image.png classifier_and_encoder_prefix"
        exit(1)
    img, contours = load_and_extract(sys.argv[1])
    classify(sys.argv[2], img, contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
