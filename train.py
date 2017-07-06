import json, sys, collections, time, pickle
import numpy as np
from sklearn import linear_model, model_selection, preprocessing, metrics


def read_data(filename):
    """ Reads a JSON file containing n training examples.
         Prints all the labels and number of occurences of each label.
        Returns: two n-length arrays, one with a 4,800-sized int list
         in each entry and one with the labels
    """
    label_dict = collections.defaultdict(int)
    try:
        with open(filename, 'r') as file:
            json_data = json.load(file)
            data = []
            labels = []
            for item in json_data:
                data.append(item['data'])
                labels.append(item['label'])
                label_dict[item['label']] += 1
        print "Imported dataset with this distribution - ", len(data), "characters in total."
        for key in sorted(label_dict):
            print key, label_dict[key]
        return np.array(data), labels
    except IOError:
        print "Could not read from", filename
        exit(1)


def train_model(datafile):
    """ Split data in datafile to to train and test set. Train model and
         evaluate its performance.
        Returns the classifier so that it can be stored by the calling functions.
    """
    # Read data from file
    data, labels = read_data(datafile)
    
    # Create a label encoder to map characters to an int as the classifier
    #  can only work with ints.
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    # Use a random state based on current time to split into train and test.
    #  Can be set to a fixed value for predictable outcomes.
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split( \
        data, le.transform(labels), test_size=0.33, random_state=int(time.time()))
    clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, \
        fit_intercept=True, intercept_scaling=1, class_weight=None, \
        random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', \
        verbose=0, warm_start=False, n_jobs=1)

    # Train the model with imported data
    clf.fit(train_data, train_labels)

    # Do prediction on test data
    predicted_labels = clf.predict(test_data)
    
    # Log the report to terminal.
    # Example for character 'H'
    #  * precision - how many of the characters the model classified as 'H'
    #     was actually the character 'H'
    #  * recall - how many of the characters 'H' in the test set did the
    #     model manage to correctl classify as 'H'
    #  * support - number of actual occurences of 'H' in the test set
    #  * f1-score - google it!
    print "ACTUAL   ", le.inverse_transform(test_labels)
    print "PREDICTED", le.inverse_transform(predicted_labels)
    print "Classification report"
    print metrics.classification_report(le.inverse_transform(test_labels),\
         le.inverse_transform(predicted_labels))

    return clf, le


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "Usage: python train.py file_with_train_data.json [file_prefix_for_classifier_and_label_encoder]"
    else:
        # Train model and print evaluation scores
        clf, le = train_model(sys.argv[1])
        if len(sys.argv) == 3:
            model_file = sys.argv[2]+"_model.pickle"
            le_file = sys.argv[2]+"_le.pickle"
            if raw_input("Save to classifier and encoder to " + model_file + \
                         ", " + le_file + " y/n: ") not in ['y', 'Y']:
                exit(0)
            else:
                with open(model_file, 'wb') as file:
                    pickle.dump(clf, file)
                    print "Stored classifier in", model_file
                with open(le_file, 'wb') as file:
                    pickle.dump(le, file)
                    print "Stored label encoder in", le_file

        else:
            print "No output file given to store model - skipping store step."
