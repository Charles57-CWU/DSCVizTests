"""
This script will run cross validation against various classification models from the scikit-learn library
Add column headers, fix missing values, and make sure class labels are in numeric form, before running models
Make sure there is only one class column, and the remaining columns are attributes you'd like to run in the model
"""


# import models
from sklearn.tree import DesicionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier


# import metrics/utilities
import pandas as pd
from sklearn.model_selection import cross_validate

# clear warnings
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# =====================================================
class CrossValidateModels:
    def __init__ (self, dataset_name, class_column_name):
        self.dataset_name = dataset_name
        self.class_column_name = class_column_name

        self.x = None
        self.y = None

    # csv to data and labels
    def get_data_and_labels(self):
        # read the dataset file
        data = pd.read_csv(self.dataset_name)

        # labels
        self.y = data[self.class_column_name]

        data = data.drop([self.class_column_name], axis=1)

        # data
        features = list(data.columns[:])
        print(features)
        self.x = data[features]

    # =====================================================

    # run through tests
    def run_tests(self, number_of_folds):
        # create dataframe header
        fold_names = ['Model']
        for i in range(number_of_folds):
            fold_names.append('Fold_' + str(i+1))
        fold_names.append('AVG')
        print(fold_names)

        # results dataframe
        results = pd.DataFrame(columns=fold_names)

        # run through models you select, easy to add more
        models = ['DT', 'SGD', 'NB', 'SVM', 'KNN', 'LR', 'MLP', 'RF']
        for model in models:
            if model == 'DT':
                clf = DecisionTreeClassifier()
            elif model == 'SGD':
                clf = SGDClassifier()
            elif model == 'RF':
                clf = RandomForestClassifier()
            elif model == 'NB':
                clf = GaussianNB()
            elif model == 'SVM':
                clf = SVC()
            elif model == 'KNN':
                clf = KNeighborsClassifier()
            elif model == 'LR':
                clf = LogisticRegression()
            elif model == 'MLP':
                clf = MLPClassifier()
            else:
                clf = None

            # run cross validation
            cv = cross_validate(clf, self.x, self.y, cv=number_of_folds)
            # get scores
            _splits = list(map(lambda n: '%.2f' % n, cv['test_score']))
            # get average
            _avg = '{:.2f}'.format(cv['test_score'].mean())

            # combine into dataframe row
            row = [model]
            for split in _splits:
                row.append(split)
            row.append(_avg)
            print(row)

            # add result
            results.loc[(len(results.index))] = row

        return results

# =====================================================


if __name__ == '__main__':
    # input filename
    dataset_name = 'iris_dataset.csv'
    class_column_name = 'class'

    # number of folds for CV
    number_of_folds = 3

    # output filename
    output_csv_name = 'CWU_RESULTS1.csv'

    m = CrossValidateModels(dataset_name, class_column_name)

    # run models
    m.get_data_and_labels()
    results = m.run_tests(number_of_folds)

    # output results
    results.to_csv(output_csv_name, index=False)
