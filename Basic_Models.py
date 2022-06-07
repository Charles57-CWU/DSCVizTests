# import models
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# import metrics/utilities
from sklearn import metrics
import pandas as pd

# clear warnings
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# =====================================================
class BasicModels:
    def __init__(self, training_name, validation_name):
        self.training_name = training_name
        self.validation_name = validation_name

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    # csv to data and labels
    def get_data_and_labels(self):
        # read the training and validation files
        training_split = pd.read_csv(self.training_name)
        validation_split = pd.read_csv(self.validation_name)

        # labels
        self.y_train = training_split['class']
        self.y_val = validation_split['class']

        training_split = training_split.drop(['class'], axis=1)
        validation_split = validation_split.drop(['class'], axis=1)

        # data
        features = list(training_split.columns[:])
        print(features)
        self.x_train = training_split[features]
        self.x_val = validation_split[features]

    # =====================================================

    # run through tests
    def run_tests(self):
        # results dataframe
        results = pd.DataFrame(columns=['Model', 'ACC'])

        # run through models you select, easy to add more
        models = ['DT', 'SGD', 'NB', 'SVM', 'KNN', 'LR', 'MLP', 'RF']
        for model in models:
            if model == 'DT':
                clf = tree.DecisionTreeClassifier()
            elif model == 'SGD':
                clf = SGDClassifier()
            elif model == 'RF':
                clf = RandomForestClassifier()
            elif model == 'NB':
                clf = GaussianNB()
            elif model == 'SVM':
                clf = svm.SVC()
            elif model == 'KNN':
                clf = KNeighborsClassifier()
            elif model == 'LR':
                clf = LogisticRegression()
            elif model == 'MLP':
                clf = MLPClassifier()
            elif model == 'LDA':
                clf = LinearDiscriminantAnalysis()
            else:
                clf = None

            # run model
            clf = clf.fit(self.x_train, self.y_train)

            # predictions
            y_pred = clf.predict(self.x_val)

            # accuracy
            _acc = '{:.2f}'.format(metrics.accuracy_score(self.y_val, y_pred))
            print(model + ' accuracy: ' + str(_acc))

            # add result
            results.loc[(len(results.index))] = [model, _acc]

        return results


# =====================================================


if __name__ == '__main__':
    # input filename
    training_name = 'LUPI_WDBC_NOT_SV.csv'
    validation_name = 'LUPI_WDBC_SV.csv'

    # output filename
    output_csv_name = 'CWU_RESULTS.csv'

    m = BasicModels(training_name, validation_name)

    # run models
    m.get_data_and_labels()
    results = m.run_tests()

    # output results
    results.to_csv(output_csv_name, index=False)