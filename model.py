from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Model:

    def __init__(self, feature_matrix, labels, cfg):

        self.X = feature_matrix
        self.y = labels
        self.cfg = cfg

        # populated in .run_cv_trials()
        self.best_estimators = None
        self.holdout_test_sets = None

        # populated in .predict_from_holdout()
        self.fnr = None
        self.fpr = None
        self.accuracy = None
        self.y_preds = None

    def run_cv_trials(self, n_trials=1):
        """
        Using Pipeline objects as they don't leak transformations
        into the validation folds as shown here: https://bit.ly/2N7rdQ0,
        and here: https://bit.ly/346THQL

        The trials iteration creates a unique train_test_split, hence
        a unique heldout test. We save this test set for predictions
        later on, and then we'll average the heldout test predictions
        (via their associated best_estimator) for a better sense of
        generalization error.

        Note that return_train_score=True and verbose=3 in GridSearchCV
        is useful for debugging as you'll see if your models are overfitting
        badly.
        """

        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)

        self.best_estimators = []
        self.holdout_test_sets = []

        for i in range(n_trials):

            X_cv, X_test, y_cv, y_test = train_test_split(
                self.X,
                self.y,
                random_state=i,
                stratify=self.y,
                **self.cfg['tt_dict'])

            pipe = Pipeline([
                ('scaler', self.cfg['scaler']),
                ('model', self.cfg['base_model'])
            ])

            kf = StratifiedKFold(**self.cfg['kf_dict'])

            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=self.cfg['param_grid'],
                                       cv=kf,
                                       scoring='balanced_accuracy',
                                       return_train_score=True,
                                       verbose=3,
                                       **self.cfg['grid_dict'])

            grid_search.fit(X_cv, y_cv)
            best_estimator = grid_search.best_estimator_

            self.best_estimators.append(best_estimator)
            self.holdout_test_sets.append((X_test, y_test))

    def predict_from_holdout(self):

        self.fpr = []
        self.fnr = []
        self.accuracy = []
        self.y_preds = []

        for i, test in enumerate(self.holdout_test_sets):

            X_test, y_test = test[0], test[1]
            scaler = self.best_estimators[i]['scaler']
            model = self.best_estimators[i]['model']

            X_test_scaled = scaler.transform(X_test)

            y_pred = model.predict(X_test_scaled)

            cnf_matrix = confusion_matrix(y_test, y_pred)

            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

            print(f'Test Set from Trial Number {i}, per class:')
            print(f'TN:{TN}, FP:{FP}, FN:{FN}, TP:{TP}')

            self.fpr.append(FP / (FP + TN))
            self.fnr.append(FN / (TP + FN))
            self.accuracy.append((TP + TN) / (TP + TN + FP + FN))

        for i in range(len(self.fpr)):

            print(f'False Positive Rate per Class, Trial {i}: {self.fpr}')
            print(f'False Negative Rate per Class, Trial {i}: {self.fnr}')
            print(f'Accuracy per Class, Trial {i}: {self.accuracy}')

