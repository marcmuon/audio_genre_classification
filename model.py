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
        self.holdout_test_set = None
        self.holdout_val_sets = None

        # populated in .predict_from_holdout()
        self.fnrs = None
        self.fprs = None
        self.accuracies = None
        self.y_preds = None


    def run_repeated_kfold(self, n_repeats=1):
        """
        Using Pipeline objects as they don't leak transformations
        into the validation folds as shown here: https://bit.ly/2N7rdQ0,
        and here: https://bit.ly/346THQL

        Note that return_train_score=True and verbose=3 in GridSearchCV
        is useful for debugging.
        """

        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)

        self.best_estimators = []
        self.holdout_test_set = []
        self.holdout_val_sets = []

        # Save a holdout test set that WON'T go through RepeatedKFold
        # We will not fit any paramter choices to the holdout test set
        X_cv, X_test, y_cv, y_test = train_test_split(
                self.X,
                self.y,
                random_state=42,
                stratify=self.y,
                **self.cfg['tt_test_dict'])

        self.holdout_test_set.append((X_test, y_test))

        for i in range(n_repeats):

            X_train, X_val, y_train, y_val = train_test_split(
                X_cv,
                y_cv,
                random_state=i,
                stratify=y_cv,
                **self.cfg['tt_val_dict'])

            # Note these val sets won't go into GridSearchCV
            # We'll predict on these in the .predict_from_holdout() method
            self.holdout_val_sets.append((X_val, y_val))
            
            pipe = Pipeline([
                ('scaler', self.cfg['scaler']),
                ('model', self.cfg['base_model'])
            ])

            kf = StratifiedKFold(**self.cfg['kf_dict'])

            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=self.cfg['param_grid'],
                                       cv=kf,
                                       return_train_score=True,
                                       verbose=3,
                                       **self.cfg['grid_dict'])

            # refit the best estimator on the FULL train set
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_

            self.best_estimators.append(best_estimator)


    def _parse_conf_matrix(self, cnf_matrix):
        TP = np.diag(cnf_matrix)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        
        TP = TP.astype(float)
        FP = FP.astype(float)
        TN = TN.astype(float)
        FN = FN.astype(float)

        return TP, FP, TN, FN


    def predict_from_holdout(self):

        self.fprs = []
        self.fnrs = []
        self.accuracies = []
        self.y_preds = []

        for i, val in enumerate(self.holdout_val_sets):
            X_val, y_val = val[0], val[1]

            scaler = self.best_estimators[i]['scaler']
            model = self.best_estimators[i]['model']

            X_val_scaled = scaler.transform(X_val)
            y_pred = model.predict(X_val_scaled)
            cnf_matrix = confusion_matrix(y_val, y_pred)

            TP, FP, TN, FN = self._parse_conf_matrix(cnf_matrix)
            
            print(f'Val Set from Trial Number {i}, per class:')
            print(f'TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')

            self.fprs.append(FP / (FP + TN))
            self.fnrs.append(FN / (TP + FN))
            self.accuracies.append((TP + TN) / (TP + TN + FP + FN))


        print(f'Avg False Positive Rate per Class Across Trials: {np.mean(self.fprs, axis=0)}')
        print(f'Avg False Negative Rate per Class, Across Trials: {np.mean(self.fnrs, axis=0)}')
        print(f'Avg Accuracy per Class, Across Trials: {np.mean(self.accuracies, axis=0)}')
        print("")

    def predict_from_test(self):

        X_test, y_test = self.holdout_test_set[0][0], self.holdout_test_set[0][1]
        scaler = self.best_estimators[0]['scaler']
        model = self.best_estimators[0]['model']

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        cnf_matrix = confusion_matrix(y_test, y_pred)

        TP, FP, TN, FN = self._parse_conf_matrix(cnf_matrix)
        
        print(f'Test Set, per class:')
        print(f'TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')

        print(f'Test False Positive Rate per Class: {FP / (FP + TN)}')
        print(f'Test False Negative Rate per Class: {FN / (TP + FN)}')
        print(f'Test Accuracy per Class: {(TP + TN) / (TP + TN + FP + FN)}')