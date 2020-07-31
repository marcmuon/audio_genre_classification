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
        self.best_estimator = None
        self.holdout_test_set = None
        self.holdout_val_set = None


    def train_kfold(self):
        """
        Using Pipeline objects as they don't leak transformations
        into the validation folds as shown here: https://bit.ly/2N7rdQ0,
        and here: https://bit.ly/346THQL

        Note that return_train_score=True and verbose=3 in GridSearchCV
        is useful for debugging.
        """

        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)

        # Save a holdout test set that WON'T go through RepeatedKFold
        # We will not fit any paramter choices to the holdout test set
        X_cv, X_test, y_cv, y_test = train_test_split(
                self.X,
                self.y,
                random_state=42,
                stratify=self.y,
                **self.cfg['tt_test_dict'])

        self.holdout_test_set = (X_test, y_test)

        # From the non-holdout-test data, split off a validation piece
        X_train, X_val, y_train, y_val = train_test_split(
            X_cv,
            y_cv,
            random_state=42,
            stratify=y_cv,
            **self.cfg['tt_val_dict'])

        # Note these val sets won't go into GridSearchCV
        # We'll predict on these in the .predict_from_val() method
        self.holdout_val_set = (X_val, y_val)
        
        pipe = Pipeline([
            ('scaler', self.cfg['scaler']),
            ('model', self.cfg['base_model'])
        ])

        # Use stratification within KFold Split inside GridSearchCV
        # I believe sklearn now defaults to StratifiedKFold if you pass
        # an integer to the cv argument in GridSearchCV, but it did not used to
        kf = StratifiedKFold(**self.cfg['kf_dict'])

        # Perform KFold many times according to our Param Grid Search
        grid_search = GridSearchCV(estimator=pipe,
                                    param_grid=self.cfg['param_grid'],
                                    cv=kf,
                                    return_train_score=True,
                                    verbose=3,
                                    **self.cfg['grid_dict'])

        # refit the best estimator on the FULL train set
        grid_search.fit(X_train, y_train)
        self.best_estimator = grid_search.best_estimator_


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


    def _predict_(self, holdout_type):
        if holdout_type == "val":
            X_holdout, y_holdout = self.holdout_val_set

        elif holdout_type == "test":
            X_holdout, y_holdout = self.holdout_test_set
        
        scaler = self.best_estimator['scaler']
        model = self.best_estimator['model']

        X_holdout_scaled = scaler.transform(X_holdout)
        y_pred = model.predict(X_holdout_scaled)
        cnf_matrix = confusion_matrix(y_holdout, y_pred)

        TP, FP, TN, FN = self._parse_conf_matrix(cnf_matrix)

        return TP, FP, TN, FN


    def predict(self, holdout_type):
        """
        Specify either "val" or "test" as a string arg
        """
        TP, FP, TN, FN = self._predict_(holdout_type)

        print(f'{holdout_type} Set, per class:')
        print(f'TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')

        print(f'{holdout_type} False Positive Rate per Class: {FP / (FP + TN)}')
        print(f'{holdout_type} False Negative Rate per Class: {FN / (TP + FN)}')
        print(f'{holdout_type} Accuracy per Class: {(TP + TN) / (TP + TN + FP + FN)}')