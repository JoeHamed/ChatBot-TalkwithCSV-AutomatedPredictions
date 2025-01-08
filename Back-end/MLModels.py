import tensorflow as tf
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class MLModels:
    @staticmethod
    def kernel_svm_model(X_train, X_test, y_train, y_test, params):
        svm_classifier = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'],
                             gamma=params['gamma'], tol=params['tol'], max_iter=params['max_iter'])
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        k_cross_score = cross_val_score(svm_classifier, X_train, y_train, cv=5)
        report = classification_report(y_test, y_pred)

        return cm, acc_score, k_cross_score, report

    @staticmethod
    def logistic_regression_model(X_train, X_test, y_train, y_test, params):
        lr_classifier = LogisticRegression(C=params['C'], max_iter=params['max_iter'], solver=params['solver'],
                                           penalty=params['penalty'], fit_intercept=params['fit_intercept'],
                                           tol=params['tol'])
        lr_classifier.fit(X_train, y_train)
        y_pred = lr_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        k_cross_score = cross_val_score(lr_classifier, X_train, y_train, cv=5)
        report = classification_report(y_test, y_pred)

        return cm, acc_score, k_cross_score, report

    @staticmethod
    def random_forest_model(X_train, X_test, y_train, y_test, params):
        rf_classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                               criterion='entropy',min_samples_split=params['min_samples_split'],
                                               min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'],
                                               bootstrap=params['bootstrap'])
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        k_cross_score = cross_val_score(rf_classifier, X_train, y_train, cv=5)
        report = classification_report(y_test, y_pred)

        return cm, acc_score, k_cross_score, report


    @staticmethod
    def tensorflow_model(X_train, X_test, y_train, y_test, params):

        print(len(X_train))
        print(len(y_train))
        # Ensure proper shapes
        X_train, X_test = np.array(X_train), np.array(X_test)
        print(X_train.shape)
        print(X_test.shape)
        y_train, y_test = np.squeeze(np.array(y_train)), np.squeeze(np.array(y_test))
        print(y_train.shape)
        print(y_test.shape)

        layer_configs, optimizer, learning_rate, batch_size, epochs = params['layer_configs'], params['optimizer'], params[
            'learning_rate'], params['batch_size'], params['epochs']

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.InputLayer(shape=(X_train.shape[1],)))

        # Hidden layers
        for config in layer_configs:
            model.add(tf.keras.layers.Dense(
                units=config['size'],
                activation=config['activation'],
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization'])
            ))
            if config['dropout_rate'] > 0:
                model.add(tf.keras.layers.Dropout(config['dropout_rate']))

        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Optimizer
        opt = None
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'adagrad':
            opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Train
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

        # Predictions and metrics
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        return cm, acc_score, report


    @staticmethod
    def xgboost_model(X_train, X_test, y_train, y_test, params):
        xgb_classifier = XGBClassifier(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                max_depth=params['max_depth'],subsample=params['subsample'],
                                colsample_bytree=params['colsample_bytree'], gamma=params['gamma'],
                                reg_lambda=params['reg_lambda'],reg_alpha=params['reg_alpha'])
        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        k_cross_score = cross_val_score(xgb_classifier, X_train, y_train, cv=5)
        report = classification_report(y_test, y_pred)

        return cm, acc_score, k_cross_score, report


    @staticmethod
    def catboost_model(X_train, X_test, y_train, y_test, params):
        cat_classifier = CatBoostClassifier(learning_rate=params['learning_rate'], iterations=params['iterations'],
                                            depth=params['depth'],l2_leaf_reg=params['l2_leaf_reg'],
                                            bagging_temperature=params['bagging_temperature'],border_count=params['border_count'],
                                            random_strength=params['random_strength'], silent=True)
        cat_classifier.fit(X_train, y_train)
        y_pred = cat_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        k_cross_score = cross_val_score(cat_classifier, X_train, y_train, cv=5)
        report = classification_report(y_test, y_pred)

        return cm, acc_score, k_cross_score, report
