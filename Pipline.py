
from unittest import result
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import time
import math
import seaborn as sns
import matplotlib.pyplot as plt
class Pipeline():
    def __init__(self, stratified: bool = False, n_repeated: int = 5, n_splits: int = 5):
        """This function is used to initialize the class Pipeline

        Args:
            stratified (bool): define if pipeline should use stratified split or not
            n_repeated (int): number of repetitions in the cross val
            n_splits (int): number of split in the kfold
        """
        #here the parameters will be set for the crossvalidation
        self.stratified = stratified
        self.n_repeated = n_repeated
        self.n_splits = n_splits


    def hyperparameter_tuning(self, model,x_train:pd.DataFrame, y_train:pd.DataFrame, hyperparameters: dict, scoring: str, random_state: int =42):
        """This function will use (stratified) Repeated k-fold and GridsearchCV to find the best hyperparameters for the given training data and hyperparameters.

        Args:
            model (sklearn model): This is the model to search the best hyperparameters for
            x_train (pd.DataFrame): Training Data Features
            y_train (pd.DataFrame): Training Data Targets
            stratified (bool): if the data should be stratified or not
            hyperparameters (dict): hyperparamters that should be tested in the evaluation
            scoring (str): For which value the hyperparameter should be optimized (e.g. balanced_accuracy or accuracy)
            random_state (int, optional): Seed for the randomstate. Defaults to 42.

        Returns:
            object: contains multiple information including the trained model
        """
        if self.stratified:
            rskf = RepeatedStratifiedKFold(n_splits=self.n_splits,n_repeats=self.n_repeated, random_state=random_state)
            clf = GridSearchCV(model,hyperparameters,cv=rskf,scoring=scoring, n_jobs=-1)
        else:
            rkf = RepeatedKFold(n_splits=self.n_splits,n_repeats=self.n_repeated, random_state=random_state)
            clf = GridSearchCV(model,hyperparameters,cv=rkf,scoring=scoring, n_jobs=-1)
        clf.fit(x_train,y_train)
        return clf

    def model_evaluation_classification(trained_model, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """this function generates the predictions of a given model and calculates the accuracy as well as the balanced accuracy score

        Args:
            trained_model (sklearn trained model): previously trained model
            x_test (pd.DataFrame): festures test data
            y_test (pd.DataFrame): target test data

        Returns:
            float: accuracy_score
            float: balanced_accuracy_score
            float: ras score
            float: f1 score 
        """
        y_hat = trained_model.predict(x_test)
        asc = accuracy_score(y_test, y_hat)
        basc = balanced_accuracy_score(y_test, y_hat)
        ras= roc_auc_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        return asc, basc, ras, f1

    def model_evaluation_float(trained_model, x_test, y_test):
        """this function generates the predictions of a given model and calculates the accuracy as well as the balanced accuracy score

        Args:
            trained_model (sklearn trained model): previously trained model
            x_test (pd.DataFrame): festures test data
            y_test (pd.DataFrame): target test data

        Returns:
            float: Mean squared error
            float: Mean absolut error
            
        """
        y_hat = trained_model.predict(x_test)
        mse = mean_squared_error(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        return mse, mae, y_hat

    def confusion_matrix(trained_model, x_test, y_test):
        """This function generates a confusion matrix and prints the matrix as well as return the confusion matrix in a confusion matrix

        Args:
            trained_model (sklearn model): the trained model to evaluate
            x_test (pd.DataFrame): test data features
            y_test (pd.DataFrame): test data class

        Returns:
            array: returns a confusion matrix
        """        
        y_hat = trained_model.predict(x_test)
        cm = confusion_matrix(y_test,y_hat)
        ConfusionMatrixDisplay(cm)
        return cm




    def run_experiments_float(self, experiments, X, Y, Scaler,scoring, random_state:int =42, shuffle:bool = True, test_size: float = 0.3):
        """This function runs the predefined experiments on a dataset for Regression problems

        Args:
            experiments (List with nested dicts): Here the experiments are defined 
            X (_type_): The dataset that contains the features
            Y (_type_): The dataset that contains the dependent variable
            Scaler (sklearn.scaler): The scaler that is applied to the data 
            scoring (sklearn.metrics): The metric that should be used to optimize the model
            random_state (int, optional): The randomstate used in the experimentation. Defaults to 42.
            shuffle (bool, optional): Defines if the data should be shuffeled before it is being split up. Defaults to True.

        Returns:
            list that contains dicts: Returns the results put into a dicitionarry format
        """
        X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = random_state, shuffle=shuffle, test_size=test_size)
        X_train_scaled = Scaler.fit_transform(X_train)
        X_test_scaled = Scaler.transform(X_test)
        df_results = []
        for experiment in experiments:
    
            start_time= time.time()

            print()
            print()
            print(experiment['name'])
            print("-----------------")
            trained_model = self.hyperparameter_tuning(model=experiment["model"],x_train=X_train_scaled, y_train=y_train, hyperparameters= experiment["parameters"], scoring=scoring, random_state=random_state)
            mse, mae, y_hat = Pipeline.model_evaluation_float(trained_model=trained_model, x_test=X_test_scaled, y_test=y_test)
            end_time = time.time() - start_time
            #print(f'Optimized for {scoring} the {experiment['name']} achieved the following scores:')
            print(f'RMSE: {math.sqrt(mse)}')
            print(f'MAE: {mae}')
            print(f'The best results achieved with parameters: {trained_model.best_params_}')
            print(f'Time: {int(round(end_time, 1))} seconds ({int(round(end_time/60, 1))} minutes)')
            df_results.append({"name": experiment["name"],
            "trained_model": trained_model,
            "best_results": {
                "MSE": mse, 
                "MAE":mae,
                "predicted_values":y_hat,
                "actual_values": y_test
            }})
            ax = sns.scatterplot(y_test,y_hat)
            ax.set(ylim=(min([min(y_test), min(y_hat)]), max([max(y_test), max(y_hat)])),xlim= (min([min(y_test), min(y_hat)]),max([max(y_test), max(y_hat)])))
            plt.show()
        return df_results


    def run_experiments_classification(self, experiments, X, Y, Scaler,scoring, random_state:int =42, shuffle:bool = True, test_size: float = 0.3):
        """This function runs the predefined experiments on a dataset for Classification problems

        Args:
            experiments (List with nested dicts): Here the experiments are defined 
            X (_type_): The dataset that contains the features
            Y (_type_): The dataset that contains the dependent variable
            Scaler (sklearn.scaler): The scaler that is applied to the data 
            scoring (sklearn.metrics): The metric that should be used to optimize the model
            random_state (int, optional): The randomstate used in the experimentation. Defaults to 42.
            shuffle (bool, optional): Defines if the data should be shuffeled before it is being split up. Defaults to True.

        Returns:
            list that contains dicts: Returns the results put into a dicitionarry format
        """
        X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = random_state, shuffle=shuffle, test_size=test_size)
        X_train_scaled = Scaler.fit_transform(X_train)
        X_test_scaled = Scaler.transform(X_test)
        df_results = []
        for experiment in experiments:
    
            start_time= time.time()

            print()
            print()
            print(experiment['name'])
            print("-----------------")
            trained_model = self.hyperparameter_tuning(model=experiment["model"],x_train=X_train_scaled, y_train=y_train, hyperparameters= experiment["parameters"], scoring=scoring, random_state=random_state)
            asc, basc, ras, f1 = Pipeline.model_evaluation_classification(trained_model, x_test=X_test_scaled, y_test=y_test)
            df_results = "test"
            end_time = time.time() - start_time
            #print(f'Optimized for {scoring} the {experiment['name']} achieved the following scores:')
            print(f'Accuracy: {asc}')
            print(f'Balanced Accuracy: {basc}')
            print(f'F1 Score: {f1}')
            print(f'RAS: {ras}')
            print(f'The best results achieved with parameters: {trained_model.best_params_}')
            print(f'Time: {int(round(end_time, 1))} seconds ({int(round(end_time/60, 1))} minutes)')
            df_results.append({"name": experiment["name"],
            "trained_model": trained_model.cv_results_["params"],
            "best_results": {
                "accuracy": asc, 
                "balanced_accuracy":basc,
                "RAS": ras, 
                "f1_score":f1
            }})
        return df_results
 