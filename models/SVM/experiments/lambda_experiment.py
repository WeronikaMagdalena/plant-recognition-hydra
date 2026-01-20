import time
import numpy as np
import pandas as pd
from models import LinearSVM
from utils import DataLoader

class LambdaExperiment:
    def __init__(self, lr=0.0001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

    def run(self, data_type="balanced"):
        data = self._load_data(data_type)
        if data is None:
            return None
        X_train, y_train, X_test, y_test = data
        results = self._test_lambda_values(X_train, y_train, X_test, y_test)
        best_lambda, best_acc = self._find_best(results)
        output_file = self._save_results(results, data_type)
        return self._create_summary(data_type, output_file, best_lambda, best_acc)

    def _load_data(self, data_type):
        try:
            return DataLoader().load(data_type)
        except FileNotFoundError:
            return None

    def _test_lambda_values(self, X_train, y_train, X_test, y_test):
        results = []
        for lmb in self.lambda_values:
            result = self._train_with_lambda(lmb, X_train, y_train, X_test, y_test)
            results.append(result)
        return results

    def _train_with_lambda(self, lmb, X_train, y_train, X_test, y_test):
        model = LinearSVM(lr=self.lr, lmb=lmb, epochs=self.epochs)
        duration = self._fit_and_time(model, X_train, y_train)
        accuracies = self._compute_accuracies(model, X_train, y_train, X_test, y_test)
        return self._create_result_dict(lmb, accuracies, duration)

    def _fit_and_time(self, model, X_train, y_train):
        start_time = time.time()
        model.fit(X_train, y_train)
        return time.time() - start_time

    def _compute_accuracies(self, model, X_train, y_train, X_test, y_test):
        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        return train_acc, test_acc

    def _create_result_dict(self, lmb, accuracies, duration):
        train_acc, test_acc = accuracies
        return {
            "Lambda": lmb,
            "Train_Accuracy": train_acc,
            "Test_Accuracy": test_acc,
            "Time_Seconds": round(duration, 2)
        }

    def _find_best(self, results):
        best_result = max(results, key=lambda x: x['Test_Accuracy'])
        return best_result['Lambda'], best_result['Test_Accuracy']

    def _save_results(self, results, data_type):
        df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/{data_type}_results_exp2_lambda_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename

    def _create_summary(self, data_type, output_file, best_lambda, best_acc):
        return {
            'data_type': data_type,
            'output_file': output_file,
            'best_lambda': best_lambda,
            'best_accuracy': best_acc
        }
