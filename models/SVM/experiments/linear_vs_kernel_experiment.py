import time
import numpy as np
import pandas as pd
from models import LinearSVM, KernelSVM
from utils import DataLoader

class LinearVsKernelExperiment:
    def __init__(self, lr=0.00001, lmb=0.0001):
        self.lr = lr
        self.lmb = lmb
        self.linear_epochs = 50
        self.kernel_epochs = 30
        self.gamma = 0.001

    def run(self, data_type="balanced"):
        data = self._load_data(data_type)
        if data is None:
            return None
        X_train, y_train, X_test, y_test = data
        linear_results = self._train_linear(X_train, y_train, X_test, y_test)
        kernel_results = self._train_kernel(X_train, y_train, X_test, y_test)
        output_file = self._save_results(y_test, linear_results, kernel_results, data_type)
        return self._create_summary(data_type, output_file, linear_results, kernel_results)

    def _load_data(self, data_type):
        try:
            return DataLoader().load(data_type)
        except FileNotFoundError:
            return None

    def _train_linear(self, X_train, y_train, X_test, y_test):
        model = LinearSVM(lr=self.lr, lmb=self.lmb, epochs=self.linear_epochs)
        model.fit(X_train, y_train)
        return self._evaluate_model(model, X_train, y_train, X_test, y_test)

    def _train_kernel(self, X_train, y_train, X_test, y_test):
        model = KernelSVM(
            lr=self.lr, lmb=self.lmb,
            epochs=self.kernel_epochs, gamma=self.gamma
        )
        model.fit(X_train, y_train)
        return self._evaluate_model(model, X_train, y_train, X_test, y_test)

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = np.mean(train_pred == y_train) * 100
        test_acc = np.mean(test_pred == y_test) * 100
        return test_pred, train_acc, test_acc

    def _save_results(self, y_test, linear_results, kernel_results, data_type):
        pred_lin, _, _ = linear_results
        pred_ker, _, _ = kernel_results
        df = self._create_results_dataframe(y_test, pred_lin, pred_ker)
        return self._write_csv(df, data_type)

    def _create_results_dataframe(self, y_test, pred_lin, pred_ker):
        return pd.DataFrame({
            'sample_index': np.arange(len(y_test)),
            'correct_label': y_test,
            'linear_pred': pred_lin,
            'linear_correct': (pred_lin == y_test).astype(int),
            'kernel_pred': pred_ker,
            'kernel_correct': (pred_ker == y_test).astype(int)
        })

    def _write_csv(self, df, data_type):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/{data_type}_linear_vs_kernel_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename

    def _create_summary(self, data_type, output_file, linear_results, kernel_results):
        _, lin_train_acc, lin_test_acc = linear_results
        _, ker_train_acc, ker_test_acc = kernel_results
        return {
            'data_type': data_type,
            'output_file': output_file,
            'linear_train_acc': lin_train_acc,
            'linear_test_acc': lin_test_acc,
            'kernel_train_acc': ker_train_acc,
            'kernel_test_acc': ker_test_acc
        }
