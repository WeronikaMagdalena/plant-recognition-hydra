from .data_loader import DataLoader
from .result_saver import ResultSaver
from .data_utils import DataUtils

class Convenience:
    @staticmethod
    def load_probes_data(data_type="balanced"):
        return DataLoader().load(data_type)

    @staticmethod
    def save_test_results(y_test, y_pred, filename, data_type="balanced"):
        return ResultSaver().save(y_test, y_pred, filename, data_type)

    @staticmethod
    def evaluate(model, X, y):
        return DataUtils.evaluate(model, X, y)

    @staticmethod
    def split_data(X, y, train_size=100):
        return DataUtils.split_data(X, y, train_size)

    @staticmethod
    def shuffle_data(X, y, seed=42):
        return DataUtils.shuffle_data(X, y, seed)

    @staticmethod
    def print_header(title):
        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
