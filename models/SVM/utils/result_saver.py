import time
import numpy as np
import pandas as pd

class ResultSaver:
    def save(self, y_test, y_pred, filename, data_type="balanced"):
        results_df = self._create_dataframe(y_test, y_pred)
        output_filename = self._generate_filename(filename, data_type)
        results_df.to_csv(output_filename, index=False)
        accuracy = self._calculate_accuracy(y_test, y_pred)
        return output_filename, accuracy

    def _create_dataframe(self, y_test, y_pred):
        return pd.DataFrame({
            'sample_index': np.arange(len(y_test)),
            'correct_label': y_test,
            'predicted_label': y_pred,
            'is_correct': (y_test == y_pred).astype(int)
        })

    def _generate_filename(self, filename, data_type):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = filename.replace('.csv', '')
        return f"results/{data_type}_{base_name}_{timestamp}.csv"

    def _calculate_accuracy(self, y_test, y_pred):
        return np.mean(y_test == y_pred) * 100
