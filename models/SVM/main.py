import threading
from datetime import datetime
from experiments import run_experiment_detailed, run_experiment_lambda

class ExperimentRunner:
    def __init__(self):
        self.experiments = [
            (run_experiment_detailed, "balanced"),
            (run_experiment_lambda, "balanced"),
            (run_experiment_detailed, "unbalanced"),
            (run_experiment_lambda, "unbalanced")
        ]

    def run_all(self):
        start_time = datetime.now()
        print(f"All experiments started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._execute_parallel()
        end_time = datetime.now()
        print(f"Experiments ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, CSV files saved successfully.")

    def _execute_parallel(self):
        threads = self._create_threads()
        self._start_threads(threads)
        self._join_threads(threads)

    def _create_threads(self):
        return [
            threading.Thread(target=self._run_experiment, args=(func, dtype))
            for func, dtype in self.experiments
        ]

    def _start_threads(self, threads):
        for thread in threads:
            thread.start()

    def _join_threads(self, threads):
        for thread in threads:
            thread.join()

    def _run_experiment(self, experiment_func, data_type):
        experiment_func(data_type)

if __name__ == "__main__":
    ExperimentRunner().run_all()
