from .linear_vs_kernel_experiment import LinearVsKernelExperiment
from .lambda_experiment import LambdaExperiment

class ExperimentFactory:
    @staticmethod
    def run_experiment_detailed(data_type="balanced"):
        experiment = LinearVsKernelExperiment()
        return experiment.run(data_type)

    @staticmethod
    def run_experiment_lambda(data_type="balanced"):
        experiment = LambdaExperiment()
        return experiment.run(data_type)
