from .linear_vs_kernel_experiment import LinearVsKernelExperiment
from .lambda_experiment import LambdaExperiment
from .experiment_factory import ExperimentFactory

__all__ = [
    'LinearVsKernelExperiment',
    'LambdaExperiment',
    'ExperimentFactory',
    'run_experiment_detailed',
    'run_experiment_lambda'
]

run_experiment_detailed = ExperimentFactory.run_experiment_detailed
run_experiment_lambda = ExperimentFactory.run_experiment_lambda
