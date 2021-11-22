import sys
import yaml
from yaml.error import YAMLError
from src.etl import load_data
from src.mcmc_solver import run_mh_mcmc

def main(targets):

    if 'test' in targets:
        with open("config/test-config.yaml", "r") as f:
            try:
                test_config = yaml.safe_load(f)
            except YAMLError as exc:
                print(exc)
        X,y = load_data(**test_config)
        solver_params = test_config['solver_params']
        run_mh_mcmc(X,y, **solver_params)


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)