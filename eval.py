import os
import sys
import subprocess
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project
    eval_type = config.dataset.data_type

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample(model_base):
        cprint(f"This is sampling.", color = "green")
        if model_base == "dream":
            subprocess.run(
                f'python dream_sample.py '
                f'config=../configs/{project_name}.yaml ',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "llada":
            subprocess.run(
                f'python llada_sample.py '
                f'config=../configs/{project_name}.yaml ',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "sdar":
            subprocess.run(
                f'python sdar_sample.py '
                f'config=../configs/{project_name}.yaml ',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "sdar_ar":
            subprocess.run(
                f'python sdar_sample_ar.py '
                f'config=../configs/{project_name}.yaml ',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "sdar_arbd":
            subprocess.run(
                f'python sdar_sample_arbd.py '
                f'config=../configs/{project_name}.yaml ',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "trado":
            subprocess.run(
                f'python trado_sample.py '
                f'config=../configs/{project_name}.yaml ',
                shell=True,
                cwd='sample',
                check=True,
            )
    
    def reward():
        cprint(f"This is the rewarding.", color = "green")
        subprocess.run(
            f'python reward.py '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    def execute():
        cprint(f"This is the execution.", color = "green")
        subprocess.run(
            f'python execute.py '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    
    sample(config.model_base)
    if eval_type == "code":
        execute()
    
    reward()




