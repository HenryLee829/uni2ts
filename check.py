import hydra
from omegaconf import DictConfig
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HYDRA_RUN_DIR"] = "."
os.environ["HYDRA_JOB_CHDIR"] = "True"


@hydra.main(version_base="1.3", config_path="cli/conf/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
