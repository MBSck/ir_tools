from pathlib import Path

from p2obt import create_obs


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    night_plan_dir = data_dir / "observations" / "P115"
    create_obs(night_plan_dir / "image.txt", user_name="MbS")
