from pathlib import Path

from p2obt import create_obs, create_ob

if __name__ == "__main__":
    # data_dir = Path().home() / "Data"
    # night_plan_dir = data_dir / "observations" / "P115"
    # create_obs(night_plan_dir / "image.txt", user_name="MbS")
    create_ob(
        "M8E-IR",
        "sci",
        "extended",
        mode="st",
        output_dir=Path().cwd(),
    )
