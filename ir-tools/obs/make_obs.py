from pathlib import Path

from p2obp import OPTIONS, create_obs, create_ob


OPTIONS.dit.gra4mat.ats.high = 1.3

if __name__ == "__main__":
    night_plan_dir = Path("/Users/scheuck/Data/observations/P113")
    output_dir = Path("/Users/scheuck/Data/observations/obs")
    # create_obs(night_plan_dir / "obs_plan_err.txt", user_name="MbS", observational_mode="sm")
    create_ob("HP Cha", "cal", "uts", "st",
              container_id=3795161, user_name="MbS")
