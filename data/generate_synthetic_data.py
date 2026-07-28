"""
generate_synthetic_data.py
---------------------------
Generates a clinically-plausible SYNTHETIC dataset that mimics the structure
of the public "PCOS Dataset" found on Kaggle (Prasoon Kottarathil's
PCOS_data_without_infertility.csv), restricted to the fields this app
actually collects from the user.

WHY THIS EXISTS:
The real Kaggle CSV cannot be downloaded automatically inside this build
environment (no internet access to Kaggle at build time), so this script
lets the whole pipeline (train.py -> pcos_model.pkl -> Flask app) run
end-to-end out of the box.

TO USE REAL DATA:
Download "PCOS_data_without_infertility.csv" from Kaggle, place it at
data/pcos_dataset.csv, and re-run:
    python app/model/train.py --data data/pcos_dataset.csv
train.py will use the real file automatically if it is present and valid.
This synthetic generator is only a fallback, and pcos_model.pkl trained on
it must NEVER be described to end users as clinically validated.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3000


def generate(n=N):
    age = RNG.integers(15, 45, n)
    height_cm = RNG.normal(160, 7, n).clip(140, 190)
    weight_kg = RNG.normal(65, 15, n).clip(35, 130)
    bmi = weight_kg / ((height_cm / 100) ** 2)

    systolic_bp = RNG.normal(118, 12, n).clip(90, 170).round()
    diastolic_bp = RNG.normal(78, 9, n).clip(55, 110).round()
    pulse = RNG.normal(78, 10, n).clip(55, 120).round()

    # Latent PCOS propensity drives correlated symptoms (keeps dataset realistic)
    latent = RNG.normal(0, 1, n)
    latent += (bmi - 23) * 0.05

    pcod_ultrasound = (RNG.normal(latent, 1) > 1.1).astype(int)  # Yes/No/NotDone collapsed later
    hair_loss = (RNG.normal(latent, 1) > 0.6).astype(int)
    hair_growth = (RNG.normal(latent, 1) > 0.7).astype(int)
    irregular_cycle = (RNG.normal(latent, 1) > 0.4).astype(int)

    period_gap_days = np.where(
        irregular_cycle == 1,
        RNG.normal(50, 20, n).clip(28, 120),
        RNG.normal(29, 3, n).clip(21, 35),
    ).round()

    weight_fluctuation = (RNG.normal(latent, 1) > 0.5).astype(int)
    junk_food = RNG.choice(["Never", "Sometimes", "Often", "Daily"], n, p=[0.15, 0.4, 0.3, 0.15])
    workout = (RNG.normal(-latent, 1) > 0.2).astype(int)  # more workout -> lower latent risk shown
    marital_status = RNG.choice(["Married", "Unmarried"], n, p=[0.45, 0.55])
    stress_score = RNG.normal(5 + latent, 2, n).clip(1, 10).round()
    skin_issue = (RNG.normal(latent, 1) > 0.5).astype(int)

    fsh = RNG.normal(6, 2, n).clip(1, 20)
    lh = RNG.normal(6 + latent * 2, 2.5, n).clip(1, 30)
    lh_fsh_ratio = lh / fsh
    amh = RNG.normal(3 + latent * 1.5, 2, n).clip(0.2, 15)
    tsh = RNG.normal(2.5, 1.2, n).clip(0.3, 10)
    prolactin = RNG.normal(15, 6, n).clip(2, 60)
    testosterone = RNG.normal(40 + latent * 8, 15, n).clip(10, 120)
    follicle_count_l = RNG.poisson(8 + latent.clip(-2, 4) * 2, n).clip(0, 30)
    follicle_count_r = RNG.poisson(8 + latent.clip(-2, 4) * 2, n).clip(0, 30)
    ovary_vol_l = RNG.normal(6 + latent, 2, n).clip(2, 20)
    ovary_vol_r = RNG.normal(6 + latent, 2, n).clip(2, 20)

    target_score = (
        1.4 * pcod_ultrasound
        + 0.9 * irregular_cycle
        + 0.7 * hair_growth
        + 0.5 * hair_loss
        + 0.4 * (bmi > 25)
        + 0.5 * weight_fluctuation
        + 0.5 * (lh_fsh_ratio > 2)
        + 0.4 * (follicle_count_l + follicle_count_r > 24)
        + 0.3 * skin_issue
        + RNG.normal(0, 0.6, n)
    )
    pcos_label = (target_score > target_score.mean() + 0.15 * target_score.std()).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "height_cm": height_cm.round(1),
            "weight_kg": weight_kg.round(1),
            "bmi": bmi.round(2),
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "pulse_rate": pulse,
            "pcod_ultrasound": np.where(pcod_ultrasound == 1, "Yes", "No"),
            "hair_loss": np.where(hair_loss == 1, "Yes", "No"),
            "hair_growth_excessive": np.where(hair_growth == 1, "Yes", "No"),
            "period_cycle": np.where(irregular_cycle == 1, "Irregular", "Regular"),
            "period_gap_days": period_gap_days,
            "weight_fluctuation": np.where(weight_fluctuation == 1, "Yes", "No"),
            "junk_food_frequency": junk_food,
            "workout_routine": np.where(workout == 1, "Yes", "No"),
            "marital_status": marital_status,
            "stress_score": stress_score,
            "skin_issues": np.where(skin_issue == 1, "Acne", "None"),
            "fsh": fsh.round(2),
            "lh": lh.round(2),
            "lh_fsh_ratio": lh_fsh_ratio.round(2),
            "amh": amh.round(2),
            "tsh": tsh.round(2),
            "prolactin": prolactin.round(2),
            "testosterone": testosterone.round(2),
            "follicle_count_l": follicle_count_l,
            "follicle_count_r": follicle_count_r,
            "ovary_volume_l": ovary_vol_l.round(2),
            "ovary_volume_r": ovary_vol_r.round(2),
            "pcos_diagnosis": pcos_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/pcos_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to {out_path} ({len(df)} rows)")
    print(df["pcos_diagnosis"].value_counts(normalize=True))
