import pandas as pd
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "train_v1.csv"

# 데이터 로드
train_df = pd.read_csv(DATA_FILE)