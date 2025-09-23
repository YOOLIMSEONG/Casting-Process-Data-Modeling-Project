import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df_num):
    df_num = df_num.dropna()
    X = df_num.copy()
    vif_df = pd.DataFrame()
    vif_df['feature'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df.sort_values('VIF', ascending=False)

df = pd.read_csv("./data/raw/train.csv")
num_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[num_cols].copy()

vif_res = compute_vif(df_numeric.fillna(df_numeric.mean()))
vif_res