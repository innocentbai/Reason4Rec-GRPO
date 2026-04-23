import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

dataset = 'Music_data'
# 读取数据
true_df = pd.read_pickle(f'./Data/{dataset}/raw_data/test.pkl')
pred_df = pd.read_pickle(f'./Data/{dataset}/rating_for_test_by_Predictor__4GPU_GRPO_final.pkl')
pred_df['pred'] = pred_df['pred'] 
# 重命名字段统一处理
true_df = true_df.rename(columns={'ratings': 'true_rating'})
pred_df = pred_df.rename(columns={'pred': 'pred_rating'})

# 合并两个 DataFrame，按 user_id 和 item_id 对齐
merged_df = pd.merge(true_df, pred_df, on=['user_id', 'item_id'], how='inner')

# 过滤缺失值（可选但推荐）
merged_df = merged_df.dropna(subset=['true_rating', 'pred_rating'])

# 计算 MAE 和 RMSE
mae = mean_absolute_error(merged_df['true_rating'], merged_df['pred_rating'])
rmse = np.sqrt(mean_squared_error(merged_df['true_rating'], merged_df['pred_rating']))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
