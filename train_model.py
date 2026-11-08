import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. قراءة البيانات
data_path = '1-Table 1.csv'
try:
  df = pd.read_csv(data_path, sep=None, engine='python', decimal=',')
except Exception:
  df = pd.read_csv(data_path, decimal=',')

df.columns = [col.strip() for col in df.columns]

# 2. تحديد المتغيرات
features = ['SEMAINE', 'ET', 'Ni', 'Pi', 'Ki', 'HAUT', 'PDS A F', 'NBR F', 'SUR F']
col_map = {c.upper().replace(' ', '').replace('_', ''): c for c in df.columns}

matched_features = []
for f in features:
  norm = f.upper().replace(' ', '').replace('_', '')
  matched_features.append(col_map.get(norm, f))

target_col = None
for cand in ['PDS A S', 'PDS_A_S', 'PDS AS', 'PDSAS']:
  norm = cand.upper().replace(' ', '').replace('_', '')
  if norm in col_map:
    target_col = col_map[norm]
    break
if not target_col:
  target_col = 'PDS A S'

# تجهيز البيانات
X = df[matched_features].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df[target_col], errors='coerce')

clean_mask = ~(X.isna().any(axis=1) | y.isna())
X = X[clean_mask]
y = y[clean_mask]

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. التدريب عبر Gradient Boosting
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
)
model.fit(X_train, y_train)

# 5. التقييم
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'\n--- Résultats du Modèle (Calibré Basilic) ---')
print(f'R² Score: {r2:.4f}')
print(f'MAE: {mae:.4f} g')

# 6. الحفظ
joblib.dump(model, 'modele_basilic.pkl')
joblib.dump(matched_features, 'model_features.pkl')
print(
    "\nModèle sauvegardé avec succès dans 'modele_basilic.pkl' et"
    " 'model_features.pkl' !"
)