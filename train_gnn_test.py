import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, RDKFingerprint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# === 1. Создание директории ===
os.makedirs("model_gnn", exist_ok=True)

# === 2. Загрузка данных ===
df = pd.read_csv("data/LogP_LogS.csv")
df = df.drop(columns=["Compound ID", "InChIKey"], errors="ignore")

# === 3. Генерация дескрипторов ===
def compute_extended_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.MolLogP(mol),
        Descriptors.ExactMolWt(mol),
        QED.qed(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumHeterocycles(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Ipc(mol),
        Descriptors.Kappa1(mol),
        Descriptors.LabuteASA(mol)
    ]

descriptor_columns = [
    "MolWt", "TPSA", "HDonors", "HAcceptors", "RotatableBonds",
    "RingCount", "HeavyAtoms", "LogP", "ExactMolWt", "QED",
    "FractionCSP3", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "NumHeterocycles", "BalabanJ",
    "HallKierAlpha", "Ipc", "Kappa1", "LabuteASA"
]

tqdm.pandas(desc="Генерация дескрипторов")
df[descriptor_columns] = df["SMILES"].progress_apply(compute_extended_descriptors).apply(pd.Series)
df = df.dropna()

# === 4. Генерация фингерпринтов ===
def smiles_to_morgan_fingerprint(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    return np.array(fingerprint)

def smiles_to_rdk_fingerprint(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = RDKFingerprint(mol, fpSize=nBits)
    return np.array(fingerprint)

df['morgan_fingerprint'] = df['SMILES'].progress_apply(smiles_to_morgan_fingerprint)
df['rdk_fingerprint'] = df['SMILES'].progress_apply(smiles_to_rdk_fingerprint)

morgan_fingerprints = np.stack(df['morgan_fingerprint'].dropna().values)
rdk_fingerprints = np.stack(df['rdk_fingerprint'].dropna().values)

# === 5. Подготовка данных ===
X = np.hstack([df[descriptor_columns].values, morgan_fingerprints, rdk_fingerprints])
y = df["logS"]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === 6. Модели ===
xgb_model = XGBRegressor(
    n_estimators=1000, max_depth=10, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=1000, max_depth=20,
    min_samples_split=2, random_state=42,
    n_jobs=-1
)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class TorchWrapper:
    def __init__(self, model, optimizer, criterion, epochs=50):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in tqdm(range(self.epochs), desc="Training MLP"):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = self.criterion(pred, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions

mlp_model = MLP(input_size=X_train.shape[1])
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
mlp_wrapper = TorchWrapper(mlp_model, optimizer, criterion, epochs=50)

# === 7. Обучение моделей ===
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
mlp_wrapper.fit(X_train, y_train)

# === 8. Предсказания ===
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_mlp = mlp_wrapper.predict(X_test)

# === 9. Ансамбль (простое среднее) ===
y_pred_ensemble = (y_pred_xgb + y_pred_rf + y_pred_mlp) / 3

# === 10. Оценка ===
print(f"[XGBoost] MAE: {mean_absolute_error(y_test, y_pred_xgb):.4f}, R²: {r2_score(y_test, y_pred_xgb):.4f}")
print(f"[Random Forest] MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}, R²: {r2_score(y_test, y_pred_rf):.4f}")
print(f"[MLP] MAE: {mean_absolute_error(y_test, y_pred_mlp):.4f}, R²: {r2_score(y_test, y_pred_mlp):.4f}")
print(f"[Ensemble] MAE: {mean_absolute_error(y_test, y_pred_ensemble):.4f}, R²: {r2_score(y_test, y_pred_ensemble):.4f}")

# === 11. Сохранение моделей ===
joblib.dump(xgb_model, "model_gnn/xgb_model.pkl")
joblib.dump(rf_model, "model_gnn/rf_model.pkl")
torch.save(mlp_wrapper.model.state_dict(), "model_gnn/mlp_model_weights.pth")
joblib.dump(scaler_X, "model_gnn/scaler_X.pkl")
joblib.dump(scaler_y, "model_gnn/scaler_y.pkl")
joblib.dump(descriptor_columns, "model_gnn/descriptor_columns.pkl")

# === 12. Важность признаков для XGBoost ===
importances = xgb_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(descriptor_columns)), importances[:len(descriptor_columns)])
plt.xticks(range(len(descriptor_columns)), descriptor_columns, rotation=90)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()
