import joblib
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, RDKFingerprint

# === 1. Загрузка моделей и скейлеров ===
xgb_model = joblib.load("model_gnn/xgb_model.pkl")
rf_model = joblib.load("model_gnn/rf_model.pkl")

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

descriptor_columns = joblib.load("model_gnn/descriptor_columns.pkl")
scaler_X = joblib.load("model_gnn/scaler_X.pkl")
scaler_y = joblib.load("model_gnn/scaler_y.pkl")

mlp_model = MLP(input_size=len(descriptor_columns) + 2048)  # 1024 + 1024 бит фингерпринтов
mlp_model.load_state_dict(torch.load("model_gnn/mlp_model_weights.pth", map_location=torch.device('cpu')))
mlp_model.eval()

# === 2. Функции для преобразования SMILES ===
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

# === 3. Ввод от пользователя ===
while True:
    smiles_input = input("\nВведите SMILES (или 'exit' для выхода): ").strip()
    if smiles_input.lower() == 'exit':
        break

    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        print("Невалидный SMILES. Попробуйте снова.")
        continue

    descriptors = compute_extended_descriptors(smiles_input)
    morgan_fp = smiles_to_morgan_fingerprint(smiles_input)
    rdk_fp = smiles_to_rdk_fingerprint(smiles_input)

    if descriptors is None or morgan_fp is None or rdk_fp is None:
        print("Ошибка обработки SMILES.")
        continue

    full_features = np.hstack([descriptors, morgan_fp, rdk_fp])
    full_features_scaled = scaler_X.transform(full_features.reshape(1, -1))

    # Предсказания
    pred_xgb = xgb_model.predict(full_features_scaled)[0]
    pred_rf = rf_model.predict(full_features_scaled)[0]

    with torch.no_grad():
        input_tensor = torch.tensor(full_features_scaled, dtype=torch.float32)
        pred_mlp = mlp_model(input_tensor).numpy().flatten()[0]

    # Ансамбль
    pred_scaled = (pred_xgb + pred_rf + pred_mlp) / 3

    # Обратная нормализация
    pred_logS = scaler_y.inverse_transform(np.array(pred_scaled).reshape(1, -1))[0][0]
    pred_S = 10 ** pred_logS

    print(f"\nПредсказанный logS: {pred_logS:.4f}")
    print(f"Предсказанная растворимость S: {pred_S:.4e} (моль/л)")
