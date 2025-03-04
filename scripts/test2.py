import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import argparse
import os
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

# 设置随机种子以确保结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data(input_file_1, input_file_2, input_file_3):
    """加载数据（适配多染色体）"""
    df1 = pd.read_csv(input_file_1, sep='\t', na_values='.', dtype=str)
    df1 = df1.dropna()
    df1['position'] = df1['position'].astype(int)
    df1['chr'] = df1['chr'].astype(str)

    df2 = pd.read_csv(input_file_2, sep='\t', header=None)
    df2.columns = ['sample_name', 'label']

    positions = pd.read_csv(input_file_3, sep='\t', header=None, names=['chr', 'position'])
    positions['position'] = positions['position'].astype(int)
    positions['chr'] = positions['chr'].astype(str)
    return df1, df2, positions


def filter_data(df1, positions):
    """过滤数据（多染色体版本）"""
    df1_index = df1.set_index(['chr', 'position']).index
    pos_index = positions.set_index(['chr', 'position']).index
    return df1[df1_index.isin(pos_index)]


def extract_features_labels(filtered_df, tagged_samples, chunk_size=10000):
    """提取特征和标签（分块处理）"""
    features, labels = [], []
    sorted_positions = filtered_df[['chr', 'position']].drop_duplicates().sort_values(['chr', 'position'])

    # 分块加载样本
    for i in range(0, len(tagged_samples), chunk_size):
        chunk = tagged_samples.iloc[i:i + chunk_size]
        for _, row in chunk.iterrows():
            sample_name = row['sample_name']
            mat_col = f'{sample_name}.mat'
            pat_col = f'{sample_name}.pat'

            if mat_col in filtered_df.columns and pat_col in filtered_df.columns:
                sorted_df = filtered_df.set_index(['chr', 'position']).loc[
                    sorted_positions.set_index(['chr', 'position']).index]
                mat = pd.to_numeric(sorted_df[mat_col], errors='coerce').fillna(0).values / 100.0
                pat = pd.to_numeric(sorted_df[pat_col], errors='coerce').fillna(0).values / 100.0

                # 创建第一个特征对 (mat, pat)
                feature1 = np.empty((len(mat), 2), dtype=np.float32)
                feature1[:, 0] = mat
                feature1[:, 1] = pat

                # 创建第二个特征对 (pat, mat)
                feature2 = np.empty((len(pat), 2), dtype=np.float32)
                feature2[:, 0] = pat
                feature2[:, 1] = mat

                features.extend([feature1, feature2])

                labels.extend(['mat', 'pat'])

    # 返回时强制转换为连续数组
    return np.ascontiguousarray(features), np.array(labels), sorted_positions


class LSTMClassifier(nn.Module):
    """LSTM模型（支持混合精度训练）"""

    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if config['num_layers'] > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, config['num_classes'])
        )

    def forward(self, x):
        x = x.contiguous()
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.classifier(context_vector)


def train_model(X, y, sorted_positions, model_filename, chunk_size=10000):
    """训练模型（优化显存版本）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    np.save("train", np.array(X))
    np.save("label", np.array(y_encoded))

    if len(label_encoder.classes_) != 2:
        raise ValueError("仅支持二分类")

    # 模型配置
    model_config = {
        'input_size': 2,
        'seq_len': X.shape[1],
        'num_classes': len(label_encoder.classes_),
        'num_layers': 2,
        'seed': seed
    }
    model = LSTMClassifier(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scaler = torch.amp.GradScaler()  # 混合精度

    # 分块训练
    X = np.ascontiguousarray(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).contiguous()
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    num_chunks = max(1, len(X) // chunk_size)
    X_chunks = torch.chunk(X_tensor, num_chunks)
    y_chunks = torch.chunk(y_tensor, num_chunks)

    best_val_acc = 0.0
    for epoch in range(20):
        model.train()
        train_loss = 0.0

        # 分块训练循环
        for X_chunk, y_chunk in zip(X_chunks, y_chunks):
            X_train, X_test, y_train, y_test = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=seed)
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                inputs, labels = inputs.to(device).contiguous(), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 混合精度
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * inputs.size(0)

        # 验证
        model.eval()
        val_loss, all_preds, all_probs, all_labels = [], [], [], []
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).contiguous(), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("valid_loss:{}".format(np.mean(val_loss)))
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        if val_acc > best_val_acc and args.save_model == 'yes':
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder_classes': label_encoder.classes_,
                'model_config': model_config,
                'sorted_positions': sorted_positions.values.tolist()
            }, model_filename)
        torch.cuda.empty_cache()  # 显存清理



def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型配置
    model_config = {
        'input_size': 2,
        'seq_len': X.shape[1],
        'num_classes': len(label_encoder.classes_),
        'num_layers': 2,
        'seed': seed
    }
    model = LSTMClassifier(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scaler = torch.amp.GradScaler()  # 混合精度

    # 分块训练
    X = np.ascontiguousarray(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).contiguous()
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    num_chunks = max(1, len(X) // chunk_size)
    X_chunks = torch.chunk(X_tensor, num_chunks)
    y_chunks = torch.chunk(y_tensor, num_chunks)

    best_val_acc = 0.0
    for epoch in range(20):
        model.train()
        train_loss = 0.0

        # 分块训练循环
        for X_chunk, y_chunk in zip(X_chunks, y_chunks):
            X_train, X_test, y_train, y_test = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=seed)
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                inputs, labels = inputs.to(device).contiguous(), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 混合精度
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * inputs.size(0)

        # 验证
        model.eval()
        val_loss, all_preds, all_probs, all_labels = [], [], [], []
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).contiguous(), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("valid_loss:{}".format(np.mean(val_loss)))
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        if val_acc > best_val_acc and args.save_model == 'yes':
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder_classes': label_encoder.classes_,
                'model_config': model_config,
                'sorted_positions': sorted_positions.values.tolist()
            }, model_filename)
        torch.cuda.empty_cache()  # 显存清理

def predict_haplotypes(model, label_encoder, filtered_df, untagged_samples, sorted_positions, device):
    """预测单倍型来源（全基因组版本）"""
    model.eval()
    predictions = []

    # 确保使用相同的位点顺序
    filtered_df = filtered_df.set_index(['chr', 'position']).loc[
        sorted_positions.set_index(['chr', 'position']).index].reset_index()

    for _, row in untagged_samples.iterrows():
        sample_name = row['sample_name']
        mat_col = f'{sample_name}.mat'
        pat_col = f'{sample_name}.pat'

        if mat_col not in filtered_df.columns or pat_col not in filtered_df.columns:
            print(f"Error: Missing columns for sample {sample_name}.")
            continue

        mat_values = pd.to_numeric(filtered_df[mat_col], errors='coerce').fillna(0).values
        pat_values = pd.to_numeric(filtered_df[pat_col], errors='coerce').fillna(0).values
        mat_values = mat_values / 100.0
        pat_values = pat_values / 100.0

        # 创建特征对（二维输入）
        feature_pairs = [
            np.column_stack((mat_values, pat_values)).reshape(1, -1, 2),  # (1, seq_len, 2)
            np.column_stack((pat_values, mat_values)).reshape(1, -1, 2)
        ]

        for i, feat in enumerate(feature_pairs):
            with torch.no_grad():
                inputs = torch.tensor(feat, dtype=torch.float32).to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs).item()
                origin = label_encoder.inverse_transform([pred_idx])[0]

            haplotype = f'{sample_name}.mat' if i == 0 else f'{sample_name}.pat'
            confidence = {label: round(probs[0][i].item(), 4) for i, label in enumerate(label_encoder.classes_)}
            predictions.append([haplotype, origin, confidence])

    return pd.DataFrame(predictions, columns=['haplotype_name', 'inferred_origin', 'confidence'])


def main(args):
    """主函数（修复 model 未定义问题）"""
    df1, df2, positions = load_data(args.input_file_1, args.input_file_2, args.input_file_3)
    filtered_df = filter_data(df1, positions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确保 device 定义在外部

    # 初始化变量
    model = None
    label_encoder = None

    # 严格筛选有效tagged样本
    valid_samples = []
    for _, row in df2.iterrows():
        sample = row['sample_name']
        if f'{sample}.mat' in filtered_df.columns and f'{sample}.pat' in filtered_df.columns:
            valid_samples.append(row)
    tagged_samples = pd.DataFrame(valid_samples).query("label == 'tagged'")

    # 提取特征时返回排序后的位点信息
    X, y, sorted_positions = extract_features_labels(filtered_df, tagged_samples, args.chunk_size)

    if args.train_model:
        train_model(X, y, sorted_positions, args.model_filename, args.chunk_size)
    elif args.load_model:
        # ==== 加载模型部分 ====
        try:
            checkpoint = torch.load(args.model_filename, map_location=device, weights_only=False)
            # 恢复标签编码器
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(checkpoint['label_encoder_classes'])
            # 恢复模型
            model = LSTMClassifier(checkpoint['model_config']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return  # 直接退出避免后续错误

    # ==== 确保 model 已定义 ====
    if model is not None:
        untagged_samples = df2[df2['label'] == 'untagged']
        predictions = predict_haplotypes(model, label_encoder, filtered_df,
                                         untagged_samples, sorted_positions, device)

        base, ext = os.path.splitext(args.model_filename)
        results_filename = f'{base}_methylation_corrected_haplotype.tsv'
        predictions.to_csv(results_filename, sep="\t", index=False)
    else:
        print("警告：未初始化模型，无法进行预测")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_1', required=True)
    parser.add_argument('--input_file_2', required=True)
    parser.add_argument('--input_file_3', required=True)
    parser.add_argument('--model_filename', required=True)
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--save_model', choices=['yes', 'no'], default='yes')
    parser.add_argument('--chunk_size', type=int, default=10000)  # 新增分块参数
    args = parser.parse_args()
    main(args)