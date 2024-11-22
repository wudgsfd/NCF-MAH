import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, Dataset, TensorDataset
import heapq
from datetime import datetime


# 制作数据   用户打过分的为正样本， 用户没打分的为负样本， 负样本这里采用的采样的方式
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []

    # 获取所有物品的集合
    all_items = set()
    for items in train.values():
        all_items.update(items)
    num_items = len(all_items)

    for user, items in train.items():
        # 正例样本
        for item in items:
            user_input.append(user)
            item_input.append(item)
            labels.append(1)

        # 负例样本
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            # 由于我们没有物品到索引的直接映射，这里需要找到一个不在用户交互列表中的物品
            j = list(all_items - set(items))[j % (num_items - len(items))]
            user_input.append(user)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels


# 一些超参数设置
topK = 10
num_factors = 8
num_negatives = 4
batch_size = 32
lr = 0.001
num_heads = 4
head_dim = 8
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_items = train.shape[1]
    for (u, i) in train.keys():  # train.keys()是打分的用户和商品
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        # negative instance
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            # print(u, j)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels
class PureMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, reg=[0, 0]):
        super(PureMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.long()
        MF_Embedding_User = self.MF_Embedding_User(input[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(input[:, 1])

        predict = torch.mul(MF_Embedding_User, MF_Embedding_Item)

        linear = self.linear(predict)
        output = self.sigmoid(linear)
        output = output.squeeze(-1)
        return output


class PureMLP(nn.Module):
    def __init__(self, num_users, num_items, layers=[20, 64, 32, 16], regs=[0, 0]):
        super(PureMLP, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        # 全连接网络
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作
        # 先把输入转成long类型
        inputs = inputs.long()
        # MF的前向传播  用户和物品的embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第一列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        # 两个隐向量堆叠起来
        x = torch.cat([MF_Embedding_User, MF_Embedding_Item], dim=-1)
        # l全连接网络
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.linear(x)
        output = self.sigmoid(x)
        output = output.squeeze(-1)
        return output


class NeuralMF(nn.Module):
    def __init__(self, num_users, num_items, mf_embed_dim, layers):
        super(NeuralMF, self).__init__()
        # Embedding 层
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_embed_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_embed_dim)

        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        # 全连接层
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], mf_embed_dim)
        # 线性层
        self.linear2 = nn.Linear(2 * mf_embed_dim, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.long()
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        MF_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)

        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])
        # 将向量进行拼接后然后将其送入到全连接层
        x = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        MLP_vec = self.linear(x)

        # 将两个合并
        vector = torch.cat([MF_vec, MLP_vec], dim=-1)

        # 预测层 线性层
        linear = self.linear2(vector)
        output = self.sigmod(linear)
        output = output.squeeze(-1)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        # Get batch size
        batch_size = query.shape[0]

        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape Q, K, and V for multi-heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.embed_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = F.softmax(energy, dim=-1)

        # Apply attention to value
        out = torch.matmul(attention, V)

        # Reshape and concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)

        # Apply final linear transformation
        out = self.out(out)
        return out


class NeuralMFWithMultiHeadAttention(nn.Module):
    def __init__(self, num_users, num_items, mf_embed_dim, layers, num_heads):
        super(NeuralMFWithMultiHeadAttention, self).__init__()

        # Embedding layers
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_embed_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_embed_dim)

        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)

        # Multi-Head Attention
        self.multihead_attn = MultiHeadAttention(embed_dim=8, num_heads=num_heads)

        # Fully connected layers
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in zip(layers[:-1], layers[1:])])
        self.linear = nn.Linear(layers[-1], mf_embed_dim)

        # Linear layers
        self.linear2 = nn.Linear(2 * mf_embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.long()
        # Matrix Factorization Embeddings
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        MF_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)

        # MLP Embeddings
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])

        # Multi-Head Attention
        attention_output = self.multihead_attn(MLP_Embedding_User, MLP_Embedding_Item, MLP_Embedding_Item)
        # Concatenate attention output with MF embeddings
        x = torch.cat([attention_output.squeeze(1), MF_vec], dim=-1)

        # Fully connected layers
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        MLP_vec = self.linear(x)

        # Concatenate MF and MLP vectors
        vector = torch.cat([MF_vec, MLP_vec], dim=-1)

        # Prediction layer (linear layer)
        linear = self.linear2(vector)
        output = self.sigmoid(linear)
        output = output.squeeze(-1)
        return output


# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

# Precision
def getPrecision(ranklist, gtItem):
    if gtItem in ranklist:
        return 1 / len(ranklist)
    return 0

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)

    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')

    # 将 test_data 移动到正确的设备上
    test_data = torch.tensor(np.vstack([users, np.array(items)]).T).long().to(device)

    predictions = _model(test_data)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].item()  # 使用 .item() 获取 tensor 中的值
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=lambda k: map_item_score[k])
    precision = getPrecision(ranklist, gtItem)
    return precision

def evaluate_model(model, testRatings, testNegatives, K):
    global _model
    global _testNegatives
    global _testRatings
    global _K

    _model = model
    _testNegatives = testNegatives
    _testRatings = testRatings
    _K = K
    precisions = []
    bar = tqdm(range(len(_testRatings)), total=len(_testRatings))
    for idx in bar:
        precision = eval_one_rating(idx)
        precisions.append(precision)
    return precisions

# 定义全局路径
BASE_DIR = "answer"
os.makedirs(BASE_DIR, exist_ok=True)  # 创建输出目录（如果不存在）

# 主函数
def main(models, dl_train, testRatings, testNegatives, data_name, model_names):
    results = {}
    for model_name, model in models.items():
        print(f'Processing {data_name}-{model_name}')
        model = model.to(device)

        # 训练参数设置
        loss_func = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        # 初始评估
        precisions = evaluate_model(model, testRatings, testNegatives, topK)
        precision10 = np.array(precisions).mean()
        precisions = evaluate_model(model, testRatings, testNegatives, 20)
        precision20 = np.array(precisions).mean()
        print(f'Init: Precision@10=%.4f, Precision@20=%.4f' % (precision10, precision20))

        # 初始化最佳精度
        best_precision10, best_iter10 = precision10, -1
        best_precision20, best_iter20 = precision20, -1

        epochs = 20
        print('开始训练')

        precision10_list = []
        precision20_list = []

        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            steps = len(dl_train)
            bar = tqdm(enumerate(dl_train, 1), total=steps)
            for step, (features, labels) in bar:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(features)
                loss = loss_func(predictions, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                bar.set_description(f'Epoch: [{epoch + 1}/{epochs}]')
                bar.set_postfix(Train_loss=loss_sum / step)

            model.eval()
            precisions = evaluate_model(model, testRatings, testNegatives, topK)
            precision10 = np.array(precisions).mean()
            precision10_list.append(precision10)
            precisions = evaluate_model(model, testRatings, testNegatives, 20)
            precision20 = np.array(precisions).mean()
            precision20_list.append(precision20)

            if precision10 > best_precision10:
                best_precision10, best_iter10 = precision10, epoch
                torch.save(model.state_dict(), f'Pretrain/{model_name}/{data_name}_{model_name}.pkl')

            if precision20 > best_precision20:
                best_precision20, best_iter20 = precision20, epoch

            info = (epoch + 1, loss_sum / step, precision10, precision20)
            print(("EPOCH = %d, loss = %.3f, precision@10 = %.3f, precision@20 = %.3f") % info)

        # 保存到 CSV 文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_precision_to_csv(precision10_list, precision20_list, model_name, timestamp)

        results[model_name] = {'Precision@10': precision10_list, 'Precision@20': precision20_list}

    plot_all_models(results, data_name, timestamp)
    return results

def save_precision_to_csv(precision10_list, precision20_list, model_name, timestamp):
    df = pd.DataFrame({
        'Epoch': range(1, len(precision10_list) + 1),
        'Precision@10': precision10_list,
        'Precision@20': precision20_list
    })
    df.to_csv(f'{BASE_DIR}/{model_name}_{timestamp}_metrics.csv', index=False)

def plot_precision_at_k(results, data_name, timestamp, k):
    output_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(output_dir, exist_ok=True)  # 创建绘图目录（如果不存在）

    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(range(1, len(metrics[f'Precision@{k}']) + 1), metrics[f'Precision@{k}'], label=f'{model_name} Precision@{k}')

    plt.title(f'{data_name} Precision@{k} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, f'{data_name}_{timestamp}_AllModels_Precision@{k}.png'))
    plt.close()

def plot_all_models(results, data_name, timestamp):
    plot_precision_at_k(results, data_name, timestamp, 10)
    plot_precision_at_k(results, data_name, timestamp, 20)
if __name__ == '__main__':
    data_names = ["Appendix"]
    model_names = ["PureMF", "PureMLP", "NeuralMF", "NCF-MHA"]
    num_negatives = 4
    batch_size = 32
    topK = 10
    lr = 0.001

    # 检查是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    for data_name in data_names:
        # 加载数据
        train = np.load(f'ProcessedData/{data_name}/train.npy', allow_pickle=True).item()
        testRatings = np.load(f'ProcessedData/{data_name}/testRatings.npy').tolist()
        testNegatives = np.load(f'ProcessedData/{data_name}/testNegatives.npy').tolist()

        num_users, num_items = train.shape
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        train_x = np.vstack([user_input, item_input]).T
        labels = np.array(labels)

        # 构建成 Dataset 和 DataLoader
        train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
        dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        models = {}
        num_factors = 8
        num_heads = 4

        for model_name in model_names:
            if model_name == "NeuralMF":
                layers = [num_factors * 2, 64, 32, 16]
                models[model_name] = NeuralMF(num_users, num_items, num_factors, layers)
            elif model_name == "NCF-MHA":
                layers = [num_factors * 2, 64, 32, 16]
                models[model_name] = NeuralMFWithMultiHeadAttention(num_users, num_items, num_factors, layers, num_heads)
            elif model_name == "PureMLP":
                layers = [num_factors * 2, 64, 32, 16]
                models[model_name] = PureMLP(num_users, num_items, layers)
            elif model_name == "PureMF":
                mf_embed_dim = num_factors
                models[model_name] = PureMF(num_users, num_items, mf_embed_dim)
            else:
                print(f"Unknown model_name: {model_name}")
                continue

        # 移动模型到指定设备
        for model_name in models:
            models[model_name] = models[model_name].to(device)

        # 调用 main 函数并接收返回的 results
        results = main(models, dl_train, testRatings, testNegatives, data_name, model_names)

        # 绘图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_all_models(results, data_name, timestamp)