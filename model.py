import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import datetime
import numpy as np
import pandas as pd
from collections import Counter
import heapq
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
from torchsummary import summary
from tqdm import tqdm
import json

# 一些超参数设置
topK = 10
num_factors = 8
num_negatives = 4
batch_size = 64
lr = 0.001
num_heads = 4
head_dim = 8

# 检查是否有可用的CUDA设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 制作数据   用户打过分的为正样本， 用户没打分的为负样本， 负样本这里采用的采样的方式
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

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, reg=[0, 0]):
        super(GMF, self).__init__()
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


class MLP(nn.Module):

    def __init__(self, num_users, num_items, layers=[20, 64, 32, 16], regs=[0, 0]):
        super(MLP, self).__init__()
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

_model = None
_testRatings = None
_testNegatives = None
_K = None

# Hit
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

# NDCG
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return np.log(2) / np.log(i + 2)
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

    test_data = torch.tensor(np.vstack([users, np.array(items)]).T)
    predictions = _model(test_data.to(device))
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].cpu().data.numpy()
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=lambda k: map_item_score[k])  # heapq是堆排序算法， 取前K个
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return hr, ndcg


def evaluate_model(model, testRatings, testNegatives, K):
    global _model
    global _testRatings
    global _testNegatives
    global _K

    _model = model
    _testNegatives = testNegatives
    _testRatings = testRatings
    _K = K
    hits, ndcgs = [], []
    bar = tqdm(range(len(_testRatings)), total=len(_testRatings))
    for idx in bar:
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs


def main(net, dl_train, testRatings, testNegatives, data_name: str, model_name: str):
    # 训练参数设置
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    # 计算出初始的评估
    (hits, ndcgs) = evaluate_model(net, testRatings, testNegatives, topK)
    hr10, ndcg10 = np.array(hits).mean(), np.array(ndcgs).mean()
    (hits, ndcgs) = evaluate_model(net, testRatings, testNegatives, 20)
    hr20, ndcg20 = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR@10=%.4f, NDCG@10=%.4f, HR@20 = %.4f, NDCG@20=%.4f' % (hr10, ndcg10, hr20, ndcg20))
    # 模型训练
    best_hr10, best_ndcg10, best_iter10 = hr10, ndcg10, -1
    best_hr20, best_ndcg20, best_iter20 = hr20, ndcg20, -1

    epochs = 100
    log_step_freq = 10000
    print('开始训练')
    hr10_list = []
    hr20_list = []
    ndcg10_list = []
    ndcg20_list = []
    for epoch in range(epochs):
        # 训练阶段
        net.train()
        loss_sum = 0.0
        steps = len(dl_train)
        bar = tqdm(enumerate(dl_train, 1), total=steps)
        for step, (features, labels) in bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = net(features)
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            # 打印batch级别日志
            loss_sum += loss.item()
            bar.set_description(f'Epoch: [{epoch + 1}/{epochs}')
            bar.set_postfix(Train_loss=loss_sum / step)
        # 验证阶段
        net.eval()
        (hits, ndcgs) = evaluate_model(net, testRatings, testNegatives, topK)
        hr10, ndcg10 = np.array(hits).mean(), np.array(ndcgs).mean()
        hr10_list.append(hr10)
        ndcg10_list.append(ndcg10)
        (hits, ndcgs) = evaluate_model(net, testRatings, testNegatives, 20)
        hr20, ndcg20 = np.array(hits).mean(), np.array(ndcgs).mean()
        hr20_list.append(hr20)
        ndcg20_list.append(ndcg20)
        if hr10 > best_hr10:
            best_hr10, best_ndcg10, best_iter10 = hr10, ndcg10, epoch
            os.makedirs('Pretrain/' + model_name, exist_ok=True)
            torch.save(net.state_dict(), 'Pretrain/' + model_name + '/' + data_name + '_' + model_name + '.pkl')
        if hr20 > best_hr20:
            best_hr20, best_ndcg20, best_iter20 = hr20, ndcg20, epoch
        info = (epoch + 1, loss_sum / step, hr10, ndcg10, hr20, ndcg20)
        print(("EPOCH = %d, loss = %.3f, hr@10 = %.3f, ndcg@10 = %.3f, hr@20 = %.3f, ndcg@20 = %.3f") % info)

    # 创建一个DataFrame来保存结果
    results_df = pd.DataFrame({
        'hr10_list': hr10_list,
        'hr20_list': hr20_list,
        'ndcg10_list': ndcg10_list,
        'ndcg20_list': ndcg20_list
    })

    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)

    # 保存为CSV文件
    results_df.to_csv(f"figures/{model_name}.csv", index=False)

    print('Finished Training...')


def plt_figure(data1, data2, data3, data4, data5, data6, title="data"):
    x = range(1, len(data1) + 1, 2)
    plt.figure()
    plt.plot(data1, label="GMF", linewidth=2)
    plt.plot(data2, label="MLP", linewidth=2)
    plt.plot(data3, label="NeuMF", linewidth=2)
    plt.plot(data4, label="NCF-MAH", linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), borderaxespad=0.5, fontsize=8)
    plt.savefig(f"figures/{title}.png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_names = ["Appendix"]
    model_names = [ "NCF-MAH", "GMF", "MLP", "NeuMF"]
    lr = 0.001
    batch_size = 64
    topK = 10
    num_factors = 8
    num_heads = 4
    num_negatives = 4
    for data_name in data_names:
        train = np.load(f'ProcessedData/{data_name}/train.npy', allow_pickle=True).tolist()
        testRatings = np.load(f'ProcessedData/{data_name}/testRatings.npy').tolist()
        testNegatives = np.load(f'ProcessedData/{data_name}/testNegatives.npy').tolist()
        num_users, num_items = train.shape
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        train_x = np.vstack([user_input, item_input]).T
        labels = np.array(labels)
        # 构建成Dataset和DataLoader
        train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
        dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for model_name in model_names:
            print('Processing %s-%s' % (data_name, model_name))
            if model_name == "GMF":
                model = GMF(num_users, num_items, num_factors).to(device)
            elif model_name == "MLP":
                layers = [num_factors * 2, 64, 32, 16, 8]
                model = MLP(num_users, num_items, layers).to(device)
            elif model_name == "NeuMF":
                layers = [num_factors * 2, 64, 32, 16]
                model = NeuralMF(num_users, num_items, num_factors, layers).to(device)
            elif model_name == "NCF-MAH":
                layers = [num_factors * 2, 64, 32, 16]
                model = NeuralMFWithMultiHeadAttention(num_users, num_items, num_factors, layers, num_heads).to(device)
            main(model, dl_train, testRatings, testNegatives, data_name, model_name)
        # 读取CSV文件并绘制图形
        gmf = pd.read_csv("figures/GMF.csv").to_dict(orient='list')
        neu = pd.read_csv("figures/NeuMF.csv").to_dict(orient='list')
        mlp = pd.read_csv("figures/MLP.csv").to_dict(orient='list')
        my = pd.read_csv("figures/NCF-MAH.csv").to_dict(orient='list')
        plt_figure(gmf["hr10_list"], mlp["hr10_list"], neu["hr10_list"], my["hr10_list"], data_name + " hr@10")
        plt_figure(gmf["hr20_list"], mlp["hr20_list"], neu["hr20_list"], my["hr20_list"],  data_name + " hr@20")
        plt_figure(gmf["ndcg10_list"], mlp["ndcg10_list"], neu["ndcg10_list"], my["ndcg10_list"], data_name + " ndcg@10")
        plt_figure(gmf["ndcg20_list"], mlp["ndcg20_list"], neu["ndcg20_list"], my["ndcg20_list"],data_name + " ndcg@20")