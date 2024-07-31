import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import threading
import queue
import time
from sklearn.cluster import KMeans
from scipy.stats import entropy
import numpy as np
import torch.nn.functional as F
import random
import os
import logging

os.environ["OMP_NUM_THREADS"] = "16"
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename='fmnist_TrimmedMean_baseline_test.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class CircularQueue_for_client:
    def __init__(self, max_length):
        self.queue = [None] * max_length
        self.max_length = max_length
        self.head = 0
        self.tail = 0
        self.size = 0

    def put(self, item):
        if self.size == self.max_length:
            # 队列满了，覆盖最旧的元素并移动头指针
            self.head = (self.head + 1) % self.max_length
        else:
            self.size += 1

        self.queue[self.tail] = item
        self.tail = (self.tail + 1) % self.max_length

    def get(self):
        if self.size == 0:
            raise IndexError("get from an empty queue")

        item = self.queue[self.head]
        self.queue[self.head] = None
        self.head = (self.head + 1) % self.max_length
        self.size -= 1
        return item

    def current_items(self):
        if self.size == 0:
            return []
        elif self.tail > self.head:
            return self.queue[self.head:self.tail]
        else:
            return self.queue[self.head:] + self.queue[:self.tail]

    def getall(self):
        # 使用 current_items 方法获取所有非空元素
        return self.current_items()

    def __len__(self):
        return self.size




# 用于将FMNIST数据集分为non-IID数据集
def split_fmnist_non_iid(dataset, num_clients):
    # 将FMNIST数据集按类别分组
    class_indices = [[] for _ in range(10)]
    for idx, (data, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 创建三个客户端的索引列表
    client_indices = [[] for _ in range(num_clients)]

    # 在每个客户端中分配类别不均匀的样本
    for class_idx in range(10):
        random_permutation = np.random.permutation(class_indices[class_idx])
        num_samples_per_client = len(random_permutation) // num_clients
        for client_idx in range(num_clients):
            start_idx = client_idx * num_samples_per_client
            end_idx = (client_idx + 1) * num_samples_per_client if client_idx < num_clients - 1 else len(random_permutation)
            client_indices[client_idx].extend(random_permutation[start_idx:end_idx])

    # 创建DataLoader对象以加载分配给每个客户端的样本
    dataloaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=True)
        dataloaders.append(dataloader)

    # all_indices = list(range(len(dataset)))
    # np.random.shuffle(all_indices)
    #
    # num_samples_per_client = len(all_indices) // num_clients
    # client_indices = []
    #
    # for client_idx in range(num_clients):
    #     start_idx = client_idx * num_samples_per_client
    #     if client_idx == num_clients - 1:
    #         end_idx = len(all_indices)
    #     else:
    #         end_idx = (client_idx + 1) * num_samples_per_client
    #     client_indices.append(all_indices[start_idx:end_idx])
    #
    # dataloaders = []
    # for indices in client_indices:
    #     subset = Subset(dataset, indices)
    #     dataloader = DataLoader(subset, batch_size=64, shuffle=True)
    #     dataloaders.append(dataloader)

    return dataloaders, client_indices


# 定义一个简单的MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

model = MLP().to(device)
server_model = MLP().to(device)

# 定义计算权重熵的损失函数
def entropy_loss(weights):
    entropy = torch.sum(-F.softmax(weights, dim=0) * F.log_softmax(weights, dim=0))
    return entropy


# 定义一个简单的服务器类，本例子没有使用server
class Client:
    def __init__(self, model, ID, peers=[]):
        self.model = model.to(device)
        self.ID = ID
        self.peers = peers

    def send_weights(self):
        return self.model.state_dict()

    def receive_weights(self, weights_list):
        new_state_dict = {}
        for key in weights_list[0].keys():
            new_state_dict[key] = torch.mean(torch.stack([weights[key] for weights in weights_list]), 0)
        self.model.load_state_dict(new_state_dict)

# 实现伪码第6行
def get_weight_x_data(client,batched_data):
    raw_data=batched_data.view(-1, 784)
    for name, param in client.model.named_parameters():
        if 'bias' not in name:
            raw_data=raw_data @ param.t()
        # print(f'Parameter: {name}, Gradient: {param.grad}')

    mean = raw_data.mean(dim=0, keepdim=True)
    std = raw_data.std(dim=0, keepdim=True)

    # 进行归一化
    raw_normalized = (raw_data - mean) / (std + 1e-10)  # 加上一个小常数以避免除以零

    return raw_normalized

# 实现伪码第8行
def cluster_data(feature_data,raw_data):
    kmeans = KMeans(n_clusters=cluster_sum)
    assignments = kmeans.fit_predict(feature_data.detach().numpy())
    # distances = torch.cdist(raw_data, torch.tensor(kmeans.cluster_centers_))
    return assignments

# 实现伪码第10行
def get_cross_entropy(feature_data,assignments):
    ce = F.cross_entropy(feature_data, torch.tensor(assignments).long())
    return ce


def get_gram_matrix(matrix: torch.Tensor):
    # using 2-Polynomial as the kernel function to calculate the gram matrix
    matrix = matrix/matrix.shape[0]
    gram_matrix = torch.mm(matrix, matrix.t())
    gram_matrix = torch.pow(gram_matrix,2)
    # gram_matrix = torch.abs(gram_matrix)
    # normalization
    # gram_matrix = gram_matrix/gram_matrix.numel()
    return gram_matrix

def von_neumann_entropy(matrix: torch.Tensor):
    # s(p) = - tr(p*log2(p))
    epsilon = 1e-10
    gram_matrix = get_gram_matrix(matrix)+epsilon
    # new_p = torch.mm(gram_matrix, torch.log2(gram_matrix))
    new_p = gram_matrix * torch.log2(gram_matrix)
    sp = torch.trace(new_p)
    return -sp

# 实现伪码第9行
def get_mutual_information(feature_data,assignments):
    assignments=torch.tensor(assignments)
    assignments=torch.unsqueeze(assignments,1)
    return mutual_information(assignments.float(),feature_data.t())


def mutual_information(matrixA: torch.Tensor, matrixB: torch.Tensor):
    # I(A,B) = H(A) + H(B) - H(A, B)
    joint_matrix = torch.mm(matrixA.t()/10, matrixB.t())
    ha=von_neumann_entropy(matrixA)
    hb = von_neumann_entropy(matrixB)
    # joint_matrix = get_gram_matrix(joint_matrix)
    # print(gram_joint.shape)
    hab = von_neumann_entropy(joint_matrix)
    # print(hab)
    im = ha + hb - hab
    if torch.isnan(im):
        return torch.tensor(0)
    return im

 #    ma = weights[i-1]
 #            mb =  weights[i]
 #            ha = von_neumann_entropy(ma)
 #            hb = von_neumann_entropy(mb)
 #            im += mutual_information(ma, mb, ha, hb)
# 实现伪码第12，13行
def get_average_4_mi_or_ce(data):
    # 将列表中的张量拼接成一个张量
    concatenated_tensor = torch.stack(data)
    # 计算张量中所有元素的平均值
    mean_value = torch.mean(concatenated_tensor.float())
    return mean_value


# 实现伪码第15行
def get_loss_avermi_minus_averce(average_mi,average_ce):
    return a_mi*average_mi-b_ce*average_ce
# 实现伪码第19，20行
def get_diff_mi_or_ce_from_average(average,records):
    diff=[item-average for item in records]
    return diff

# 实现伪码第24行
def get_sample_weight(mi_record,ce_record):

    # 计算所有值的对数
    mi_record_safe = torch.clamp(torch.stack(mi_record), min=1e-10)

    mi_log_values = torch.log(mi_record_safe)
    # 计算所有对数值的总和
    mi_log_sum = torch.sum(mi_log_values)
    # 计算每个对数值相对于总和的比例（即权重）
    mi_weights = mi_log_values / mi_log_sum

    ce_record_safe = torch.clamp(torch.stack(ce_record), min=1e-10)
    ce_log_values = torch.log(ce_record_safe)
    # 计算所有对数值的总和
    ce_log_sum = torch.sum(ce_log_values)
    # 计算每个对数值相对于总和的比例（即权重）
    ce_weights = ce_log_values / ce_log_sum


    reciprocal_weights1 = 1.0 / mi_weights
    reciprocal_weights2 = 1.0 / ce_weights

    # 对倒数进行相加
    sum_reciprocal_weights = reciprocal_weights1 + reciprocal_weights2

    # 对相加的结果取倒数，即为调和权重
    harmonic_weights = 1.0 / sum_reciprocal_weights

    return harmonic_weights

# 实现伪码第21行
def get_threshold_mi_or_ce(data_tensor, threshold=3):
    # 计算均值和标准差
    mean = torch.mean(torch.stack(data_tensor))
    std = torch.std(torch.stack(data_tensor))
    # 标准化数据
    # standardized_data = (data_tensor - mean) / std
    # 设置阈值
    lower_threshold = mean - threshold * std
    upper_threshold = mean + threshold * std
    return lower_threshold,upper_threshold

# 实现伪码第21，22行
def detect_outliers(data_tensor, lower_threshold,upper_threshold):
    # 标记异常值
    data_tensor=torch.stack(data_tensor)
    outliers = torch.where((data_tensor < lower_threshold) | (data_tensor > upper_threshold))[0]

    return outliers


# 实现伪码第23行
def get_normal_samples(full_length,mi_anomaly,ce_anomaly):

   mi=[mi.numpy()  for mi in mi_anomaly]
   ce=[ce.numpy() for ce in ce_anomaly]
   mi_or_ce=set(mi).union(set(ce))
   full_set = np.arange(0, full_length)
   normal_samples=set(full_set)-mi_or_ce
   return normal_samples

# 实现伪码第24行
def get_sample_weight_x_model_weight(sample_weight,normal_samples_id,client):

    agg_weights_temp=[]
    normal_samples_number= len(normal_samples_id)
    for name, param in client.model.named_parameters():
        if 'bias' not in name:
            # raw_data = raw_data @ param.t()
            agg_param_weight=torch.zeros_like(param)

            for id in list(normal_samples_id):
                temp=sample_weight[id]*param
                agg_param_weight+=temp
            agg_weights_temp.append(agg_param_weight)
    # print(f'Parameter: {name}, Gradient: {param.grad}')
    return agg_weights_temp

def get_acclumated_sample_weight(sample_weight,normal_samples_id):
    # 使用列表推导式获取索引对应的元素
    selected_elements = [sample_weight[i] for i in normal_samples_id]
    # 计算元素之和
    sum_result = torch.sum(torch.stack(selected_elements))
    return sum_result



# 实现伪码第24行
def append_client_agg_weights(ID,data):
    # selected_array = eval(f"client_agg_weights_0{ID}")
    # selected_array.append(data)

    selected_queue=eval(f"client_agg_weights_queue_0{ID}")
    selected_queue.put(data)

# 实现伪码第25行
def append_client_weights_sum(ID,data):
    # selected_array = eval(f"client_weights_sum_0{ID}")
    # selected_array.append(data)

    selected_queue = eval(f"client_weights_sum_queue_0{ID}")
    selected_queue.put(data)


# def get_average_weight_from_queue(current_items):
#
#     if len(current_items)>0:
#         param_temp_list=[]
#         param_temp=[]
#         # 展平 c 并分别收集所有 a 和 b
#         flattened_c = [item for sublist in current_items for item in sublist]
#         for item in current_items[0]:
#             temp=[tensor for tensor in flattened_c if tensor.shape == item.shape]
#             param_temp_list.append(temp)
#         for i in range(len(param_temp_list)):
#             temp =torch.mean(torch.stack(param_temp_list[i]),dim=0)
#             param_temp.append(temp)
#
#         return param_temp
#     return []
#
# # 实现伪码第37行
# def adjust_weight(ID):
#     client_weights_sum_queue = eval(f"client_weights_sum_queue_0{ID}")
#     client_agg_weights_queue = eval(f"client_agg_weights_queue_0{ID}")
#
#     average_weights_sum=torch.sum(torch.stack(client_weights_sum_queue.current_items()))/len(client_weights_sum_queue)
#
#     param_=[]
#     if average_weights_sum>0:
#         all_item=client_agg_weights_queue.current_items()
#         param_temp=get_average_weight_from_queue(all_item)
#         if len(param_temp)>0:
#             for item in param_temp:
#                temp=item/average_weights_sum
#                param_.append(temp)
#     return param_
#
# adjust_coef=0.1
#
# def update_weight(ID,ajusted_param):
#
#     model=clients[ID].model
#     i=0
#     with torch.no_grad():  # 禁用梯度计算
#         for name, param in model.named_parameters():
#             if 'bias' not in name:
#                 param.data = param.data+ajusted_param[i]*adjust_coef  # 替换原来的参数
#                 i+=1
#     return None
#
# def get_client_weight(ID):
#     model = clients[ID].model
#     params=[]
#     for name, param in model.named_parameters():
#         if 'bias' not in name:
#             params.append(param.data)
#     return params

def trimmedmean_aggregate(models, beta=0):

    if len(models) == 0:
        raise ValueError("No models to aggregate")

    models = list(models.values())
    models_params = [m for m, _ in models]

    # 创建一个零模型
    accum = (models[-1][0]).copy()
    for layer in accum:
        accum[layer] = torch.zeros_like(accum[layer])

    # 计算每个参数的trimmed mean
    for layer in accum:
        weight_layer = accum[layer]
        l_shape = list(weight_layer.shape)
        number_layer_weights = torch.numel(weight_layer)

        if l_shape == []:
            weights = torch.tensor(
                [models_params[j][layer] for j in range(len(models))])
            weights = weights.double()
            w = get_trimmedmean(weights, beta)
            accum[layer] = w
        else:
            weight_layer_flatten = weight_layer.view(number_layer_weights)
            models_layer_weight_flatten = torch.stack(
                [models_params[j][layer].view(number_layer_weights) for j in
                 range(len(models))], 0)
            trimmedmean = get_trimmedmean(models_layer_weight_flatten, beta)
            accum[layer] = trimmedmean.view(l_shape)
    return accum


def get_trimmedmean(weights, beta):

    weight_len = len(weights)
    if weight_len == 0:
        raise ValueError("No weights to aggregate")

    if weight_len <= 2 * beta:
        remaining_weights = weights
        res = torch.mean(remaining_weights, 0)
    else:
        arr_weights = np.asarray(weights)
        nobs = arr_weights.shape[0]
        start = beta
        end = nobs - beta
        atmp = np.partition(arr_weights, (start, end - 1), 0)
        sl = [slice(None)] * atmp.ndim
        sl[0] = slice(start, end)
        arr_trimmedmean = np.mean(atmp[tuple(sl)], axis=0)
        res = torch.tensor(arr_trimmedmean)
    return res


def get_average_weight_from_queue(current_items):
    if len(current_items) > 0:
        param_temp_list = []
        param_temp = []
        # 展平 c 并分别收集所有 a 和 b
        flattened_c = [item for sublist in current_items for item in sublist]
        for item in current_items[0]:
            temp = [tensor for tensor in flattened_c if
                    tensor.shape == item.shape]
            param_temp_list.append(temp)
        for i in range(len(param_temp_list)):
            temp = torch.mean(torch.stack(param_temp_list[i]), dim=0)
            param_temp.append(temp)

        return param_temp
    return []


def adjust_weight(ID, beta=1):
    client_weights_sum_queue = eval(f"client_weights_sum_queue_0{ID}")
    client_agg_weights_queue = eval(f"client_agg_weights_queue_0{ID}")

    models = {}
    for i, weights in enumerate(client_agg_weights_queue.current_items()):
        model_dict = {f'layer_{j}': w for j, w in enumerate(weights)}
        models[i] = (model_dict, 1)  # 这里假设每个模型样本数为1，可以根据实际情况调整

    aggregated_weights = trimmedmean_aggregate(models, beta)

    param_ = []
    for layer in aggregated_weights:
        param_.append(aggregated_weights[layer])

    return param_


adjust_coef = 0.1


def update_weight(ID, ajusted_param):
    model = clients[ID].model
    i = 0
    with torch.no_grad():  # 禁用梯度计算
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.data = param.data + ajusted_param[
                    i] * adjust_coef  # 替换原来的参数
                i += 1
    return None


def get_client_weight(ID):
    model = clients[ID].model
    params = []
    for name, param in model.named_parameters():
        if 'bias' not in name:
            params.append(param.data)
    return params

# def integrate_other_client_weight(ID):
#     clients_ID = list(range(num_clients))
#     num_choices = random.choice(clients_ID)
#     choices = random.sample(clients_ID, num_choices)
#     integrating_client_ids=list(set(choices)-set(ID))
#     model = clients[ID].model
#     if len(integrating_client_ids)>0:
#         for id in range(len(integrating_client_ids)):
#            param_temp=get_client_weight(id)
#            with torch.no_grad():  # 禁用梯度计算
#              i=0
#              for name, param in model.named_parameters():
#                 if 'bias' not in name:
#                     param.add_(param_temp[i]*adjust_coef)
#
#                     # param.data=(param.data+param_temp[i])/2
#                     i += 1
#     return None

# 函数用于客户端的训练和通信
def client_train_and_sync(client, data_loader_4_client, num_epochs):
    for _ in range(num_epochs):
        client.model.train()
        optimizer_with_id = eval(f"optimizer_0{client.ID}")
        for images, labels in data_loader_4_client:
            images, labels = images.to(device), labels.to(device)
            optimizer_with_id.zero_grad()
            mini_batched_images=torch.split(images,mini_batch_size,dim=0)
            mini_batched_labels=torch.split(labels,mini_batch_size,dim=0)
            ce_records=[]
            mi_records=[]
            loss_all=[]
            for i in range(len(mini_batched_images)):
                outputs=client.model(mini_batched_images[i])
                loss=criterion(outputs,mini_batched_labels[i])
                # 6
                feature_data = get_weight_x_data(client,mini_batched_images[i])
                # 8
                assignments = cluster_data(feature_data, mini_batched_images[i])
                # 10
                cross_entropy = get_cross_entropy(feature_data, assignments)
                # 9
                mutual_inf = get_mutual_information(feature_data, assignments)
                ce_records.append(cross_entropy)
                mi_records.append(mutual_inf)
                loss_all.append(loss)
            # 12, 13
            average_mi = get_average_4_mi_or_ce(mi_records)
            average_ce = get_average_4_mi_or_ce(ce_records)
            # 15
            loss_extra=get_loss_avermi_minus_averce(average_mi,average_ce)
            # testing
            # loss_extra=0
            loss_extra=torch.abs(loss_extra)
            loss_all=torch.sum(torch.stack(loss_all))+loss_extra
            loss_all.backward()
            print('client id ：',client.ID,f'Loss: {loss_all.item():.4f}')

            optimizer_with_id.step()
            # 18-35
            diff_mi = get_diff_mi_or_ce_from_average(average_mi, mi_records)
            diff_ce = get_diff_mi_or_ce_from_average(average_ce, ce_records)
            mi_lower_threshold, mi_upper_threshold = get_threshold_mi_or_ce(diff_mi)
            ce_lower_threshold, ce_upper_threshold = get_threshold_mi_or_ce(diff_ce)
            mi_anomaly = detect_outliers(diff_mi, mi_lower_threshold, mi_upper_threshold)
            ce_anomaly = detect_outliers(diff_ce, ce_lower_threshold, ce_upper_threshold)
            normal_samples_id=get_normal_samples(len(diff_ce), mi_anomaly, ce_anomaly)
            sample_weight = get_sample_weight(mi_records, ce_records)
            sample_weight_x_model_weight=get_sample_weight_x_model_weight(sample_weight, normal_samples_id, client)
            append_client_agg_weights(client.ID,sample_weight_x_model_weight)
            acclumated_sample_weight=get_acclumated_sample_weight(sample_weight, normal_samples_id)
            append_client_weights_sum(client.ID,acclumated_sample_weight)



   # if weights_sum

    # print('herehere........')
    # 实现伪码第36-39行
    exchanged_weights = [peer.send_weights() for peer in clients if
                         peer.ID != client.ID]
    exchanged_weights.append(client.send_weights())
    # aggregate weights
    new_weights = {}
    for key in exchanged_weights[0].keys():
        new_weights[key] = torch.mean(
            torch.stack([weights[key] for weights in exchanged_weights]), dim=0)
    client.model.load_state_dict(new_weights)
    # integrate_other_client_weight(client.ID)

    # print('client id',client.ID)
    # client.send_weights()
    # client.receive_weights()

# 函数用于模拟服务器的更新和通信，本例中没有使用
# def server_update_and_sync():
#     for _ in range(T):
#         total_weights = [torch.zeros_like(param) for param in next(iter(clients)).model.parameters()]
#         for client in clients:
#             client.send_weights()
#             client_weights = client.receive_weights()
#             for i, param in enumerate(total_weights):
#                 if client_weights!=None:
#                     if client_weights[i]!=None:
#                        param += client_weights[i] / len(clients)
#
#         # 将张量列表转换为模型参数字典
#         total_weights_dict = {k: v for k, v in zip(next(iter(clients)).model.state_dict().keys(), total_weights)}
#         server.model.load_state_dict(total_weights_dict)
#         server.send_weights(server.model.state_dict())

print("cuda",torch.cuda.is_initialized())

# 加载 FMNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)

#test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 加载 FMNIST 数据集
fmnist_full = FashionMNIST(root='./data', train=True, download=False, transform=ToTensor())
#
# index_1 = range(60000)
# fmnist_full.data = fmnist_full.data[index_1]
# fmnist_full.targets = fmnist_full.targets[index_1]


# 实例化模型，定义损失函数和优化器
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

cluster_sum=5

a_mi=0.1
b_ce=0.3

agg_weights=[]
weights_sum=[]


# 定义联邦学习参数
num_epochs = 15
learning_rate = 0.02
T = 10  # 每个客户端训练的轮数

max_client_weight_window=100
# 定义每个客户端的训练数据
num_clients = 10
client_agg_weights_00=[]
client_agg_weights_01=[]
client_agg_weights_02=[]
client_agg_weights_03=[]
client_agg_weights_04=[]
client_agg_weights_05=[]
client_agg_weights_06=[]
client_agg_weights_07=[]
client_agg_weights_08=[]
client_agg_weights_09=[]

client_weights_sum_00=[]
client_weights_sum_01=[]
client_weights_sum_02=[]
client_weights_sum_03=[]
client_weights_sum_04=[]
client_weights_sum_05=[]
client_weights_sum_06=[]
client_weights_sum_07=[]
client_weights_sum_08=[]
client_weights_sum_09=[]

client_agg_weights_queue_00=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_01=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_02=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_03=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_04=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_05=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_06=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_07=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_08=CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_09=CircularQueue_for_client(max_client_weight_window)

client_weights_sum_queue_00=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_01=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_02=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_03=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_04=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_05=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_06=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_07=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_08=CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_09=CircularQueue_for_client(max_client_weight_window)

batch_size = 64
mini_batch_size=8
# data_per_client = len(fmnist_full) // num_clients
# data_indices = [list(range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]

# 创建客户端和服务器实例
clients = [Client(MLP().to(device), ID) for ID in range(num_clients)]

# for decentralization
for client in clients:
    client.peers = [peer for peer in clients if peer.ID != client.ID]

optimizer_00 = optim.SGD(clients[0].model.parameters(), lr=learning_rate)
optimizer_01 = optim.SGD(clients[1].model.parameters(), lr=learning_rate)
optimizer_02 = optim.SGD(clients[2].model.parameters(), lr=learning_rate)
optimizer_03 = optim.SGD(clients[3].model.parameters(), lr=learning_rate)
optimizer_04 = optim.SGD(clients[4].model.parameters(), lr=learning_rate)
optimizer_05 = optim.SGD(clients[5].model.parameters(), lr=learning_rate)
optimizer_06 = optim.SGD(clients[6].model.parameters(), lr=learning_rate)
optimizer_07 = optim.SGD(clients[7].model.parameters(), lr=learning_rate)
optimizer_08 = optim.SGD(clients[8].model.parameters(), lr=learning_rate)
optimizer_09 = optim.SGD(clients[9].model.parameters(), lr=learning_rate)

# 定义损失函数和优化器
# optimizer = optim.Adam(server.model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 加载数据并创建数据加载器
# data_loaders = [DataLoader(Subset(fmnist_full, indices), batch_size=batch_size, shuffle=True) for indices in data_indices]
data_loaders,clients_index = split_fmnist_non_iid(fmnist_full,num_clients)
print('splitted fmnist data')
#


#
# # 模拟联邦学习迭代
# for epoch in range(num_epochs):
#     # 使用线程池并行训练客户端
#     with ThreadPoolExecutor() as executor:
#         for client, data_loader in zip(clients, data_loaders):
#             executor.submit(client_train_and_sync, client, data_loader)

    # 更新服务器的权重
    # server_update_and_sync()


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 在训练主循环中，评估模型
with ThreadPoolExecutor() as executor:
    for round in range(T):
        for client, data_loader in zip(clients, data_loaders):
            executor.submit(client_train_and_sync, client, data_loader, num_epochs)

        for i in range(num_clients):
            accuracy = evaluate_model(clients[i].model, test_loader)
            print('client ID:', i+1, '--', f'Accuracy: {accuracy} %')
            logging.info(f'client ID: {i+1} -- Accuracy: {accuracy} %')
