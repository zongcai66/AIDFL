import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
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
from ResNet import ResNet18

# os.environ["OMP_NUM_THREADS"] = "16"
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(filename='CIFAR10_AIDFL_baseline.txt',
                    level=logging.INFO, format='%(asctime)s - %(message)s')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        return self.current_items()

    def __len__(self):
        return self.size


# Split the dataset to non-IID
def split_cifar10_non_iid(dataset, num_clients):
    # split non-iid
    class_indices = [[] for _ in range(10)]
    for idx, (data, label) in enumerate(dataset):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    for class_idx in range(10):
        random_permutation = np.random.permutation(class_indices[class_idx])
        num_samples_per_client = len(random_permutation) // num_clients
        for client_idx in range(num_clients):
            start_idx = client_idx * num_samples_per_client
            end_idx = (client_idx + 1) * num_samples_per_client if client_idx < num_clients - 1 else len(random_permutation)
            client_indices[client_idx].extend(random_permutation[start_idx:end_idx])

    dataloaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=True)
        dataloaders.append(dataloader)

    # split iid data
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


# Define a MLP model
class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        self.cl1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.cl2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        # op_size(input=8,kernel=3,padding=0,stride=1)*op_size(input=8,kernel=3,padding=0,stride=1)), 100)
        self.fc1 = nn.Linear((16 * 6 * 6), 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.cl1(x)))
        x = self.pool(F.relu(self.cl2(x)))
        # op_size(input=8,kernel=3,padding=0,stride=1)*op_size(input=8,kernel=3,padding=0,stride=1)))
        x = x.view(-1, (16 * 6 * 6))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32)  # flatten
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def Resnet_prepare():
    model = ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, 10)  # 将最后的全连接层改掉
    model.load_state_dict(
        torch.load('checkpoint/cifar10_pretrain.pt', map_location="cpu"))
    model = model.to(device)
    return model


model = Resnet_prepare()
server_model = Resnet_prepare()


# Define the loss function
def entropy_loss(weights):
    entropy = torch.sum(-F.softmax(weights, dim=0) * F.log_softmax(weights, dim=0))
    return entropy


# Define a client class
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

# 6
def get_weight_x_data(client,batched_data):
    raw_data=batched_data.view(-1, 784)
    for name, param in client.model.named_parameters():
        if 'bias' not in name:
            raw_data=raw_data @ param.t()
        # print(f'Parameter: {name}, Gradient: {param.grad}')

    mean = raw_data.mean(dim=0, keepdim=True)
    std = raw_data.std(dim=0, keepdim=True)

    raw_normalized = (raw_data - mean) / (std + 1e-10)  # 加上一个小常数以避免除以零

    return raw_normalized

# 8
def cluster_data(feature_data,raw_data):
        kmeans = KMeans(n_clusters=cluster_sum)
        assignments = kmeans.fit_predict(feature_data.detach().numpy())
        # distances = torch.cdist(raw_data, torch.tensor(kmeans.cluster_centers_))
        return assignments

# 10
def get_conditional_entropy(feature_data, assignments):
    # Compute the entropy of the feature data
    H_F = von_neumann_entropy(feature_data)

    # Reshape assignments to have the same shape as feature_data
    assignments = torch.tensor(assignments)
    assignments = torch.unsqueeze(assignments, 1)

    # Compute the mutual information
    I_FA = mutual_information(assignments.float(), feature_data.t())

    # Compute the conditional entropy H(F|A) = H(F) - I(F;A)
    H_F_given_A = H_F - I_FA

    return H_F_given_A


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

# 9
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
# 12, 13
def get_average_4_mi_or_ce(data):
    concatenated_tensor = torch.stack(data)
    mean_value = torch.mean(concatenated_tensor.float())
    return mean_value

# 15
def get_loss_avermi_minus_averce(average_mi,average_ce):
    return a_mi*average_mi-b_ce*average_ce

# 19, 20
def get_diff_mi_or_ce_from_average(average,records):
    diff = [item-average for item in records]
    return diff

# 24
def get_sample_weight(mi_record,ce_record):
    mi_record_safe = torch.clamp(torch.stack(mi_record), min=1e-10)
    mi_log_values = torch.log(mi_record_safe)
    mi_log_sum = torch.sum(mi_log_values)
    mi_weights = mi_log_values / mi_log_sum

    ce_record_safe = torch.clamp(torch.stack(ce_record), min=1e-10)
    ce_log_values = torch.log(ce_record_safe)
    ce_log_sum = torch.sum(ce_log_values)
    ce_weights = ce_log_values / ce_log_sum


    reciprocal_weights1 = 1.0 / mi_weights
    reciprocal_weights2 = 1.0 / ce_weights

    sum_reciprocal_weights = reciprocal_weights1 + reciprocal_weights2

    harmonic_weights = 1.0 / sum_reciprocal_weights

    return harmonic_weights

# 21
def get_threshold_mi_or_ce(data_tensor, threshold=3):
    mean = torch.mean(torch.stack(data_tensor))
    std = torch.std(torch.stack(data_tensor))
    # standardize the data
    # standardized_data = (data_tensor - mean) / std
    # set the threshold
    lower_threshold = mean - threshold * std
    upper_threshold = mean + threshold * std
    return lower_threshold,upper_threshold

# 21, 22
def detect_outliers(data_tensor, lower_threshold,upper_threshold):
    # mark the outliers
    data_tensor = torch.stack(data_tensor)
    outliers = torch.where((data_tensor < lower_threshold) | (data_tensor > upper_threshold))[0]

    return outliers


# 23
def get_normal_samples(full_length, mi_anomaly, ce_anomaly):
   mi = [mi.numpy()  for mi in mi_anomaly]
   ce = [ce.numpy() for ce in ce_anomaly]
   mi_or_ce = set(mi).union(set(ce))
   full_set = np.arange(0, full_length)
   normal_samples = set(full_set)-mi_or_ce
   return normal_samples

# 24
def get_sample_weight_x_model_weight(sample_weight,normal_samples_id,client):
    agg_weights_temp=[]
    normal_samples_number= len(normal_samples_id)
    for name, param in client.model.named_parameters():
        if 'bias' not in name:
            # raw_data = raw_data @ param.t()
            agg_param_weight=torch.zeros_like(param)

            for id in list(normal_samples_id):
                temp=sample_weight[id]*param
                agg_param_weight += temp
            agg_weights_temp.append(agg_param_weight)
    # print(f'Parameter: {name}, Gradient: {param.grad}')
    return agg_weights_temp

def get_acclumated_sample_weight(sample_weight,normal_samples_id):
    selected_elements = [sample_weight[i] for i in normal_samples_id]
    sum_result = torch.sum(torch.stack(selected_elements))
    return sum_result



# 24
def append_client_agg_weights(ID,data):
    # selected_array = eval(f"client_agg_weights_0{ID}")
    # selected_array.append(data)

    selected_queue=eval(f"client_agg_weights_queue_0{ID}")
    selected_queue.put(data)

# 25
def append_client_weights_sum(ID,data):
    # selected_array = eval(f"client_weights_sum_0{ID}")
    # selected_array.append(data)

    selected_queue = eval(f"client_weights_sum_queue_0{ID}")
    selected_queue.put(data)


def get_average_weight_from_queue(current_items):

    if len(current_items)>0:
        param_temp_list=[]
        param_temp=[]
        # flatten c and collect a and b
        flattened_c = [item for sublist in current_items for item in sublist]
        for item in current_items[0]:
            temp=[tensor for tensor in flattened_c if tensor.shape == item.shape]
            param_temp_list.append(temp)
        for i in range(len(param_temp_list)):
            temp =torch.mean(torch.stack(param_temp_list[i]),dim=0)
            param_temp.append(temp)
        return param_temp
    return []

# 37
def adjust_weight(ID):
    client_weights_sum_queue = eval(f"client_weights_sum_queue_0{ID}")
    client_agg_weights_queue = eval(f"client_agg_weights_queue_0{ID}")

    average_weights_sum=torch.sum(torch.stack(client_weights_sum_queue.current_items()))/len(client_weights_sum_queue)

    param_= []
    if average_weights_sum>0:
        all_item=client_agg_weights_queue.current_items()
        param_temp=get_average_weight_from_queue(all_item)
        if len(param_temp)>0:
            for item in param_temp:
               temp=item/average_weights_sum
               param_.append(temp)
    return param_

adjust_coef=0.1

def update_weight(ID,ajusted_param):
    model=clients[ID].model
    i=0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.data = param.data+ajusted_param[i]*adjust_coef
                i+=1
    return None

def get_client_weight(ID):
    model = clients[ID].model
    params=[]
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
#            with torch.no_grad():
#              i=0
#              for name, param in model.named_parameters():
#                 if 'bias' not in name:
#                     param.add_(param_temp[i]*adjust_coef)
#
#                     # param.data=(param.data+param_temp[i])/2
#                     i += 1
#     return None

def client_train_and_sync(client, data_loader_4_client, num_epochs):
    for _ in range(num_epochs):
        client.model.train()
        optimizer_with_id = eval(f"optimizer_0{client.ID}")
        for images, labels in data_loader_4_client:
            images, labels = images.to(device), labels.to(device)
            optimizer_with_id.zero_grad()
            outputs = client.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_with_id.step()
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
                conditional_entropy = get_conditional_entropy(feature_data, assignments)
                # 9
                mutual_inf = get_mutual_information(feature_data, assignments)
                ce_records.append(conditional_entropy)
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
    # 36-39
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
#         total_weights_dict = {k: v for k, v in zip(next(iter(clients)).model.state_dict().keys(), total_weights)}
#         server.model.load_state_dict(total_weights_dict)
#         server.send_weights(server.model.state_dict())

from torch.utils.data import Subset

def add_x_to_image(img):
    """
    Add a 10*10 pixels X at the top-left of a image
    """
    size = 5
    for i in range(0, size):
        for j in range(0, size):
            img[i][j] = 255
        # img[i][size - i - 1] = 255
    return torch.tensor(img).clone().detach()

import copy

def labelFlipping(dataset, indices, poisoned_persent=0, targeted=False, target_label=4, target_changed_label=7):
    """
    select flipping_persent of labels, and change them to random values.
    Args:
        dataset: the dataset of training data, torch.util.data.dataset like.
        indices: Indices of subsets, list like.
        flipping_persent: The ratio of labels want to change, float like.
    """
    new_dataset = copy.deepcopy(dataset)
    targets = new_dataset.targets.detach().clone()
    num_indices = len(indices)
    # classes = new_dataset.classes
    # class_to_idx = new_dataset.class_to_idx
    # class_list = [class_to_idx[i] for i in classes]
    class_list = set(targets.tolist())
    if targeted == False:
        num_flipped = int(poisoned_persent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_flipped > num_indices:
            return new_dataset
        flipped_indice = random.sample(indices, num_flipped)

        for i in flipped_indice:
            t = targets[i]
            flipped = torch.tensor(random.sample(class_list, 1)[0])
            while t == flipped:
                flipped = torch.tensor(random.sample(class_list, 1)[0])
            targets[i] = flipped
    else:
        for i in indices:
            if int(targets[i]) == int(target_label):
                targets[i] = torch.tensor(target_changed_label)
    new_dataset.targets = targets
    return new_dataset

from skimage.util import random_noise

def datapoison(dataset, indices, poisoned_persent, poisoned_ratio, targeted=False, target_label=3, noise_type="salt", backdoor_validation=False):
    """
    Function to add random noise of various types to the dataset.
    """
    new_dataset = copy.deepcopy(dataset)
    train_data = new_dataset.data
    targets = new_dataset.targets
    num_indices = len(indices)

    if not targeted:
        num_poisoned = int(poisoned_persent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_poisoned > num_indices:
            return new_dataset
        poisoned_indice = random.sample(indices, num_poisoned)

        for i in poisoned_indice:
            t = train_data[i]
            if noise_type == "salt":
                # Replaces random pixels with 1.
                noise_img = random_noise(t, mode=noise_type, amount=poisoned_ratio)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                poisoned = torch.tensor(noise_img)

            elif noise_type == "gaussian":
                # Gaussian-distributed additive noise.
                # poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
                noise_img = random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                poisoned = torch.tensor(noise_img)
            elif noise_type == "s&p":
                # Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
                # poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
                noise_img = random_noise(t, mode=noise_type, amount=poisoned_ratio)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                poisoned = torch.tensor(noise_img)
            elif noise_type == "nlp_rawdata":
                # for NLP data, change the word vector to 0 with p=poisoned_ratio
                # poisoned = poison_to_nlp_rawdata(t, poisoned_ratio)
                poisoned=None
            else:
                print("ERROR: @datapoisoning: poison attack type not supported.")
                poisoned = t
            train_data[i] = poisoned
    else:
        if backdoor_validation:
            # mark all instances for testing
            print("Datapoisoning: generating watermarked samples for testing (all classes)")
            for i in indices:
                t = train_data[i]
                poisoned = add_x_to_image(t)
                train_data[i] = poisoned
        else:
            # only mark samples from specific target for training
            print("Datapoisoning: generating watermarked samples for training, target: " + str(target_label))
            for i in indices:
                if int(targets[i]) == int(target_label):
                    t = train_data[i]
                    poisoned = add_x_to_image(t)
                    train_data[i] = poisoned
    new_dataset.data = train_data
    return new_dataset


class ChangeableSubset(Subset):
    """
    Could change the elements in Subset Class
    """

    def __init__(self,
                 dataset,
                 indices,
                 label_flipping=False,
                 data_poisoning=False,
                 poisoned_persent=0,
                 poisoned_ratio=0,
                 targeted=False,
                 target_label=0,
                 target_changed_label=0,
                 noise_type="salt"):
        super().__init__(dataset, indices)
        new_dataset = copy.copy(dataset)
        self.dataset = new_dataset
        self.indices = indices
        self.label_flipping = label_flipping
        self.data_poisoning = data_poisoning
        self.poisoned_persent = poisoned_persent
        self.poisoned_ratio = poisoned_ratio
        self.targeted = targeted
        self.target_label = target_label
        self.target_changed_label = target_changed_label
        self.noise_type = noise_type

        if self.label_flipping:
            self.dataset = labelFlipping(self.dataset, self.indices, self.poisoned_persent, self.targeted, self.target_label, self.target_changed_label)
        if self.data_poisoning:
            self.dataset = datapoison(self.dataset, self.indices, self.poisoned_persent, self.poisoned_ratio, self.targeted, self.target_label, self.noise_type)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

print("cuda",torch.cuda.is_initialized())

# load FMNIST
transform_data = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

test_dataset = torchvision.datasets.CIFAR10(root='./data3', train=False,
                                            download=True,
                                            transform=transform_data)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64,
                                          shuffle=False)

# load FMNIST
cifar10_full = torchvision.datasets.CIFAR10(root='./data3', train=True,
                                           download=True,
                                           transform=transform_data)
# index_1 = range(60000)
# fmnist_full.data = fmnist_full.data[index_1]
# fmnist_full.targets = fmnist_full.targets[index_1]


# MLP model and loss function
model = Resnet_prepare()
criterion = nn.CrossEntropyLoss()

cluster_sum = 5

a_mi = 0.1
b_ce = 0.3

agg_weights = []
weights_sum = []

num_epochs = 3
learning_rate = 0.001
T = 10

max_client_weight_window = 100
num_clients = 5
client_agg_weights_00 = []
client_agg_weights_01 = []
client_agg_weights_02 = []
client_agg_weights_03 = []
client_agg_weights_04 = []
# client_agg_weights_05 = []
# client_agg_weights_06 = []
# client_agg_weights_07 = []
# client_agg_weights_08 = []
# client_agg_weights_09 = []

client_weights_sum_00 = []
client_weights_sum_01 = []
client_weights_sum_02 = []
client_weights_sum_03 = []
client_weights_sum_04 = []
# client_weights_sum_05 = []
# client_weights_sum_06 = []
# client_weights_sum_07 = []
# client_weights_sum_08 = []
# client_weights_sum_09 = []

client_agg_weights_queue_00 = CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_01 = CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_02 = CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_03 = CircularQueue_for_client(max_client_weight_window)
client_agg_weights_queue_04 = CircularQueue_for_client(max_client_weight_window)
# client_agg_weights_queue_05 = CircularQueue_for_client(max_client_weight_window)
# client_agg_weights_queue_06 = CircularQueue_for_client(max_client_weight_window)
# client_agg_weights_queue_07 = CircularQueue_for_client(max_client_weight_window)
# client_agg_weights_queue_08 = CircularQueue_for_client(max_client_weight_window)
# client_agg_weights_queue_09 = CircularQueue_for_client(max_client_weight_window)

client_weights_sum_queue_00 = CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_01 = CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_02 = CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_03 = CircularQueue_for_client(max_client_weight_window)
client_weights_sum_queue_04 = CircularQueue_for_client(max_client_weight_window)
# client_weights_sum_queue_05 = CircularQueue_for_client(max_client_weight_window)
# client_weights_sum_queue_06 = CircularQueue_for_client(max_client_weight_window)
# client_weights_sum_queue_07 = CircularQueue_for_client(max_client_weight_window)
# client_weights_sum_queue_08 = CircularQueue_for_client(max_client_weight_window)
# client_weights_sum_queue_09 = CircularQueue_for_client(max_client_weight_window)

batch_size = 64
mini_batch_size = 8
# data_per_client = len(fmnist_full) // num_clients
# data_indices = [list(range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]

# server_model = MLP().to(device)
# server = Server(server_model)



clients = [Client(Resnet_prepare().to(device), ID) for ID in range(num_clients)]

# for decentralization
for client in clients:
    client.peers = [peer for peer in clients if peer.ID != client.ID]

optimizer_00 = optim.SGD(clients[0].model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
optimizer_01 = optim.SGD(clients[1].model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
optimizer_02 = optim.SGD(clients[2].model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
optimizer_03 = optim.SGD(clients[3].model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
optimizer_04 = optim.SGD(clients[4].model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
# optimizer_05 = optim.SGD(clients[5].model.parameters(), lr=learning_rate,
#                          momentum=0.9, weight_decay=5e-4)
# optimizer_06 = optim.SGD(clients[6].model.parameters(), lr=learning_rate,
#                          momentum=0.9, weight_decay=5e-4)
# optimizer_07 = optim.SGD(clients[7].model.parameters(), lr=learning_rate,
#                          momentum=0.9, weight_decay=5e-4)
# optimizer_08 = optim.SGD(clients[8].model.parameters(), lr=learning_rate,
#                          momentum=0.9, weight_decay=5e-4)
# optimizer_09 = optim.SGD(clients[9].model.parameters(), lr=learning_rate,
#                          momentum=0.9, weight_decay=5e-4)

# optimizer = optim.Adam(server.model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# load data
# data_loaders = [DataLoader(Subset(fmnist_full, indices), batch_size=batch_size, shuffle=True) for indices in data_indices]
data_loaders, clients_index = split_cifar10_non_iid(cifar10_full, num_clients)

client_subset_01= ChangeableSubset(
    cifar10_full,clients_index[0], label_flipping=True, data_poisoning=False, poisoned_persent=1, targeted=False)
client_subset_02= ChangeableSubset(
    cifar10_full,clients_index[2], label_flipping=True, data_poisoning=False, poisoned_persent=1, targeted=False)
client_subset_04= ChangeableSubset(
    cifar10_full,clients_index[4], label_flipping=True, data_poisoning=False, poisoned_persent=1, targeted=False)


data_loaders[0]=DataLoader(client_subset_01, batch_size=64, shuffle=True)
data_loaders[2]=DataLoader(client_subset_02, batch_size=64, shuffle=True)
data_loaders[4]=DataLoader(client_subset_04, batch_size=64, shuffle=True)

print('splitted cifar10 data')
#



# for epoch in range(num_epochs):
#     with ThreadPoolExecutor() as executor:
#         for client, data_loader in zip(clients, data_loaders):
#             executor.submit(client_train_and_sync, client, data_loader)

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

# evaluation
with ThreadPoolExecutor() as executor:
    for round in range(T):
        for client, data_loader in zip(clients, data_loaders):
            executor.submit(client_train_and_sync, client, data_loader,
                            num_epochs)

        for i in range(num_clients):
            accuracy = evaluate_model(clients[i].model, test_loader)
            print('client ID:', i + 1, '--', f'Accuracy: {accuracy} %')
            logging.info(f'client ID: {i + 1} -- Accuracy: {accuracy} %')

