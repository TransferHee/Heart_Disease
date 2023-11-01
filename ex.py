import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch import nn
import torch
from tqdm import tqdm
from cnn_models import BaseCNN, M_resnet, M_vgg16, CNN_2, SE_CNN, CNN_3
from vit_models import VisionTransformer, UNetTransformer, UNetPPTransformer, EfficientUNetTransformer
from torchvision import models
from losses import Balanced_Loss

import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
class EarlyStopping:
    def __init__(self, patience=5, save_path=''):
        self.min_loss = np.inf
        self.cnt = 0
        self.patience = patience
        self.path = save_path
        
    def should_stop(self, model, loss):
        if loss <= self.min_loss:
            self.min_loss = loss
            self.cnt = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.cnt += 1
        return self.cnt >= self.patience
    
    def load(self,model):
        model.load_state_dict(torch.load(self.path))
        return model

def experiment(lr, weight_decay, batch_size, model_name, best_acc, balanced):
    seed_everything(42)
    batch_size = batch_size
    lr = lr
    num_epochs = 500
    weight_decay = weight_decay

    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = np.load('./X_train.npy'), np.load('./y_train.npy')
    X_val, y_val = np.load('./X_val.npy'), np.load('./y_val.npy')
    X_test, y_test = np.load('./X_test.npy'), np.load('./y_test.npy'),

    from torch.utils.data import TensorDataset, DataLoader

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    if model_name == 'BaseCNN':
        model = BaseCNN().to(device)
    elif model_name == 'M_resnet':
        model = M_resnet().to(device)
    elif model_name == 'M_vgg16':
        model = M_vgg16().to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif model_name == 'Densenet':
        model = models.densenet121().to(device)
    elif model_name == 'CNN_2':
        model = CNN_2().to(device)
    elif model_name == 'SE_CNN':
        model = SE_CNN().to(device)
    elif model_name == 'CNN_3':
        model = CNN_3().to(device)
    elif model_name == 'ViT':
        model = VisionTransformer(device).to(device)
    elif model_name == 'Unet':
        model = UNetTransformer(device).to(device)
    elif model_name == 'Unetpp':
        model = UNetPPTransformer(device).to(device)
    elif model_name == 'EffUnet':
        model = EfficientUNetTransformer(device).to(device)
    
    if model_name != 'M_vgg16':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f'\n==========================================================={model_name}=========================================================\n')
    
    from collections import Counter

    if balanced:
        y_val_np = np.array(y_val)
        c = Counter(y_val_np)
        samples_per_class = [i[1] for i in sorted(c.items())]
        criterion = Balanced_Loss(samples_per_class=samples_per_class, beta=0.9999)
        tmp = 'b'
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        tmp = 'nb'

    print('Train Start!!')
    
    best_acc = -1
    es = EarlyStopping(save_path=f'./Results/{model_name}_{batch_size}_{lr}_{weight_decay}_notv_{tmp}.pt')
    train_loss_l, val_loss_l, val_acc_l = [], [], []
    for epoch in range(num_epochs):
        sum_loss = 0
        model.train()
        for x, y, v in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            sum_loss += loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            total_correct = 0
            total_instances = 0
            for x, y, v in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y)
                correct = torch.argmax(pred, 1) == y
                total_correct += correct
                total_instances += len(x)
            accuracy = (total_correct / total_instances).detach().cpu().numpy().mean()
        print(f'[Epoch: {epoch+1:>3}, train_loss = {sum_loss:>.5}, val_loss = {val_loss.item():>.5}, val_acc = {accuracy:>.5}]')
        train_loss_l.append(sum_loss.item())
        val_loss_l.append(val_loss.item())
        val_acc_l.append(accuracy)
        
        if es.should_stop(model, val_loss.item()):
            print(f'Early Stopping!!')
            break
    
    model = es.load(model)
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        total_correct = 0
        total_instances = 0
        for x, y, v in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            y_pred.append(torch.argmax(pred, 1).item())
            y_true.append(y.item())
            correct = torch.argmax(pred, 1) == y
            total_correct += correct
            total_instances += len(x)
        accuracy = (total_correct / total_instances).detach().cpu().numpy().mean()
    print(f'[test acc = {accuracy:>.5}]')
    print('Current acc',accuracy, 'Best acc',best_acc)
    from sklearn.metrics import classification_report
    
    if best_acc < accuracy:
        plt.clf()
        plt.plot(train_loss_l, label='train loss')
        plt.plot(val_loss_l, label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'./Results/{model_name}_{batch_size}_{lr}_{weight_decay}_notv_{tmp}.png')
        print(classification_report(y_true, y_pred, digits=4))
    return accuracy