import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import get_cls_map
import time
import SSFTTnet
import SSFTTnet_DCT
from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2

def loadData():
    # 读入数据
    # data = sio.loadmat('/content/drive/MyDrive/Data/WHU data/WHU-Hi-HongHu/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
    # labels = sio.loadmat('/content/drive/MyDrive/Data/WHU data/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    data = sio.loadmat('/content/HSI_SSFTT_PCD/data/Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = sio.loadmat('/content/HSI_SSFTT_PCD/data/Indian_pines_gt.mat')['indian_pines_gt']
    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels

output_units = 16

# SuperPixel Segmentation
def superpixel_segmentation(image):
  # image = img_as_float(io.imread("/content/drive/Othercomputers/Dell Inspiron (Aka Alpha)/IIT BHU/16888.png"))
  # apply SLIC and extract (approximately) the supplied number
  # of segments
  
  ld = int(''.join(map(str, image[1,1].shape)))
  xc = img_as_float(image.copy())
  xcp = img_as_float(image.copy())
  for i in range(ld):
    xcp[:,:,i]= slic(xc[:,:,i],n_segments = 300,compactness=1)
  return xcp


def kmeansnew(image):
  last_dimension = int(''.join(map(str, image[1,1].shape)))
  pixel_vals = image.reshape((-1,last_dimension))
  # Convert to float type
  pixel_vals = np.float32(pixel_vals)
  #the below line of code defines the criteria for the algorithm to stop running,
  #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
  #becomes 85%
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1)
  
  # then perform k-means clustering wit h number of clusters defined as 3
  #also random centres are initially choosed for k-means clustering
  k = output_units
  retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  # convert data into 8-bit values
  centers = np.uint8(centers)
  segmented_data = centers[labels.flatten()]
  # reshape data into the original image dimensions
  segmented_image = segmented_data.reshape((image.shape))
  return segmented_image

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 128

def custom_softmax(y):
  """Softmax function for 2D array (rows)"""
  for i in range(15):
    e_x = np.exp(y[:,:,i] - np.max(y[:,:,i], axis=1, keepdims=True))
    y[:,:,i] = e_x / np.sum(e_x, axis=1, keepdims=True)
  return y

patch_size = 9

def create_data_loader():
    # 地物类别
    class_num = 16
    # 读入数据
    X, y = loadData()
    # 用于测试样本的比例
    test_ratio = 0.98
    # 每个像素周围提取 patch 的尺寸
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... SLIC tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components//2)
    X_slic = superpixel_segmentation(X_pca)
    X_slic = custom_softmax(X_slic)
    print('Data shape after SLIC: ', X_slic.shape)
    X_slic = X_slic + X_pca
    print('Data shape after SLIC: ', X_slic.shape)

    print('\n... ... K-means tranformation ... ...')
    X_kmean = kmeansnew(X)
    X_kmean = X_kmean + X
    X_pca = applyPCA(X_kmean, numComponents=pca_components//2)
    X_pca = custom_softmax(X_pca)
    print('Data shape after K-means: ', X_pca.shape)

    # concatenation
    X_pca = np.concatenate((X_slic, X_pca), axis = 2)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, device='cpu'):
        super(FocalLoss, self).__init__(weight)
        # focusing hyper-parameter gamma
        self.gamma = gamma

        # class weights will act as the alpha parameter
        self.weight = weight
        
        # using deivce (cpu or gpu)
        self.device = device
        
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, _input, _target):
        focal_loss = 0
        poly1 = 0
        epsilon = 1e-9
        
        for i in range(len(_input)):
            # -log(pt)
            cur_ce_loss = self.ce_loss(_input[i].view(-1, _input[i].size()[-1]), _target[i].view(-1))
            # pt
            pt = torch.exp(-cur_ce_loss)

            if self.weight is not None:
                # alpha * (1-pt)^gamma * -log(pt)
                cur_focal_loss = self.weight[_target[i]] * ((1 - pt) ** self.gamma) * cur_ce_loss
            else:
                # (1-pt)^gamma * -log(pt)
                cur_focal_loss = ((1 - pt) ** self.gamma) * cur_ce_loss
            
            focal_loss = focal_loss + cur_focal_loss
        
        focal_loss = focal_loss + epsilon*((1 - pt)**(self.gamma+1))

        if self.weight is not None:
            focal_loss = focal_loss / self.weight.sum()
            return focal_loss.to(self.device)
        
        focal_loss = focal_loss / output_units   
        return focal_loss.to(self.device)

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = SSFTTnet_DCT.SSFTTnet_DCT().to(device)
    # 交叉熵损失函数
    criterion = FocalLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats' , 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats' ]
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=200)
    # 只保存模型参数
    torch.save(net.state_dict(), 'cls_params/SSFTTnet_params.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = "cls_result/classification_report"+str(patch_size)+".txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    # get_cls_map.get_cls_map(net, device, all_data_loader, y_all)
