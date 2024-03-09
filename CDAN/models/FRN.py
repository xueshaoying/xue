import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4,ResNet
# from models import TripletAttention
class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None):
        
        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        # temperature scaling, correspond to gamma in the paper
        #nn.Parameter()函数是torch.nn模块中的一个类，用于将张量包装成可学习的参数，并注册到当前Module中。
        #这里将标量张量[1.0]包装成一个可学习的参数self.scale，并赋值给当前Module的成员变量self.scale。
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        #requires_grad=True则表示需要计算该参数的梯度，
        #这是因为在反向传播过程中需要使用梯度来更新参数，从而使得模型更好地逼近训练数据的目标
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25
        # self.cab=CAB(3)
        # self.sknet=SKConv(3,WH=7056,M=2,G=1,r=2)
        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        #我们将一个大小为2的零张量包装成一个可学习的参数self.r。
        #在模型的训练过程中，该参数的值将被不断更新，从而使得模型更好地拟合训练数据
        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)
        # self.r_Q = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)
        if is_pretraining:
            # number of categories during pre-training
            #预训练的类数
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            #可学习参数self.cat_mat，num_cat表示类别的数量，resolution表示序列的长度，d表示特征的维度。
            #这个参数用于将输入序列中的每个元素映射到对应的类别上，并对每个类别进行特征表示。
            #在前向传播过程中，模型将输入序列与self.cat_mat进行加权求和，得到每个类别对应的特征表示。
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)   
    
#用于将输入序列inp
    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        # print("inp",inp.shape)#200 3 84 84 
        # inp=self.cab(inp)
#该方法首先通过调用self.feature_extractor来提取输入序列的特征，然后对其进行一些处理得到一个三维张量feature_map。
#第一维表示序列的数量，第二维和第三维则表示序列的空间维度和特征维度，
        feature_map = self.feature_extractor(inp)#400,64,5,5
        # print("feature_map的形状",feature_map.shape)
#如果模型的resnet属性被设置为True，则在特征提取完毕后，还会对feature_map做一个缩放的操作，即将其除以一个常数值np.sqrt(640)。
#这里的640是ResNet的特征维度，通过这个缩放操作，可以将特征值的范围控制在一个较小的范围内，有利于后续处理。
        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
#get_feature_map将feature_map重新reshape成(batch_size, H * W, C)的形状，并将第二维和第三维交换，即变成(batch_size, C, H * W)的形状。
#这里需要使用contiguous()方法将张量变为在内存中连续存储的形式，以便后续的计算。
        #需要把这里调整为四维
       # return feature_map.view(batch_size,self.d,-1).permute(0,2,1).contiguous() # N,HW,C
        return feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()#
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not
    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            #sts表示对于每个类别的支持集，将其按通道进行转置后与其自身进行矩阵乘法得到的结果，是一个 way x d x d 的张量。
            #然后加上一个单位矩阵（在PyTorch中使用torch.eye生成）与其乘以一个正则化参数λ（即lam），
            #然后对得到的矩阵求逆（使用.inverse()方法），最后将其与原始的 sts 矩阵相乘得到 hat，也是一个 way x d x d 的矩阵。
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
            hat = st.matmul(m_inv).matmul(support) # way, d, d
        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d
        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        return dist
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        feature_map = self.get_feature_map(inp)#400,25,64
        support = feature_map[:way*shot].view(way, shot*resolution , d)
        query = feature_map[way*shot:].view(way*query_shot*resolution, d)
        recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist
    def meta_test(self,inp,way,shot,query_shot):
#函数调用了一个名为get_neg_l2_dist的方法，该方法用于计算输入数据和训练集中样本之间的负L2距离。
        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)
        _,max_index = torch.max(neg_l2_dist,1)
#最后，函数返回预测结果（max_index）。
        return max_index
    def forward_pretrain(self,inp):
        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)
        feature_map = feature_map.view(batch_size*self.resolution,self.d)
        alpha = self.r[0]
        beta = self.r[1]
        #cat_mat 是将支持集中的所有样本在第一维度（即类别维度）上进行拼接得到的张量。
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        #其中num_cat 是分类任务中的类别数。然后取每个查询集样本在特征图中的平均值，得到一个形状为 (batch_size,num_cat) 的张量
        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat
        #代码将 neg_l2_dist 乘上一个尺度参数 scale，并使用 softmax 函数将其转换为概率分布
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)
        #其中每行是对应查询样本属于每个类别的概率的对数。
        return log_prediction
    def forward(self,inp):
        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
            
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction, support
