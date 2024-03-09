import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4,ResNet
from models.CDA import CDA
class CDAN(nn.Module):
    
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
        self.cda=CDA(640,16,['avg', 'max'],False)
        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)
        if is_pretraining:
            self.num_cat = num_cat
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)   
    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)#400,64,5,5
        feature_map =self.cda(feature_map)
        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
        return feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()#
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not
    def get_recon_dist(self,query,support,alpha,beta,Woodbury=False):
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
