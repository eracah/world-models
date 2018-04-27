
# coding: utf-8

# 6. ~~set up MDN_RNN with VAE data~~ 
# 8. ~~add model saving to MDN-RNN~~
# 9. ~~tensboardize~~
# 10. put in exact alex graves hyperparams
# 11. add weight init
# 7. ~~add temperature to MDN-RNN~~
# 9. add sampling from mu and sigma of z for teacher forcing insteado fusing z exactly

# In[1]:


import torch

import numpy as np

from torch import nn

from torch.autograd import Variable

from torch.nn import functional as F
from torch.nn import init
import time
from torch import optim
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import os
from torch.utils.data import DataLoader, Dataset
import sys

if __name__ == "__main__":
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook = True

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file",type=str,default="None")
    parser.add_argument("--lr",type=float, default=0.0001)
    parser.add_argument("--ctl_type",type=str, default="lstm")
    parser.add_argument("--opt",type=str, default="adam")
    parser.add_argument("--epochs",type=int, default=200)
    parser.add_argument("--savedir",type=str, default="/data/milatmp1/racaheva")
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--az_file",type=str, default="/data/milatmp1/racaheva/az_pairs/vae_lr=0.0001_rollouts=1000_batch_size=128_opt=adam_shuffle=True/az.npz")
    args = parser.parse_args()
    
    basename="mdn_rnn"
    def mkstr(key):
        d = args.__dict__
        return "=".join([key,str(d[key])])

    output_dirname = "_".join([basename,mkstr("lr"),mkstr("opt")])

    if test_notebook:
        output_dirname = "notebook_" + output_dirname
    saved_model_dir = os.path.join(args.savedir,("%s_models/%s" %(basename, output_dirname)))
    log_dir = os.path.join(args.savedir,'.%s_logs/%s'%(basename,output_dirname))

    writer = SummaryWriter(log_dir=log_dir)

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)


# In[2]:


is_cuda = torch.cuda.is_available()


# In[3]:


def get_optim(name,  model, lr, momentum=0):
    if name == "adam":
        return optim.Adam(params=model.parameters(),
                        lr=lr)
    elif name == "sgd":
        return optim.SGD(params=model.parameters(),
                        lr=lr,
                        momentum=momentum)
    elif name == "rmsprop":
          return optim.RMSprop(params=model.parameters(),
                        lr=lr,
                        momentum=momentum)

def print_info(mode,loss,t0,it):
    print("time: %8.4f"% (time.time() - t0))
    print("%s Loss for it %i: %8.4f"%(mode.capitalize(),it,loss))
    #print("%s Accuracy for epoch %i: %8.4f"%(mode.capitalize(),epoch,acc))


# In[4]:


class LSTM(nn.Module):
    def __init__(self,batch_size,input_size,hidden_size,num_layers):
       
        super(LSTM,self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers)
        
        self.hidden_size = hidden_size

        self.batch_size = batch_size
        self.h_prev = None
        self.c_prev = None
        self.reset()
    
    def reset(self):
        self.h_prev = Variable(torch.Tensor(1,self.batch_size,self.hidden_size).normal_()) #.cuda()
        self.c_prev = Variable(torch.Tensor(1,self.batch_size,self.hidden_size).normal_())#.cuda()
        if is_cuda:
            self.h_prev = self.h_prev.cuda()
            self.c_prev = self.c_prev.cuda()
    def forward(self, az):
        

        lstm_out, (self.h_prev,self.c_prev) = self.rnn(az[None,:],(self.h_prev,self.c_prev))
        

        return lstm_out, self.h_prev


class MDN(nn.Module):
    def __init__(self,input_size,output_size,num_gaussians,nz,temperature):
        super(MDN,self).__init__()

        self.nz = nz
        self.num_gaussians = num_gaussians
        self.mdn_fc = nn.Linear(in_features=input_size,
                                out_features=output_size)
        self.temperature = temperature
            
    def postproc_mdn_out(self,mdn_out):
        mu = mdn_out[:,:self.num_gaussians*self.nz]

        sigma = mdn_out[:,self.num_gaussians*self.nz:2*self.num_gaussians*self.nz]
        
        pi = mdn_out[:,-self.num_gaussians:]
        
        
        mu = mu.resize(mu.size(0),self.num_gaussians,self.nz)
        
        sigma = torch.exp(sigma)
        sigma = sigma.resize(sigma.size(0),self.num_gaussians,self.nz)
        pi = self.temperature * pi
        pi = F.softmax(pi,dim=1)
        return mu, sigma, pi
    
    def forward(self,lstm_out):
        raw_mdn_out = self.mdn_fc(lstm_out)
        mu, sigma, pi = self.postproc_mdn_out(raw_mdn_out)
        return mu, sigma, pi
    

class M(nn.Module):
    def __init__(self,batch_size,env="CarRacing", num_gaussians=5,num_layers=1,temperature=1.): 
        super(M,self).__init__()
        if env == "CarRacing":
            self.nz = 32
            self.nh = 256
            self.action_len = 3 #3 continuous values
        elif env == "Doom":
            pass # self.nz, self.nh = 64, 512

        self.batch_size = batch_size
        self.temperature = temperature
        self.num_gaussians = num_gaussians
        self.mu_len, self.sigma_len, self.pi_len = self.nz, self.nz, 1

        self.len_mdn_output = self.num_gaussians*(self.sigma_len + self.mu_len + self.pi_len)
        
        self.lstm = LSTM(batch_size=self.batch_size,
                          input_size=self.nz + self.action_len,
                          hidden_size=self.nh,
                          num_layers=num_layers)
        
        self.mdn = MDN(input_size=self.nh,
                       output_size=self.len_mdn_output,
                       num_gaussians=self.num_gaussians,
                       nz=self.nz,temperature=self.temperature)
    

    
    def forward(self,a,z):
        self.lstm.reset()
        #print(a.size(), z.size())
        az = torch.cat((z,a),dim=-1)
        mus, sigmas, pis, hs = [],[],[],[]

        for azi in az:
            lstm_out,h = self.lstm(azi)
            mu, sigma, pi = self.mdn(lstm_out[0])
            
            mus.append(mu[None,:])
            sigmas.append(sigma[None,:])
            pis.append(pi[None,:])
            hs.append(h[None,:])
        mus = torch.cat(mus)
        sigmas = torch.cat(sigmas)
        pis = torch.cat(pis)
        hs = torch.cat(hs)

        
        return mus,sigmas,pis,hs
        
    



def mdn_criterion(mus,sigmas,pis,label):
    # z is batch of seq_len number of z's, where each z is nz long
    #print(z.size())
    # mus is for each element in the seqence, a batch of a mixture of num_guasians mean vectors, where each mean vector has nz elements
    #print(mus.size())
    # sigmas is for each element in the sequence,  a batch of a mixture of num_guassians covariance vectors, where each covariance vector has nz elements
    # and represents the diagonal of the covariance matrix
    #print(sigmas.size())
    # we want to compute the density of a batch of z's under this batch of mixtures
    # pad z with a dummy dimension to enable broadcasting over the num_mixture_components dimension
    label = torch.unsqueeze(label,dim=2)
    #print(z.size())
    # we parametrize a normal distribution for every element in the sequence for every example in the batch for every mixture for every dimension
    nd = torch.distributions.Normal(mus,sigmas)
    # because the covariance matrix is diagonal the probability if z under a given mean vector and cov matrix is the product
    # of the density of each element of z under a univariate guassian. For log prob, this turns into a sum. So
    # if we sum in the dimension of the elements of z, then we get the log density of each sequence index for each example under each mixture
    log_prob_elwise = nd.log_prob(label)
    #print(log_prob_elwise.size())
    log_prob = log_prob_elwise.sum(dim=-1)
    #print(log_prob.size())
    # pis is number of examples by mixture coefficients, so we can just elementwise multoply this with log_probs
    # and sum along the mixture component direction
    #print(pis.size())
    NLL = -(pis * log_prob).sum(dim=-1)
    # now we have negative log likelihood for each element in the sequence for each example in the batch
    #print(NLL.size())

    # now lets sum over each element in the sequence
    seq_NLL = NLL.sum(dim=0)
    #print(seq_NLL.size())
    # now we take the mean over the batch
    loss = seq_NLL.mean()
    #print(loss.size())
    return loss[0]


# In[ ]:


class AZDataset(Dataset):
    def __init__(self, a,z,transform=None):
        self.a = a
        self.z = z

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        at = self.a[idx]
        zt = self.z[idx]
        return at,zt


# In[ ]:


if __name__ == "__main__":
    az_f = np.load(args.az_file)

    a,z = az_f["a"],az_f["z"]

    a = a[:,:-1]

    azds = AZDataset(a,z)

    azdl = DataLoader(azds,shuffle=True, batch_size=args.batch_size)


    m = M(batch_size=args.batch_size)
    if is_cuda:
        m = m.cuda()
    opt = get_optim(args.opt,  m, lr=args.lr)
    for epoch in range(args.epochs):
        torch.save(m.state_dict(), '%s/curr_%s.pth' % (saved_model_dir,basename))
        losses = []
        for it, (a,z) in enumerate(azdl):  
            a.transpose_(1,0)
            z.transpose_(1,0)
            a = Variable(a).float()
            zinp = Variable(z[:-1]).float()
            #our label is the NEXT frame in the sequence, so the az that is input is matched with the next frame down
            # so we don't need the first frame for our labels
            label = Variable(z[1:]).float()
            if is_cuda:
                zinp = zinp.cuda()
                a = a.cuda()
                label = label.cuda()
            m.zero_grad()
            # we will have one more frame than action because we don't take an action after the last frame
            # here we push the az's through the rnn to get parameters of a mixture of guassians
            # we don't throw the last z in there b/c it has no action for it
            mus,sigmas,pis,hs = m(a,zinp)
            loss = mdn_criterion(mus,sigmas,pis,label)
            losses.append(loss.data[0])
            
            #print(loss.data[0])
            writer.add_scalar("iter_loss",loss.data[0],global_step=epoch*len(azds) + it)
            loss.backward()
            opt.step()
        torch.save(m.state_dict(), '%s/curr_%s.pth' % (saved_model_dir,basename))
        if epoch % 20 == 0:
            torch.save(m.state_dict(), '%s/curr_%s_%s.pth' % (saved_model_dir,basename,epoch))
        writer.add_scalar("loss",np.mean(losses),global_step=epoch)

