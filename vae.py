
# coding: utf-8

# 5. ~~set up car racing rollout dataloader for VAE~~
# 6. ~~add nice way to save action,z pairs in same dir for VAE~~
#     ~~* do we save a,z pairs or we just train it up~~
# 7. ~~add in weight init~~
# 5. ~~tensorboardize~~
# 6. ~~add weight saving~~
# 7. add disentangling
# 
# 

# In[1]:


import gym
import matplotlib.pyplot as plt
import torch

# setup rendering before importing other stuff (weird hack to avoid pyglet errors)
env = gym.make("CarRacing-v0")
_ = env.reset()
_ = env.render("rgb_array")
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
from torch import optim
import os
import time
from tensorboardX import SummaryWriter
import sys
import torchvision
from torchvision.transforms import Compose,Normalize,Resize,ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math
from torchvision.utils import make_grid


# In[2]:


if __name__ == "__main__":
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file",type=str,default="None")
    parser.add_argument("--lr",type=float, default=0.001)
    parser.add_argument("--opt",type=str, default="adam")
    parser.add_argument("--rollouts",type=int, default=4)
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--savedir",type=str, default="/data/milatmp1/racaheva")
    args = parser.parse_args()


    len_action = 3
    rollout_len = 1000
    basename="vae"
    def mkstr(key):
        d = args.__dict__
        return "=".join([key,str(d[key])])
    shuffle = True
    output_dirname = "_".join([basename,mkstr("lr"),mkstr("rollouts"),mkstr("batch_size"),mkstr("opt"),"shuffle=%s"%str(shuffle)])

    if test_notebook:
        output_dirname = "notebook_" + output_dirname
    saved_model_dir = os.path.join(args.savedir,("models/%s" % output_dirname))
    log_dir = os.path.join(args.savedir,'.%s_logs/%s'%(basename,output_dirname))
    az_pair_dir = os.path.join(args.savedir,("az_pairs/%s" % output_dirname))
    writer = SummaryWriter(log_dir=log_dir)

    if not os.path.exists(az_pair_dir):
        os.makedirs(az_pair_dir)

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)


# In[4]:


import matplotlib.pyplot as plt
#%matplotlib inline
# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
#####
# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))


# In[5]:


class FrameDataset(Dataset):
    """Dataset from a single rollout"""

    def __init__(self, data,transform=None):
        self.data = data
        # get between -1 and 1
#         self.data = ((data / 255) - 0.5 ) * 2
#         self.data = self.data.transpose(0,3,1,2)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


# In[6]:


def generate_rollout(env_name="CarRacing-v0"):
    env = gym.make(env_name)
    frames = []
    actions = []
    state = env.reset()
    frame = Image.fromarray(state, 'RGB')
    frames.append(frame)
    s = env.render("rgb_array")
    done=False
    
    while not done:
        action = env.action_space.sample()
        state,r,done,_ = env.step(action)
        frame = Image.fromarray(state, 'RGB')
        frames.append(frame)
        actions.append(action[None,:])
    actions = np.concatenate(actions)
#     frames = np.concatenate(frames)
    return frames,actions
    


# In[7]:


def make_frame_iterator(frames,batch_size=128,env_name="CarRacing-v0"):
    # b/c we use sigmoid
    transforms = Compose([Resize((64,64)),ToTensor()])#,Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    rds = FrameDataset(frames,transform=transforms)


    frame_iter = DataLoader(rds,batch_size=batch_size,shuffle=True) # hopefully this will encourage vae to learn curves
    return frame_iter
    


# In[8]:


def get_optim(name,  model, lr, momentum=0.):
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


# In[45]:


class VAE(nn.Module):
    def __init__(self,env="CarRacing"):
        super(VAE,self).__init__()
        if env == "CarRacing":
            self.nz = 32
        elif env == "Doom":
            self.nz = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4, stride=2),
            nn.ReLU())
        self.sigma_fc = nn.Linear(in_features=256*2*2,out_features=self.nz)
        self.mu_fc = nn.Linear(in_features=256*2*2,out_features=self.nz)
        
        self.decode_fc = nn.Linear(in_features=32,out_features=1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=128,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=6,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=6,stride=2),
            nn.Sigmoid())

        self._initialize_weights()

    def forward(self,x):
        vec = self.encoder(x)
        
        #flatten
        vec = vec.view(vec.size(0),-1)
        mu, sigma = self.mu_fc(vec), self.sigma_fc(vec)
        z = self.reparameterize(mu,sigma)
        im = self.decode_fc(z)
        
        #reshape into im
        im = im[:,:,None,None]
        
        xr = self.decoder(im)
        
        return xr,mu,sigma,z
        
    
    def reparameterize(self,mu,sigma):
        if self.training:
            eps = Variable(torch.randn(*sigma.size()))
            if sigma.is_cuda:
                eps = eps.cuda()
            z = mu + eps*sigma
            return z
        else:
            return mu
    
    def _initialize_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        

def vae_loss(x,xr,mu,sigma):
    mu_sum_sq = (mu*mu).sum(dim=1)
    sig_sum_sq = (sigma*sigma).sum(dim=1)
    log_term = (1 + torch.log(sigma**2)).sum(dim=1)
    kldiv = -0.5 * (log_term - mu_sum_sq - sig_sum_sq)
    
    rec = F.mse_loss(xr,x)
    
    return rec + kldiv.mean()
    
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD 


# In[50]:


if __name__ == "__main__":
    V = VAE().cuda()
    V.train()
    all_actions = np.zeros((args.rollouts,rollout_len+1,len_action))
    all_z = np.zeros((args.rollouts,rollout_len+1,V.nz))
    az_fn = os.path.join(az_pair_dir,"az.npz")

    opt = get_optim(name=args.opt,lr=args.lr,model=V)
    criterion = vae_loss
    for epoch in range(args.rollouts):
        frames,actions = generate_rollout()
        actions = all_actions[epoch,:rollout_len] = actions
        dataloader = make_frame_iterator(frames)
        it_losses = []
        zs = []
        opt.zero_grad()
        for it, x in enumerate(dataloader):
            xv = Variable(x).float().cuda()
            xr,mu,sigma,z = V(xv)

            rows = 6 #int(math.sqrt(args.batch_size))
            num_ims = rows**2
            xr_grid = make_grid(xr.data[:num_ims], rows)
            writer.add_image("x_rec", xr_grid, epoch)
            x_grid = make_grid(xv.data[:num_ims],rows)
            writer.add_image("x_orig", x_grid, epoch)
            
            
            zs.append(z)
            loss = criterion(xv,xr,mu,sigma)
            it_losses.append(loss.data[0])
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss = np.mean(it_losses)
        print(loss)
        zs = torch.cat(zs)
        all_z[epoch,:,:] = zs
        #az = zip(torch.from_numpy(actions),zs)
        writer.add_scalar("loss",scalar_value=loss,global_step=epoch)
        #save weights as cpu weights
        torch.save(V.state_dict(), '%s/currVAE.pth' % (saved_model_dir))
        if epoch % 100 == 0:
            np.savez(az_fn,a=all_actions,z=all_z)
            torch.save(V.state_dict(), '%s/currVAE_%s.pth' % (saved_model_dir,epoch))
    np.savez(az_fn,a=all_actions,z=all_z)

