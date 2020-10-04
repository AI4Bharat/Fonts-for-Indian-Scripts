import torch
import torch.nn as nn
import basicBlock as block
import ConvNetworks as CNN
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class StylisationNetwork(nn.Module):
    def __init__(self, lmbda_l1, lmbda_gp, norm_layer = 'InN'):
        super(StylisationNetwork, self).__init__()
        
        self.inp_enc = CNN.Encoder(block.BasicBlock, 1, norm_layer = norm_layer).to(device)        
        self.font_enc = CNN.Encoder(block.BasicBlock, 26, norm_layer = norm_layer).to(device)        
        self.mixer = CNN.Mixer(block.BasicBlock, norm_layer = norm_layer).to(device)
        self.dec = CNN.Generator(block.BasicBlock, self.inp_enc, norm_layer = norm_layer).to(device)
        self.critic = CNN.Critic(block.BasicBlock, norm_layer = norm_layer).to(device)
        
        self.inp_enc_optim = optim.Adam(self.inp_enc.parameters(), lr = 0.0002, weight_decay=0)
        self.font_enc_optim = optim.Adam(self.font_enc.parameters(), lr = 0.0002, weight_decay=0)
        self.dec_optim = optim.Adam(self.dec.parameters(), lr = 0.0002, weight_decay=0)
        self.mixer_optim = optim.Adam(self.mixer.parameters(), lr = 0.0002, weight_decay=0)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = 0.0002, weight_decay=0)
        
        self.l1 = nn.L1Loss()
        self.lmbda_l1 = lmbda_l1
        self.lmbda_gp = lmbda_gp
        self.embeddings = None
        
    def alpha_loss(self, input, target, epsilon):
        return torch.mean(torch.sqrt((input - target)**2 + epsilon**2))
    
    def zero_grad(self):
        self.inp_enc_optim.zero_grad()
        self.font_enc_optim.zero_grad()
        self.critic_optim.zero_grad()
        self.mixer_optim.zero_grad()
        self.dec_optim.zero_grad()
        
    def step(self):
        self.inp_enc_optim.step()
        self.font_enc_optim.step()
        self.mixer_optim.step()
        self.dec_optim.step()
        
    def train(self):
        self.inp_enc.train()
        self.font_enc.train()
        self.critic.train()
        self.mixer.train()
        self.dec.train()
        
    def eval(self):
        self.inp_enc.eval()
        self.font_enc.eval()
        self.critic.eval()
        self.mixer.eval()
        self.dec.eval()
        
    def load(self):  
            
        self.inp_enc.load_state_dict(torch.load("ProcessFont/Model/params/inp_enc.pth", map_location=device))
        self.font_enc.load_state_dict(torch.load("ProcessFont/Model/params/font_enc.pth", map_location=device))
        self.mixer.load_state_dict(torch.load("ProcessFont/Model/params/mixer.pth", map_location=device))
        self.dec.load_state_dict(torch.load("ProcessFont/Model/params/dec.pth", map_location=device))
        self.critic.load_state_dict(torch.load("ProcessFont/Model/params/critic.pth", map_location=device))
        
    def save(self):
        
        torch.save(self.inp_enc.state_dict(), "inp_enc.pth")
        torch.save(self.font_enc.state_dict(), "font_enc.pth")
        torch.save(self.mixer.state_dict(), "mixer.pth")
        torch.save(self.dec.state_dict(), "dec.pth")
        torch.save(self.critic.state_dict(), "critic.pth")
    
    def test(self, inp_img, font_img, stage):
        
        x = self.mixer(self.inp_enc(inp_img, stage), self.font_enc(font_img, stage))
        self.embeddings = x.detach().cpu().numpy()
        x = self.dec(x, stage)
        return x
    
    def calc_gradient_penalty(self, critic, real_data, fake_data, stage):
        
        alpha = torch.rand(real_data.shape[0], 1, 1, 1).to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = critic(interpolates, stage)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def forward(self, inp_img, font_img, label, stage):
        
        self.zero_grad()
        
        x = self.mixer(self.inp_enc(inp_img, stage), self.font_enc(font_img, stage))
        x = self.dec(x, stage)
        
        font_img = font_img[:, np.random.randint(0, 26), :, :]
        font_img = font_img.reshape(font_img.shape[0], 1, font_img.shape[1], font_img.shape[2])
        
        #Loss for Critic
        fake_input = torch.cat((inp_img, x.detach(), font_img), dim = 1)
        fake_output = self.critic(fake_input, stage)
        
        real_input = torch.cat((inp_img, label, font_img), dim = 1)
        real_output = self.critic(real_input, stage)
        
        gradient_penalty = self.calc_gradient_penalty(self.critic, real_input, fake_input, stage)
        adv_loss_critic = fake_output.mean() - real_output.mean() + (self.lmbda_gp * gradient_penalty)
        adv_loss_critic.backward()
        self.critic_optim.step()
        self.zero_grad()
        
        #Loss For Generator
        fake_input = torch.cat((inp_img, x, font_img), dim = 1)
        fake_output = self.critic(fake_input, stage)
        adv_loss_gen = (self.lmbda_l1 * self.l1(x, label)) - fake_output.mean()
        adv_loss_gen.backward()
        self.step()
        
        return adv_loss_critic, adv_loss_gen, x
