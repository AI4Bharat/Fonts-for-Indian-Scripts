import torch.nn as nn
import basicBlock as block
import torch.nn.functional as F
import torchvision.transforms as transforms

def stage_downsampling(batch, stage):
    input_destyle, style_images, label = batch
    input_destyle = nn.Upsample((2**stage)*10)(input_destyle)
    style_images = nn.Upsample((2**stage)*10)(style_images)
    label = nn.Upsample((2**stage)*10)(label)
    return input_destyle, style_images, label

def print_stat(full_plot = False, label_img = True):
    
    if full_plot:
        imgs = []
        labs = []
        eng_imgs = 22 if label_img == True else 23
        count = 0
        
        imgs.append(inv_normalize(inp_img[0, :, :, :].detach()).cpu().numpy().transpose(1, 2, 0).squeeze())
        labs.append("Input Font")
        for i in range(eng_imgs):
            imgs.append(inv_normalize(font_img[0, :, :, :].detach()).cpu().numpy().transpose(1, 2, 0)[:, :, i])
            labs.append("English Font Char %d"%(i+1))
        
        if label_img == True:
            imgs.append(label[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0).squeeze())
            labs.extend(["Label"])
            
        imgs.append(x.detach()[0, :, :, :].cpu().numpy().transpose(1, 2, 0).squeeze())
        labs.extend(["Generated"])
        
        fig, ax = plt.subplots(5, 5, figsize=(13, 15))
        for i in range(5):
            for j in range(5):
                ax[i][j].imshow(imgs[count], cmap='gray')
                ax[i][j].set_title(labs[count])
                count += 1
        fig.tight_layout()
        plt.show()
        
        return 
    indx = np.random.randint(0, 26)
    imgs = [inv_normalize(inp_img[0, :, :, :].detach()).cpu().numpy().transpose(1, 2, 0).squeeze(),
            inv_normalize(font_img[0, :, :, :].detach()).cpu().numpy().transpose(1, 2, 0)[:, :, indx],
            inv_normalize(label[0, :, :, :].detach()).cpu().numpy().transpose(1, 2, 0).squeeze(),
            inv_normalize(x.detach()[0, :, :, :]).cpu().numpy().transpose(1, 2, 0).squeeze()]
    lab = ['Input Font', 'Ref Font', 'Label', 'Generated']
    fig, ax = plt.subplots(1, 4, figsize=(13, 12))
    for i in range(4):
        ax[i].imshow(imgs[i], cmap='gray')
        ax[i].set_title(lab[i])
    fig.tight_layout()
    plt.show()


transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485], [0.229]),])
inv_normalize = transforms.Normalize(mean=[-0.485/0.229], std=[1/0.229])

data = ImageDataset(data_path, transform = transform_img)

train_size = int(0.9 * len(data))
test_size = len(data) - train_size
print(train_size)

train_data, validation_data = random_split(data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=1, shuffle=True)

lmbda_l1 = 10
lmbda_gp = 10
styl_net = StylisationNetwork(lmbda_l1, lmbda_gp, norm_layer = 'InN')

styl_net.load()

Gen_Loss = []
Critic_Loss = []
stage = 3
epochs = 2
count = 0

styl_net.train()

for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
#        tic = time.time()
        inp_img, font_img, label  = stage_downsampling(batch, stage)
        
        inp_img = inp_img.to(device)
        font_img = font_img.to(device)
        label = label.to(device)
        
        adv_loss_critic, adv_loss_gen, x = styl_net(inp_img, font_img, label, stage)
        
        if count == 0:
            print("Epoch [%d/%d] after iteration %d: \n" %(epoch+1, epochs, count))
            print_stat()
            
        if count%100 == 0:
            Gen_Loss.append(adv_loss_gen.item())
            Critic_Loss.append(adv_loss_critic.item())
        count += 1
#        print('Time Taken : ', (time.time() - tic))
    print("Epoch [%d/%d] after iteration %d: \n" %(epoch+1, epochs, count))
    print_stat()
styl_net.save()

plt.plot(Gen_Loss, label = 'Gen Loss')
plt.plot(Critic_Loss, label = 'Critic Loss')
plt.legend()
plt.show()

count = 0
styl_net.eval()

for i, batch in enumerate(validation_loader):
    inp_img, font_img, label = stage_downsampling(batch, stage)

    inp_img = inp_img.to(device)
    font_img = font_img.to(device)
    label = label.to(device)

    x = styl_net.test(inp_img, font_img, stage)

    print_stat()
    
    if count == 20:
        break
    count += 1
