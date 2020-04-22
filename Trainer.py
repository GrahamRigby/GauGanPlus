from torch.utils.data import Dataset, DataLoader
from Preprocess import get_data
from Models import *
import random
import cv2
from torch.autograd import Variable

dataset = get_data()
#----------------------------------------------
#Hyper parameters
batch_size = 28
img_size = 128
latent_dim = 100
d_thresh = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------
#Training Loop
# Initialize generator and discriminator and loss functions
bceLoss = nn.MSELoss()
generator = Generator(img_size, latent_dim)
multi_discriminator = MultiDiscriminator(img_size)
print(generator)
print(multi_discriminator)
generator.apply(weights_init_normal)
multi_discriminator.apply(weights_init_normal)

if False:
    generator.load_state_dict(torch.load('generator_model.pt'))
    multi_discriminator.load_state_dict(torch.load('multi_discriminator_model.pt'))

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    multi_discriminator.cuda()
    bceLoss.cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(multi_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
schedulerG = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.1)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.2)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
training_epochs = 5000
for epoch in range(1000):
    for i, data in enumerate(loader, 0):
        # Final Data preparation stuff
        inputs, feature_maps, medium_maps, small_maps, tiny_maps, micro_maps = data
        gen_inputs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny, gen_maps_micro = \
            inputs[:(len(inputs)//2)].to(device), feature_maps[:(len(inputs)//2)].to(device),\
            medium_maps[:(len(inputs)//2)].to(device), small_maps[:(len(inputs)//2)].to(device),\
            tiny_maps[:(len(inputs)//2)].to(device), micro_maps[:(len(inputs)//2)].to(device)
        disc_inputs, disc_maps_large, disc_maps_medium, disc_maps_small, disc_maps_tiny, disc_maps_micro = \
            inputs[(len(inputs)//2):].to(device), feature_maps[(len(inputs)//2):].to(device),\
            medium_maps[(len(inputs)//2):].to(device), small_maps[(len(inputs)//2):].to(device),\
            tiny_maps[(len(inputs)//2):].to(device), micro_maps[(len(inputs)//2):].to(device)

        #noise injection
        gen_inputs = torch.stack([x for x in gen_inputs]).to(device)
        disc_inputs = torch.stack([x for x in disc_inputs]).to(device)

        #define labels
        real_labels = Variable(torch.ones((disc_inputs.shape[0]), device=device))
        fake_labels = Variable(torch.zeros((gen_inputs.shape[0]), device=device))

        #Discriminator step
        optimizer_D.zero_grad()
        d1_out, d2_out = multi_discriminator(disc_inputs, disc_maps_large, disc_maps_medium, disc_maps_small, disc_maps_tiny, disc_maps_micro)
        loss1 = bceLoss(d1_out[0], real_labels)
        loss2 = bceLoss(d1_out[1], real_labels)
        loss4 = bceLoss(d2_out[0], real_labels)
        D_loss_real = ((loss1+loss2)/4 + (loss4))
        D_loss_real.backward()
        gen_imgs, kld_loss = generator(gen_inputs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny, gen_maps_micro)
        d1_out, d2_out = multi_discriminator(gen_imgs.detach(), gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny, gen_maps_micro)
        loss1 = bceLoss(d1_out[0], fake_labels)
        loss2 = bceLoss(d1_out[1], fake_labels)
        loss4 = bceLoss(d2_out[0], fake_labels)
        D_loss_fake = ((loss1+loss2)/4 + (loss4))
        d_loss = (D_loss_fake + D_loss_real)/2
        optimizer_D.step()

        #Generator step
        optimizer_G.zero_grad()
        d1_out, d2_out = multi_discriminator(gen_imgs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny, gen_maps_micro)
        loss1 = bceLoss(d1_out[0], real_labels)
        loss2 = bceLoss(d1_out[1], real_labels)
        loss4 = bceLoss(d2_out[0], real_labels)
        g_fake_loss = ((loss1+loss2)/4 + (loss4)) + kld_loss
        g_fake_loss.backward()
        optimizer_G.step()


    if epoch % 25==0 and epoch > 0:
        print("saving...")
        torch.save(generator.state_dict(), 'generator_model.pt')
        torch.save(multi_discriminator.state_dict(), 'multi_discriminator_model.pt')
        print("saving complete")

    if epoch % 5 == 0:
        print("epoch " + str(epoch) + "/%d" % training_epochs)
        print("Generator Loss: %.2f" % g_fake_loss)
        print("Discriminator Loss: %.2f" % d_loss)
        print("KLD Loss: %.2f" % kld_loss)
        im = torchvision.utils.make_grid(gen_imgs[:25].detach().cpu(), nrow=5, padding=2, normalize=False)
        torchvision.utils.save_image(gen_imgs[0].detach().cpu(), "images/Pokemon%dA.png" % epoch, normalize=False)
        torchvision.utils.save_image(gen_imgs[1].detach().cpu(), "images/Pokemon%dB.png" % epoch, normalize=False)
        torchvision.utils.save_image(gen_imgs[2].detach().cpu(), "images/Pokemon%dC.png" % epoch, normalize=False)
        torchvision.utils.save_image(gen_imgs[3].detach().cpu(), "images/Pokemon%dD.png" % epoch, normalize=False)
        torchvision.utils.save_image(gen_imgs[4].detach().cpu(), "images/Pokemon%dE.png" % epoch, normalize=False)
        torchvision.utils.save_image(im, "images/Pokemon%dGrid.png" % epoch, normalize=True)
        gen_imgs = torch.transpose(gen_imgs, 1, 2)
        gen_imgs = torch.transpose(gen_imgs, 2, 3)
        generated_poke = gen_imgs[0].detach().cpu()
        pokemon = generated_poke.numpy()
        pokemon = (pokemon-pokemon.mean()) / pokemon.std()
        pokemon = (pokemon + 1) / 2 * 255
        pokemon = np.clip(pokemon, 0, 255).astype(np.uint8)
        cv2.imwrite("images/Pokemon%d.png" % epoch, pokemon)