from torch.utils.data import Dataset, DataLoader
from Preprocess import get_data
from Models import *
import random
import cv2

dataset = get_data()
#----------------------------------------------
#Hyper parameters
batch_size = 16
img_size = 128
latent_dim = 100
d_thresh = 125
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------
#Training Loop
# Initialize generator and discriminator and loss functions
bceLoss = torch.nn.BCELoss()
generator = Generator(img_size, latent_dim)
multi_discriminator = MultiDiscriminator(img_size)
generator.apply(weights_init_normal)
multi_discriminator.apply(weights_init_normal)

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    multi_discriminator.cuda()
    bceLoss.cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9999))
optimizer_D = torch.optim.Adam(multi_discriminator.parameters(), lr=0.0004, betas=(0.5, 0.9999))
schedulerG = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=500, gamma=0.5)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=500, gamma=0.5)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
training_epochs = 5000
for epoch in range(5000):
    for i, data in enumerate(loader, 0):
        inputs, feature_maps, medium_maps, small_maps, tiny_maps, micro_maps = data
        gen_inputs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny, gen_maps_micro = \
            inputs[:(len(inputs)//2)].to(device), feature_maps[:(len(inputs)//2)].to(device),\
            medium_maps[:(len(inputs)//2)].to(device), small_maps[:(len(inputs)//2)].to(device),\
            tiny_maps[:(len(inputs)//2)].to(device), micro_maps[:(len(inputs)//2)].to(device)
        disc_inputs, disc_maps, disc_maps_medium, disc_maps_small, disc_maps_tiny, disc_maps_micro = \
            inputs[(len(inputs)//2):].to(device), feature_maps[(len(inputs)//2):].to(device),\
            medium_maps[(len(inputs)//2):].to(device), small_maps[(len(inputs)//2):].to(device),\
            tiny_maps[(len(inputs)//2):].to(device), micro_maps[(len(inputs)//2):].to(device)
        large_maps_cat = torch.cat((disc_maps, gen_maps_large), 0)
        medium_maps_cat = torch.cat((disc_maps_medium, gen_maps_medium), 0)
        small_maps_cat = torch.cat((disc_maps_small, gen_maps_small), 0)
        tiny_maps_cat = torch.cat((disc_maps_tiny, gen_maps_tiny), 0)
        micro_maps_cat = torch.cat((disc_maps_micro, gen_maps_micro), 0)

        #noise injection
        gen_inputs = torch.stack([noiseer(x) for x in gen_inputs]).to(device)
        disc_inputs = torch.stack([noiseer(x) for x in disc_inputs]).to(device)

        #define labels
        real_labels = torch.ones((disc_inputs.shape[0], 1), device=device)
        fake_labels = torch.zeros((gen_inputs.shape[0], 1), device=device)

        gen_imgs, kld_loss = generator(gen_inputs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny,
                                       gen_maps_micro)

        optimizer_D.zero_grad()
        imgs_cat = torch.cat((disc_inputs, gen_imgs.detach()), 0)
        d1_out, d2_out = multi_discriminator(imgs_cat, large_maps_cat, medium_maps_cat, small_maps_cat, tiny_maps_cat, micro_maps_cat)
        loss1 = bceLoss((d1_out[0][d1_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss2 = bceLoss((d1_out[1][d1_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss3 = bceLoss((d1_out[2][d1_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss4 = bceLoss((d1_out[3][d1_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss5 = bceLoss((d1_out[4][d1_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss6 = bceLoss((d2_out[0][d2_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss7 = bceLoss((d2_out[1][d2_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss8 = bceLoss((d2_out[2][d2_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss9 = bceLoss((d2_out[3][d2_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss10 = bceLoss((d2_out[4][d2_out[0].shape[0]//2:]).mean(), torch.tensor(0).to(device).float())
        loss11 = bceLoss((d1_out[0][:d1_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss12 = bceLoss((d1_out[1][:d1_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss13 = bceLoss((d1_out[2][:d1_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss14 = bceLoss((d1_out[3][:d1_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss15 = bceLoss((d1_out[4][:d1_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss16 = bceLoss((d2_out[0][:d2_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss17 = bceLoss((d2_out[1][:d2_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss18 = bceLoss((d2_out[2][:d2_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss19 = bceLoss((d2_out[3][:d2_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())
        loss20 = bceLoss((d2_out[4][:d2_out[0].shape[0]//2]).mean(), torch.tensor(1).to(device).float())

        d_real_loss = (loss1 + loss2 + loss6 + loss7 + loss10 + loss11 + loss16 + loss17)
        d_real_loss.backward()
        optimizer_D.step()

        #discriminators
        optimizer_G.zero_grad()
        imgs_cat = torch.cat((disc_inputs, gen_imgs), 0)
        g1_out, g2_out = multi_discriminator(imgs_cat, large_maps_cat, medium_maps_cat, small_maps_cat, tiny_maps_cat, micro_maps_cat)
        loss1 = bceLoss((g1_out[0][g1_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss2 = bceLoss((g1_out[1][g1_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss3 = bceLoss((g1_out[2][g1_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss4 = bceLoss((g1_out[3][g1_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss5 = bceLoss((g1_out[4][g1_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss6 = bceLoss((g2_out[0][g2_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss7 = bceLoss((g2_out[1][g2_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss8 = bceLoss((g2_out[2][g2_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss9 = bceLoss((g2_out[3][g2_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())
        loss10 = bceLoss((g2_out[4][g2_out[0].shape[0]//2:]).mean(), torch.tensor(1).to(device).float())

        g_fake_loss = (loss1 + loss2 + loss6 + loss7) + kld_loss
        g_fake_loss.backward()
        optimizer_G.step()

    schedulerG.step()
    schedulerD.step()

    if epoch % 5 == 0:
        print("epoch " + str(epoch) + "/%d" % training_epochs)
        print("Generator Loss: %.2f" % g_fake_loss)
        print("Discriminator Loss: %.2f" % d_real_loss)
        print("KLD Loss: %.2f" % kld_loss)
        im = torchvision.utils.make_grid(gen_imgs[:25].detach().cpu(), nrow=5, padding=2, normalize=True)
        torchvision.utils.save_image(gen_imgs[0].detach().cpu(), "images/Pokemon%dA.png" % epoch, normalize=True)
        torchvision.utils.save_image(gen_imgs[1].detach().cpu(), "images/Pokemon%dB.png" % epoch, normalize=True)
        torchvision.utils.save_image(gen_imgs[2].detach().cpu(), "images/Pokemon%dC.png" % epoch, normalize=True)
        torchvision.utils.save_image(gen_imgs[3].detach().cpu(), "images/Pokemon%dD.png" % epoch, normalize=True)
        #torchvision.utils.save_image(gen_imgs[4].detach().cpu(), "images/Pokemon%dE.png" % epoch, normalize=True)
        torchvision.utils.save_image(im, "images/Pokemon%dGrid.png" % epoch, normalize=True)
        gen_imgs = torch.transpose(gen_imgs, 1, 2)
        gen_imgs = torch.transpose(gen_imgs, 2, 3)
        generated_poke = gen_imgs[0].detach().cpu()
        pokemon = generated_poke.numpy()
        pokemon = (pokemon-pokemon.mean()) / pokemon.std()
        pokemon = (pokemon + 1) / 2 * 255
        pokemon = np.clip(pokemon, 0, 255).astype(np.uint8)
        cv2.imwrite("images/Pokemon%d.png" % epoch, pokemon)

