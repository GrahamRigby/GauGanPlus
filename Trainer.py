from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from Preprocess import get_data
from Models import *
import cv2

def run(batch_size, img_size, training_epochs, save_interval, train, Load_Model_State):
    #---------------------------------------
    #Get Data Loader
    dataset = get_data(train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    #---------------------------------------
    #Initialize Model
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(img_size)
    generator.apply(weights_init_normal)
    if train:
        # Initialize generator and discriminator and loss functions
        bceLoss = nn.BCEWithLogitsLoss()
        multi_discriminator = MultiDiscriminator(img_size)
        multi_discriminator.apply(weights_init_normal)
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(multi_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.1)
        if cuda:
            multi_discriminator.cuda()
            bceLoss.cuda()
    if cuda:
        generator.cuda()
    if Load_Model_State:
        generator.load_state_dict(torch.load('generator_model.pt'))
        if train:
            multi_discriminator.load_state_dict(torch.load('multi_discriminator_model.pt'))

    #---------------------------------------
    # Run Eval
    if train==False:
        inputs, feature_maps, medium_maps, small_maps, tiny_maps, micro_maps = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
        inputs = inputs.to(device)
        gen_imgs, kld_loss = generator(inputs, feature_maps, medium_maps, small_maps, tiny_maps, micro_maps)
        torchvision.utils.save_image(gen_imgs[0].detach().cpu(), "images/Pokemon%dA.png", normalize=False)
        torchvision.utils.save_image(gen_imgs[1].detach().cpu(), "images/Pokemon%dB.png", normalize=False)
        torchvision.utils.save_image(gen_imgs[2].detach().cpu(), "images/Pokemon%dC.png", normalize=False)
        torchvision.utils.save_image(gen_imgs[3].detach().cpu(), "images/Pokemon%dD.png", normalize=False)
        torchvision.utils.save_image(gen_imgs[4].detach().cpu(), "images/Pokemon%dE.png", normalize=False)

    #---------------------------------------
    #Run Training
    if train:
        for epoch in range(training_epochs):
            for i, data in enumerate(loader, 0):
                # Final Data preparation stuff
                inputs, feature_maps, medium_maps, small_maps, tiny_maps, micro_maps = data
                gen_inputs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny, gen_maps_micro = \
                    inputs[:(len(inputs) // 2)].to(device), feature_maps[:(len(inputs) // 2)].to(device), \
                    medium_maps[:(len(inputs) // 2)].to(device), small_maps[:(len(inputs) // 2)].to(device), \
                    tiny_maps[:(len(inputs) // 2)].to(device), micro_maps[:(len(inputs) // 2)].to(device)
                disc_inputs, disc_maps_large, disc_maps_medium, disc_maps_small, disc_maps_tiny, disc_maps_micro = \
                    inputs[(len(inputs) // 2):].to(device), feature_maps[(len(inputs) // 2):].to(device), \
                    medium_maps[(len(inputs) // 2):].to(device), small_maps[(len(inputs) // 2):].to(device), \
                    tiny_maps[(len(inputs) // 2):].to(device), micro_maps[(len(inputs) // 2):].to(device)

                # noise injection
                gen_inputs = torch.stack([noiseer(x) for x in gen_inputs]).to(device)
                disc_inputs = torch.stack([noiseer(x) for x in disc_inputs]).to(device)

                # Discriminator step
                optimizer_D.zero_grad()
                d1_out, d2_out = multi_discriminator(disc_inputs, disc_maps_large, disc_maps_medium, disc_maps_small,
                                                     disc_maps_tiny, disc_maps_micro)
                loss1 = torch.mean(bceLoss(d1_out[0], Variable(torch.ones((d1_out[0].shape), device=device))))
                loss2 = torch.mean(bceLoss(d2_out[0], Variable(torch.ones((d2_out[0].shape), device=device))))
                D_loss_real = (loss1 +loss2)/2
                D_loss_real.backward()
                gen_imgs, kld_loss = generator(gen_inputs, gen_maps_large, gen_maps_medium, gen_maps_small, gen_maps_tiny,
                                               gen_maps_micro)
                d1_out, d2_out = multi_discriminator(gen_imgs.detach(), gen_maps_large, gen_maps_medium, gen_maps_small,
                                                     gen_maps_tiny, gen_maps_micro)
                loss1 = torch.mean(bceLoss(d1_out[0], Variable(torch.zeros((d1_out[0].shape), device=device))))
                loss2 = torch.mean(bceLoss(d2_out[0], Variable(torch.zeros((d2_out[0].shape), device=device))))
                D_loss_fake = (loss1 +loss2)/2
                D_loss_fake.backward()
                d_loss = (D_loss_fake + D_loss_real)
                optimizer_D.step()

                # Generator step
                optimizer_G.zero_grad()
                d1_out, d2_out = multi_discriminator(gen_imgs, gen_maps_large, gen_maps_medium, gen_maps_small,
                                                     gen_maps_tiny, gen_maps_micro)
                loss1 = torch.mean(bceLoss(d1_out[0], Variable(torch.ones((d1_out[0].shape), device=device))))
                loss2 = torch.mean(bceLoss(d2_out[0], Variable(torch.ones((d2_out[0].shape), device=device))))
                g_fake_loss = (loss1 +loss2)/2 + kld_loss
                g_fake_loss.backward()
                optimizer_G.step()

            if epoch == 50:
                schedulerG.step()

            if epoch % save_interval == 0 and epoch > 0:
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
                torchvision.utils.save_image(im, "images/Pokemon%dGrid.png" % epoch, normalize=False)
                gen_imgs = torch.transpose(gen_imgs, 1, 2)
                gen_imgs = torch.transpose(gen_imgs, 2, 3)
                generated_poke = gen_imgs[0].detach().cpu()
                pokemon = generated_poke.numpy()
                pokemon = (pokemon - pokemon.mean()) / pokemon.std()
                pokemon = (pokemon + 1) / 2 * 255
                pokemon = np.clip(pokemon, 0, 255).astype(np.uint8)
                cv2.imwrite("images/Pokemon%d.png" % epoch, pokemon)
    return "Complete"