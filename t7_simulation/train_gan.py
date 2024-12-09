import os
import glob
import numpy as np
import cv2
import torch
import torchvision
import scipy.io as sio
import lpips
import matplotlib.pyplot as plt
import argparse

import gan_helpers.dataloader as dl
import gan_helpers.gan as gan


def define_loss(device):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    def gen_loss(in1, in2): 
        return torch.mean(loss_fn_alex(in1, in2, normalize=True))
        
    return gen_loss, loss_fn_alex


def load_dataset(data_dir):
    ct = torchvision.transforms.Compose([dl.ToTensor()])

    dataset_list = []
    dataset_list_test = []

    i = 0
    while True:
        img_dir = os.path.join(data_dir, f"{i:03d}")
        if not os.path.exists(img_dir):
            break
        else:
            i += 1
        
        if i % 5 != 4:
            dataset_list.append(dl.GetSampleBatch(img_dir, ct, bsize=16))
        else:
            dataset_list_test.append(dl.GetSampleBatch(img_dir, ct, bsize=16))

    return dataset_list, dataset_list_test


def train(device, tap, data_dir, save_dir):
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    lambda_gp = 10
    num_epochs_gain = 5
    num_epochs = 5
    n_critic = 5

    g_losses = []
    d_losses = []
    lpips_losses = []

    gen_loss, loss_fn_alex = define_loss(device)

    dataset_list, dataset_list_test = load_dataset(data_dir)
    train_dataset = torch.utils.data.ConcatDataset(dataset_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = torch.utils.data.ConcatDataset(dataset_list_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    generator = gan.NoiseGenerator(device, data_dir, r=480, c=640, n=1)
    generator.cuda(device)
    optimizer_g = torch.optim.Adam([
            {"params": generator.g, "lr": 0.0002},
            {"params": generator.h, "lr": 0.0002},
            {"params": generator.shot, "lr": 0.0008},
            {"params": generator.read, "lr": 0.008},
            {"params": generator.row, "lr": 0.008},
            {"params": generator.rowt, "lr": 0.008},
            {"params": generator.quant, "lr": 0.008},
            {"params": generator.dark, "lr": 0.0002},
        ], lr=lr, betas=(b1, b2))
    discriminator = gan.Discriminator(n=1)
    discriminator.cuda(device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    kld_list = []

    for epoch in range(num_epochs_gain):
        print(tap + f" gain epoch {epoch} / {num_epochs_gain}")

        for i, sample in enumerate(train_loader):
            noisy = sample["noisy_imgs"][0].to(device)
            clean = sample["clean_imgs"][0].to(device)
            exp = sample["exp"].to(device)[0]

            # If last two dims are not 480 x 640, pad them
            if noisy.shape[-2:] != (480, 640):
                noisy = torch.nn.functional.pad(noisy, (0, 640 - noisy.shape[-1], 0, 480 - noisy.shape[-2]), mode="constant", value=0)
                clean = torch.nn.functional.pad(clean, (0, 640 - clean.shape[-1], 0, 480 - clean.shape[-2]), mode="constant", value=0)

            gen_noisy = generator(clean, exp, False)

            loss_l2 = torch.mean((noisy - gen_noisy) ** 2)

            optimizer_g.zero_grad()
            loss_l2.backward()
            optimizer_g.step()

    generator.g.requires_grad = False
    generator.h.requires_grad = False

    for epoch in range(num_epochs):
        print(tap + f" epoch {epoch} / {num_epochs}")

        for i, sample in enumerate(train_loader):
            noisy = sample["noisy_imgs"][0].to(device)
            clean = sample["clean_imgs"][0].to(device)
            exp = sample["exp"][0].to(device)

            if noisy.shape[-2:] != (480, 640):
                noisy = torch.nn.functional.pad(noisy, (0, 640 - noisy.shape[-1], 0, 480 - noisy.shape[-2]), mode="constant", value=0)
                clean = torch.nn.functional.pad(clean, (0, 640 - clean.shape[-1], 0, 480 - clean.shape[-2]), mode="constant", value=0)

            gen_noisy = generator(clean, exp, True)

            noisy = gan.split_into_patches(noisy / (2**12-1)).to(device)
            gen_noisy = gan.split_into_patches(gen_noisy / (2**12-1)).to(device)

            noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(noisy, norm="ortho")))
            gen_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

            real_validity = discriminator(noisy)
            fake_validity = discriminator(gen_noisy)

            gp = gan.compute_gradient_penalty(discriminator, noisy.data, gen_noisy.data)

            loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()

            if i % n_critic == 0:
                gen_noisy = generator(clean, exp, True)

                gen_noisy = gan.split_into_patches(gen_noisy / (2**12-1)).to(device)

                gen_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

                fake_validity = discriminator(gen_noisy)

                loss_lpips = gen_loss(noisy, gen_noisy)
                loss_g = -torch.mean(fake_validity) + loss_lpips

                loss_g.backward()
                optimizer_g.step()

            if i % 3 == 0:
                print(tap + f" epoch {epoch} / {num_epochs}, batch {i} / {len(train_loader)}, " + \
                        f"d loss {loss_d.item()}, g loss {loss_g.item()}, lpips loss {loss_lpips.item()}")
                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())
                lpips_losses.append(loss_lpips.item())

        klds = []

        for i, sample in enumerate(test_loader):
            with torch.no_grad():
                noisy = sample["noisy_imgs"][0].to(device)
                clean = sample["clean_imgs"][0].to(device)
                exp = sample["exp"][0].to(device)

                gen_noisy = generator(clean, exp, True)

                noisy = gan.split_into_patches(noisy / (2**12-1)).to(device)
                gen_noisy = gan.split_into_patches(gen_noisy / (2**12-1)).to(device)

                noisy_np = noisy.detach().cpu().numpy()
                gen_noisy_np = gen_noisy.detach().cpu().numpy()
                kld = gan.cal_kld(noisy_np, gen_noisy_np)
                klds.append(kld)

        avg_kld = sum(klds) / len(klds)
        kld_list.append(avg_kld)
        print(tap + f" average test kld {avg_kld}")

        with torch.no_grad():
            sample = test_dataset[(4 * len(test_dataset)) // 5]

            noisy = sample["noisy_imgs"].to(device)
            clean = sample["clean_imgs"].to(device)
            exp = sample["exp"].to(device)

            gen_noisy = generator(clean, exp, True)

            fix, ax = plt.subplots(1, 3, figsize=(16, 8))
            ax[0].imshow(clean.detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,0],
                    vmin=0, vmax=2**12-1, cmap="gray")
            ax[0].set_title("Clean Input")
            ax[1].imshow(gen_noisy.detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,0],
                    vmin=0, vmax=2**12-1, cmap="gray")
            ax[1].set_title("Simulated Noisy")
            ax[2].imshow(noisy.detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,0],
                    vmin=0, vmax=2**12-1, cmap="gray")
            ax[2].set_title("Real Noisy")
            plt.savefig(os.path.join(save_dir, f"epoch{epoch:03d}_kldavg{avg_kld:.6f}.png"), bbox_inches="tight")
            plt.clf()

    np.save(os.path.join(save_dir, "klds.npy"), np.array(kld_list))
    torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pt"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pt"))

    params = {}
    params[tap + "_fwc"] = generator.fwc.detach().cpu().numpy()
    params[tap + "_g"] = generator.g.detach().cpu().numpy()
    params[tap + "_h"] = generator.h.detach().cpu().numpy()
    params[tap + "_shot"] = generator.shot.detach().cpu().numpy()
    params[tap + "_read"] = generator.read.detach().cpu().numpy()
    params[tap + "_row"] = generator.row.detach().cpu().numpy()
    params[tap + "_rowt"] = generator.rowt.detach().cpu().numpy()
    params[tap + "_quant"] = generator.quant.detach().cpu().numpy()
    params[tap + "_dark"] = generator.dark.detach().cpu().numpy()
    sio.savemat(os.path.join(save_dir, tap + "_params.mat"), params)

    # Plot losses and save
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(lpips_losses, label="LPIPS Loss")
    plt.legend()
    # x-axis is number of iterations / 3
    plt.savefig(os.path.join(save_dir, "losses.png"), bbox_inches="tight")
    plt.clf()

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data_root",
            type=str,
            help="the root direrctory of the data",
            required=True
    )
    args = parser.parse_args()

    left_data = os.path.join(args.data_root, "left_data")
    left_save = os.path.join(args.data_root, "left_save")
    if not os.path.exists(left_save):
        os.makedirs(left_save)

    right_data = os.path.join(args.data_root, "right_data")
    right_save = os.path.join(args.data_root, "right_save")
    if not os.path.exists(right_save):
        os.makedirs(right_save)

    if not torch.cuda.is_available():
        print("no gpu available")
        exit()
    else:
        device = torch.device("cuda")

    left_params = train(device, "left", left_data, left_save)
    # left_params = sio.loadmat(os.path.join(left_save, "left_params.mat"))

    right_params = train(device, "right", right_data, right_save)
    # right_params = sio.loadmat(os.path.join(right_save, "right_params.mat"))
    
    params = {**left_params, **right_params}
    sio.savemat(os.path.join(args.data_root, "final_T7_params.mat"), params)
