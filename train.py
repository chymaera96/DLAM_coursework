import os
import numpy as np
import argparse
import torch
import gc
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


from util import load_ckp, save_ckp, load_augmentation_index, create_fp_dir
from sfnet.gpu_transformations import GPUTransformNeuralfp
from sfnet.data_sans_transforms import NeuralfpDataset
from sfnet.modules.simclr import SimCLR
from sfnet.modules.residual import SlowFastNetwork, ResidualUnit
from eval import eval_faiss
from test_fp import create_fp_db, create_dummy_db

# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")
# data_dir = os.path.join(root,"data/fma_8000")
# ir_dir = os.path.join(root,'data/augmentation_datasets/ir_filters')
# noise_dir = os.path.join(root,'data/augmentation_datasets/noise')

device = torch.device("cuda")


parser = argparse.ArgumentParser(description='Neuralfp Training')
parser.add_argument('--data_dir', default='', type=str,
                    help='Path to training data')
parser.add_argument('--ir_dir', default='', type=str,
                    help='Path to impulse response data (augmentation)')
parser.add_argument('--noise_dir', default='', type=str,
                    help='Path to background noise data (augmentation)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--sr', default=22050, type=int,
                    help='Sampling rate ')
parser.add_argument('--ckp', default='sfnet', type=str,
                    help='checkpoint_name')
parser.add_argument('--n_dummy_db', default=500, type=int)
parser.add_argument('--n_query_db', default=20, type=int)


def ntxent_loss(z_i, z_j, tau=0.05):
    """
    NTXent Loss function.
    Parameters
    ----------
    z_i : torch.tensor
        embedding of original samples (batch_size x emb_size)
    z_j : torch.tensor
        embedding of augmented samples (batch_size x emb_size)
    Returns
    -------
    loss
    """
    z = torch.stack((z_i,z_j), dim=1).view(2*z_i.shape[0], z_i.shape[1])
    a = torch.matmul(z, z.T)
    a /= tau
    Ls = []
    for i in range(z.shape[0]):
        nn_self = torch.cat([a[i,:i], a[i,i+1:]])
        softmax = torch.nn.functional.log_softmax(nn_self, dim=0)
        Ls.append(softmax[i if i%2 == 0 else i-1])
    Ls = torch.stack(Ls)
    
    loss = torch.sum(Ls) / -z.shape[0]
    return loss

def train(train_loader, model, optimizer, ir_idx, noise_idx, sr):
    loss_epoch = 0
    augment = GPUTransformNeuralfp(ir_dir=ir_idx, noise_dir=noise_idx, sample_rate=sr).to(device)
    for idx, (x_i, x_j) in enumerate(train_loader):

        # print(f"Inside train function x_i, x_j {x_i.shape} {x_j.shape}")
        # augment = GPUTransformNeuralfp(ir_dir=ir_idx, noise_dir=noise_idx, sample_rate=sr).to(device)
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        x_i, x_j = augment(x_i, x_j)
        # print(f"Output from augmenter x_i, x_j {x_i.shape} {x_j.shape}")

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        loss = ntxent_loss(z_i, z_j)

        # if torch.count_nonzero(torch.isnan(loss)) > 0:
        #     print(z_i)
        loss.backward()

        optimizer.step()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t Loss: {loss.item()}")
            del augment
            gc.collect()
            torch.cuda.empty_cache()
            augment = GPUTransformNeuralfp(ir_dir=ir_idx, noise_dir=noise_idx, sample_rate=sr).to(device)

        loss_epoch += loss.item()

    return loss_epoch

def validate(query_loader, dummy_loader, augment, model, output_root_dir):
    create_dummy_db(dummy_loader, augment=augment, model=model, output_root_dir=output_root_dir)
    create_fp_db(query_loader, augment=augment, model=model, output_root_dir=output_root_dir)
    hit_rates = eval_faiss(emb_dir=output_root_dir, test_ids='all')
    print("-------Validation hit-rates-------")
    print(f'Top-1 exact hit rate = {hit_rates[0]}')
    print(f'Top-1 near hit rate = {hit_rates[1]}')
    return hit_rates

def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    ir_dir = args.ir_dir
    noise_dir = args.noise_dir
    
    # Hyperparameters
    batch_size = 120
    learning_rate = 1e-4
    num_epochs = args.epochs
    sample_rate = args.sr
    model_name = args.ckp
    random_seed = 42
    shuffle_dataset = True

    print(ir_dir)
    print(noise_dir)
    # assert data_dir == os.path.join(root,"data/fma_8000")

    print("Loading dataset...")
    train_dataset = NeuralfpDataset(path=train_dir, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)
    
    valid_dataset = NeuralfpDataset(path=valid_dir, train=False)
    print("Creating validation dataloaders...")
    dataset_size = len(valid_dataset)
    indices = list(range(dataset_size))
    split1 = args.n_dummy_db
    split2 = args.n_query_db
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]

    dummy_db_sampler = SubsetRandomSampler(dummy_indices)
    query_db_sampler = SubsetRandomSampler(query_db_indices)

    dummy_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=dummy_db_sampler)
    
    query_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=query_db_sampler)

    
    print("Creating new model...")
    model = SimCLR(encoder=SlowFastNetwork(ResidualUnit, layers=[1,1,1,1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 500, eta_min = 3e-5)
    print("Intializing augmentation pipeline...")
    noise_idx = load_augmentation_index(noise_dir, splits=[0.6,0.2,0.2])["train"]
    ir_idx = load_augmentation_index(ir_dir, splits=[0.6,0.2,0.2])["train"]
    augment = GPUTransformNeuralfp(ir_dir=ir_idx, noise_dir=noise_idx, sample_rate=args.sr).to(device)
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log = load_ckp(args.resume, model, optimizer, scheduler)
            output_root_dir = create_fp_dir(resume=args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        loss_log = []
        hit_rate_log = []
        output_root_dir = create_fp_dir(ckp=args.ckp)


    print("Calculating initial loss ...")
    best_loss = train(train_loader, model, optimizer, ir_idx, noise_idx, args.sr)

    # training
    model.train()
    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(train_loader, model, optimizer, ir_idx, noise_idx, args.sr)
        hit_rates = validate(query_loader, dummy_loader, augment, model, output_root_dir)
        loss_log.append(loss_epoch)
        hit_rate_log.append(hit_rates[0])
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            
            checkpoint = {
                'epoch': epoch,
                'loss': loss_log,
                'valid_acc' : hit_rate_log,
                'hit_rate': hit_rates,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_ckp(checkpoint,epoch, model_name, model_folder)
        scheduler.step()
    
  
        
if __name__ == '__main__':
    main()