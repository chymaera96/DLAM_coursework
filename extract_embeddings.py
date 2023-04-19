import torch
import torch.nn.functional as F
import argparse
import json
import os
from sfnet.data import NeuralfpDataset
from sfnet.modules.simclr import SimCLR
from sfnet.modules.residual import SlowFastNetwork, ResidualUnit



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

parser = argparse.ArgumentParser(description='Extract SFNet embeddings')
parser.add_argument('--ckp', default='', type=str)
parser.add_argument('--test_dir', default='', type=str)

def extract(dataloader, model):
    embs_per_file = []
    for idx, (db,q) in enumerate(dataloader):
        splits = zip(db[0], q[0])
        emb = []
        for x_i, x_j in splits:
            x_i = torch.unsqueeze(x_i,0).to(device)

            with torch.no_grad():
                _, _, z_i, _= model(x_i,x_i)
                # z_i = z_i.detach()

            emb.append(z_i)

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
        emb = torch.flatten(torch.cat(emb))
        if emb.shape[-1] < 1856:
            emb = F.pad(emb, (1856 - emb.size(-1), 0))
        elif emb.shape[-1] > 1856:
            emb = emb[:1856]
        embs_per_file.append(emb)
        torch.save(torch.stack(embs_per_file), 'sfnet_embeddings.pt')

    return

def main():
    args = parser.parse_args()
    dataset = NeuralfpDataset(path=args.test_dir, train=False)
    model = SimCLR(encoder=SlowFastNetwork(ResidualUnit, layers=[1,1,1,1])).to(device)

    assert os.path.isfile(args.ckp)
    print("=> loading checkpoint '{}'".format(args.ckp))
    checkpoint = torch.load(args.ckp)
    model.load_state_dict(checkpoint['state_dict'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False)
    
    extract(dataloader, model)


if __name__ == '__main__':
    main()