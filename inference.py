import os
import json
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from util import save_ckp



root = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Inferencing Network')
parser.add_argument('--ckp', default='', type=str)
parser.add_argument('--model_dir', default='', type=str)
parser.add_argument('--emb_path', default='', type=str)
parser.add_argument('--index_path', default='', type=str)
parser.add_argument('--emb_type', default='', type=str)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def get_num_encodings(index_file):
    label_encoder = LabelEncoder()
    with open(index_file, 'r') as fp:
        filenames = json.load(fp)
        labels = [f.split('/')[-1].split('.')[0] for f in filenames]
    return label_encoder.fit_transform(labels)



def train(loader, model, optimizer, criterion):
    loss_epoch = 0
    for batch_idx , [data, target] in enumerate(loader):
        data = data.to(device)
        target = target.to(device) 

        scores = model(data) 
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        if batch_idx % 4 == 0:
            print(f"Step [{batch_idx}/{len(loader)}] ------> train loss = {loss.item()}")

    return loss_epoch

def validate(model, loader, criterion):

  loss_epoch = 0
  model.eval()
  with torch.no_grad():
    for batch_idx , [data, target] in enumerate(loader):
        data = data.to(device)
        target = target.to(device) 
        scores = model(data)

        loss = criterion(scores, target)
        loss_epoch += loss.item()

        if batch_idx % 4 == 0:
            print(f"                                 ------> valid loss = {loss.item()}")

  return loss_epoch



class ClassificationDataset(Dataset):
    def __init__(self, index_path, emb_path):
        self.index_path = index_path
        # if framework == 'clmr':
        #     pass
        # elif framework == 'sfnet':
        #     pass
        
        self.embs = torch.load(emb_path)
        self.targets = get_num_encodings(index_path)
        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        try:
            data = self.embs[idx]
        except Exception:
            print("Error loading:" + self.embs[idx])
            self.ignore_idx.append(idx)
            return self[idx+1]
        
        return data, self.targets[idx]
    
    def __len__(self):
        return len(self.targets)



class LinearEvaluation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

def main():

    args = parser.parse_args()
    dataset = ClassificationDataset(index_path=args.index_path, emb_path = args.emb_path)
    batch_size = 32
    train_split = .7
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    n_epochs = 30

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split1 = int(np.floor(train_split * dataset_size))
    split2 = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:split1], indices[split1: split1 + split2], indices[split1 + split2:]

    dataset_split = {}
    dataset_split["train"] = train_indices
    dataset_split["val"] = val_indices
    dataset_split["test"] = test_indices

    with open('data_splits.json','w') as fp:
        json.dump(dataset_split, fp)


    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)



    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8,
        sampler=train_sampler
        )
    valid_loader = DataLoader(dataset, batch_size=batch_size,
        sampler=valid_sampler, 
        num_workers=8
        )
    
    if args.emb_type == 'sfnet':
        input_dim = 1984
    elif args.emb_type == 'clmr':
        input_dim = 5632
    model = LinearEvaluation(input_dim, 128, 10).to(device)
    criterion   = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    t_loss_log = []
    v_loss_log = []

    for epoch in range(1, n_epochs + 1):
        print(f"------ Epoch {epoch} ------")
        train_loss = train(loader=train_loader, model=model, optimizer=optimizer, criterion=criterion)
        valid_loss = validate(loader=valid_loader, model=model, criterion=criterion)

        if valid_loss < best_loss:
            best_loss = valid_loss   
            t_loss_log.append(train_loss)
            v_loss_log.append(valid_loss)
            checkpoint = {
                'epoch': epoch,
                'train_loss': t_loss_log,
                'valid_loss' : v_loss_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_ckp(checkpoint,epoch, args.ckp, args.model_dir)




if __name__ == '__main__':
    main()