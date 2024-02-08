# imports
import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
import torch_geometric
import subprocess
import sys
from torch_geometric.datasets import ShapeNet
import plotly.express  as px
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from pytorch_metric_learning.losses import NTXentLoss
from sklearn.manifold import TSNE

global cat_list

def install(version=torch.__version__, package=None):
    """
    Args: version: version of torch
          package: package to install
    This function will speed up the installation of torch_geometric, torch-sparse, torch-scatter, torch-cluster
    since we are also passing in the version of torch to install the correct version of the package
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package,"-f",version])

def dataset(size=5000, categories=None) -> ShapeNet:
    """
    Args: categories: list of categories to download
    Returns: ShapeNet dataset
    
    """
    global cat_list
    dataset = ShapeNet(root = "data/ShapeNet", categories = categories).shuffle()[:size]
    cat_list = [key for key in dataset.categories]
    return dataset

def map_category(category: int) -> str:
    category_str = cat_list[category]
    return category_str

def plot3d(data, eval= False, x = None,y=None,val=None,ax=None) -> None:
    """
    Args: data: torch_geometric.data.Data
    Returns: None
    This function will plot the 3D data
    if eval is True, it will also plot the labels
    """
    if not eval:
        title = map_category(data.category)    
        fig = px.scatter_3d(x= data.pos[:,0],y = data.pos[:,1],z=data.pos[:,2],opacity=0.3,title= title)
        fig.show()
    if eval:
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(int(point['val'])))

def dataloader(dataset, batch_size=16, shuffle=True) -> DataLoader:

    """
    Args: dataset: torch_geometric.data.Dataset
          batch_size: int
          shuffle: bool
    Returns: torch_geometric.data.DataLoader
    This function will return a DataLoader object
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def augmentations(data) -> torch_geometric.data.Data:
    """
    Args: data: torch_geometric.data.Data
    Returns: torch_geometric.data.Data
    This function will return the augmented data
    """
    augmentation = T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])
    return augmentation(data)

class Model(torch.nn.Module):
    def __init__(self, k = 20, aggr = 'max'):
        """
        Args: k: int
              aggr: str
        Note: k a hyperparameter, it is the number of nearest neighbors to consider. 
              k is very small for training purposes
        """
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2*3 , 64, 64]),k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]),k, aggr)
        self.lin1 = Linear(128+64, 128)
        self.mlp = MLP([128, 256, 32], norm=None)

    def forward(self, data,train =True):
        if train:
            aug_1 = augmentations(data)
            aug_2 = augmentations(data)

            pos_1, batch_1 = aug_1.pos, aug_1.batch
            pos_2, batch_2 = aug_2.pos, aug_2.batch
            
            x1 = self.conv1(pos_1, batch_1)
            x2 = self.conv2(x1, batch_1)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            x1 = self.conv1(pos_2, batch_2)
            x2 = self.conv2(x1, batch_2)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))
            
            # Global representation
            h_1 = global_max_pool(h_points_1, batch_1)
            h_2 = global_max_pool(h_points_2, batch_2)
        else:
            x1 = self.conv1(data.pos, data.batch)
            x2 = self.conv2(x1, data.batch)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, data.batch)
        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        return h_1, h_2, compact_h_1, compact_h_2

def train():
    loss_func = NTXentLoss(temperature=0.10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(dataloader(dataset(5000)))):
        data = data.to(device)
        optimizer.zero_grad()
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        embeddings = torch.cat((compact_h_1, compact_h_2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() + data.num_graphs

    return total_loss/len(data)

def label_points(x, y, val, ax)->None:
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(int(point['val'])))

def test():
    test_data = dataset(5000)
    sample = next(iter(dataloader(test_data)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    # Get representations
    h = model.forward(sample.to(device), train=Falsec)
    h = h.cpu().detach()
    labels = sample.category.cpu().detach().numpy()

    # Get low-dimensional t-SNE Embeddings
    h_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(h.numpy())

    # Plot
    ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels, 
                        alpha=0.5, palette="tab10")

    # Add labels to be able to identify the data points
    annotations = list(range(len(h_embedded[:,0])))

    plot3d(sample,pd.Series(h_embedded[:,0]), 
                pd.Series(h_embedded[:,1]), 
                pd.Series(annotations), 
                plt.gca()) 



def main():
    # Uncomment the following lines to install the required packages 
    
    # install(version = torch.__version__,package="torch_geometric")
    # install(version = torch.__version__,package="torch-sparse")
    # install(version = torch.__version__,package="torch-scatter")
    # install(version = torch.__version__,package="torch-cluster")

    data = dataset(5000)
    epochs = 10
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # for epoch in range(epochs):
    #     loss = train()
    #     print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    #     scheduler.step()
    test()

if __name__=="__main__":
    main()