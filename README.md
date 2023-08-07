# GNNLR
This is the code file for the research paper Neural-Symbolic Recommendation Model with
Graph-Enhanced Information.
environment
## Example to run the codes
```
> cd \src  
> conda env create -f environment.yml
> sh GNNLR.sh  
```
You may find problems with the environment installed using the above commands, the main package used in this project is pytorch2.0.0+cu118 with the corresponding torch_geometric. I would suggest you to install pytorch first and then go to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html to install torch_geometric. Finally install the rest of the packages.  
Two datasets are currently available: GiftCard and Luxury.  
When you need to change the dataset, modify the dataset parameter in the .sh file.  
More details of the dataset and runs will be provided after the paper is accepted!

