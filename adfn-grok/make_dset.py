import torch
import einops 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_dataset(p):
    
    """## Define Task
    * Define modular addition
    * Define the dataset & labels

    Input format:
    |a|b|=|
    """

    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)

    dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % p

    return dataset, labels