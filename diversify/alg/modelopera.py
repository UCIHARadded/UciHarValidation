import torch
from network import act_network

def get_fea(args):
    net = act_network.ActNetwork(args.dataset)
    args.featurizer_out_dim = net.in_features  # pass this to args
    return net

def accuracy(network, loader, weights, device='cuda'):
    """Compute accuracy with proper model state handling"""
    was_training = network.training
    network.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            labels = data[1].to(device).long()
            
            outputs = network.predict(inputs)
            
            if weights is None:
                batch_weights = torch.ones(len(inputs))
            else:
                batch_weights = weights
            
            batch_weights = batch_weights.to(device)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).float().mul(batch_weights).sum().item()
            total += batch_weights.sum().item()
    
    # Restore original training state
    if was_training:
        network.train()
    
    return correct / total if total > 0 else 0
