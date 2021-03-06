import torch
import os
import io

def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(model, modality, name=''):
    # name = save_load_name(args, name)
    name = 'best_model_' + modality
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')


def load_model(modality, name=''):
    # name = save_load_name(args, name)
    name = 'best_model_' + modality
    with open(f'pre_trained_models/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t

def save_hidden(tensor, name=''):
    if not os.path.exists('hidden_vectors'):
        os.mkdir('hidden_vectors')
    torch.save(tensor, f'hidden_vectors/{name}.pt')


def load_hidden(modality=''):
    with open(f'hidden_vectors/{modality}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    H = torch.load(buffer)
    return H