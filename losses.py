import numpy as np
import torch
import torch.nn.functional as F

class Balanced_Loss(torch.nn.Module):
    def __init__(self, loss_type='cross_entropy', beta=0.999, fl_gamma=2, samples_per_class=None, class_balanced=True):
        super(Balanced_Loss, self).__init__()
        
        if class_balanced and samples_per_class is None:
            raise ValueError('blrblr')
        
        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced
        
    def forward(self, logits, labels):
        # criterion(pred, y)
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()
        
        else:
            weights = None
            
        if self.loss_type == 'cross_entropy':
            cb_loss = F.cross_entropy(input=logits, target=labels, weight=weights)
        else:
            print('??')
        return cb_loss