print('Started')

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from tqdm import tqdm
import time
# from st_moe_pytorch import MoE
import torch.nn.functional as F
from transformers import SwitchTransformersEncoderModel, SwitchTransformersConfig



#torch.flatten(tensor, start_dim=1)



def calculate_auxiliary_loss(router_probs):
    """
    Calculates the load balancing loss which encourages even distribution across experts.
    router_probs: Tensor of shape [batch_size, sequence_length, num_experts]
                  This contains the probability of each token being routed to each expert.
    """
    # Sum probabilities across the sequence length to get the load on each expert
    expert_load = router_probs.sum(dim=1)  # Shape: [batch_size, num_experts]

    # Calculate the mean load per expert (across the batch)
    mean_load = expert_load.mean(dim=0)  # Shape: [num_experts]

    # Compute MSE loss between each expert's load and the mean load
    load_balancing_loss = F.mse_loss(expert_load, mean_load.expand_as(expert_load))

    return load_balancing_loss

def calculate_router_z_loss(router_logits):
    """
    Calculates the router z-loss to minimize the confidence of routing decisions.
    router_logits: Tensor of shape [batch_size, sequence_length, num_experts]
                   Logits used to compute routing probabilities.
    """
    # Compute the log-sum-exp across experts for each token
    logsumexp_logits = torch.logsumexp(router_logits, dim=-1)  # Shape: [batch_size, sequence_length]

    # Calculate the mean squared value of these log-sum-exp values
    router_z_loss = (logsumexp_logits ** 2).mean()

    return router_z_loss


def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
    """Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import coral_loss
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> coral_loss(logits, levels)
    tensor(0.6920)
    """

    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    term1 = (F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss


def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.

    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels
                          + (F.logsigmoid(pred) - pred)*(1-train_labels))
        losses += loss

    return losses/num_examples


class CoralLoss(torch.nn.Module):
    """Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import CoralLoss
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> loss = CoralLoss()
    >>> loss(logits, levels)
    tensor(0.6920)
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, levels, importance_weights=None):
        """
        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.

        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).

        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.
        """
        return coral_loss(
            logits, levels,
            importance_weights=importance_weights,
            reduction=self.reduction)


class CornLoss(torch.nn.Module):
    """
    Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, y_train):
        """
        Parameters
        ----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
            Outputs of the CORN layer.

        y_train : torch.tensor, shape=(num_examples)
            Torch tensor containing the class labels.

        Returns
        ----------
        loss : torch.tensor
            A torch.tensor containing a single loss value.
        """
        return corn_loss(logits, y_train, num_classes=self.num_classes)


class Chess_Model_Actor(nn.Module):
    def __init__(self, dim = 2048, num_layers = 24, nheads = 4):
        super().__init__()
        self.position_matrix = torch.nn.Embedding(65, dim)
        self.piece_matrix = torch.nn.Embedding(14, dim)
        self.layers = num_layers
        self.nheads = nheads

        config = SwitchTransformersConfig(
            vocab_size = 0,
            d_model = dim, 
            num_layers = self.layers,
            num_heads= self.nheads,
            num_experts= 32
        )

        self.switch_encoder = SwitchTransformersEncoderModel(config)
        self.value_ff = nn.Linear(dim, dim)
        self.action_ff = nn.Linear(dim, dim)
        self.action_value_ff = nn.Linear(dim, dim)
        # self.layer_norm = nn.LayerNorm(dim)
        self.output_layer1 = nn.Linear(dim*65,1968)  # Output layer 1
        # self.output_layer2 = nn.Linear(dim*65,1968*63)  # Output layer 2
        self.output_layer3 = nn.Linear(dim,14)
        self.reward_layer = nn.Linear(dim*65,1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        # self.value_densenet = DenseNet1D(num_classes= 32)
        # self.action_densenet = DenseNet1D(num_classes= 4097)

    def forward(self, input1, input2, mask):
        positions = self.position_matrix(input1) # Embed input tokens using matrix 1
        pieces = self.piece_matrix(input2)  # Embed input tokens using matrix 2
        embedded = positions +  pieces   # Add the embeddings element-wise

        switch_encoder_outputs = self.switch_encoder( inputs_embeds = embedded, output_router_logits = True) # Pass added embeddings through Transformer encoder
        
        encoded = switch_encoder_outputs.last_hidden_state
        losses = switch_encoder_outputs.router_probs 
        # encoded = self.layer_norm(encoded)

        #Predicting best moves
        # output1_temp = self.action_ff(encoded)

        output1 = self.output_layer1(torch.flatten(encoded, start_dim=1))
        mask_unsqueezed = mask  < -1  #.unsqueeze(1)  # Now shape is (batch_size, 1, 4097)
        mask_repeated = mask_unsqueezed.repeat(1, 65, 1) < -1


        # output1 = output1 + mask # Apply output layer 1 to encoded features

        _MASKING_VALUE = -1e+30 if output1.dtype == torch.float32 else -1e+4

        output1_final = output1.masked_fill(mask_unsqueezed, _MASKING_VALUE)

        dense_for_value_forward = encoded[:,-1,:].clone()

        #Predicting Action Values
        # output2_temp = self.action_value_ff(encoded[:,-1,:])
        output2 = None #(self.output_layer2(torch.flatten(encoded, start_dim=1)))
        # action_values_output = output2.view(-1, 1968, 63)


        #Predicting Values
        output3 = self.output_layer3(encoded)
        # values_temp = self.value_ff(encoded[:,-1,:])
        # values_temp = nn.ELU()(values_temp)


        #Predicting Values
        # self.value_ff(encoded[:,-1,:])
        # print(encoded.shape)
        # print(encoded[:,-1,:].unsqueeze(1).shape)
        # values = self.value_densenet(encoded[:,-1,:].unsqueeze(1))
        values = self.tanh(self.reward_layer((torch.flatten(encoded, start_dim=1))))
        # values_post_tan = self.tanh(values)
        # values = nn.Sigmoid()(values_temp)
        return output1_final ,  output2 , output3 ,encoded[:,-1,:], values, losses


from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, data, targets, mask, targets_move, values):
        """
        Args:
            data (Tensor): A tensor containing the features of shape [m, 2, 65].
            targets (Tensor): A tensor containing the targets of shape [m, 65].
        """
        self.data = data
        self.targets = targets
        self.mask = mask
        self.targets_move = targets_move
        self.values = values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        mask = self.mask[idx]
        targets_move = self.targets_move[idx]
        values = self.values[idx]
        return sample, target, mask, targets_move,values

def convert_to_bin_indices(values, num_bins=32):
    # Ensure the values are in the range [0, 1)
    values = values.clamp(0, 1 - 1e-6)

    # Scale the values to the range [0, num_bins)
    scaled_values = values * num_bins

    # Convert to integer indices
    bin_indices = scaled_values.floor().long()

    return bin_indices

def apply_mask_to_bins(bin_indices, mask, ignored_index=32):
    # Create a mask where condition is True for values < 0 in the original mask tensor
    condition_mask = mask < 0

    # Use the condition mask to set corresponding locations in bin_indices to ignored_index
    bin_indices[condition_mask] = ignored_index

    return bin_indices


print('Classes Loaded')

if __name__ == '__main__':
    model = Chess_Model_Actor(dim = 512, num_layers= 8, nheads = 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load('/media/joey/Elements/Actor_medium_23.pth'))



    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")

    model.to(device)
    model.train()
    # model.half()



    # features = torch.load("/media/joey/Elements/trainingdata.pt")
    # features = features
    # targets = torch.load("/media/joey/Elements/trainingtargets.pt")
    # mask = torch.load("/media/joey/Elements/trainingdata_mask.pt")
    # target_moves = torch.load("/media/joey/Elements/trainingtargets_moves.pt")
    # target_values = torch.load("/media/joey/Elements/trainingtags.pt")

    # targets_flat = targets[:,-1]

    # class_counts = torch.zeros(4097)
    # for i in range(4096):
    #     class_counts[i] = (targets_flat == i).sum()

    # default_weight = 1.0  # This could be set to the mean of the non-zero weights, for example
    # inverse_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.tensor(default_weight))
    # normalized_weights = 4097 * inverse_weights / inverse_weights.sum()
    # normalized_weights = normalized_weights.to(device)

    # my_dataset = MyDataset(data=features, targets=targets, mask= mask, targets_move = target_moves, values= target_values)

    batch_size = 128  # Choose a batch size that fits your needs

    # data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)


    criterion = nn.CrossEntropyLoss(ignore_index=1968) #, weight= normalized_weights)
    criterion2 = CornLoss(64)#nn.CrossEntropyLoss(ignore_index=32) #pos_weight = normalized_weights)
    criterion3 = nn.MSELoss()#CornLoss(32)#nn.MSELoss() #nn.CrossEntropyLoss(ignore_index=32) #nn.MSELoss() #
    criterion4 = nn.CrossEntropyLoss(ignore_index=13)
    optimizer =  optim.AdamW(model.parameters(), lr = 1e-5,betas=(0.9, 0.95), eps = 1e-5, weight_decay= 1e-4)    #optim.SGD(model.parameters(), lr = 1e-3, momentum=.9)#optim.Adam(model.parameters(), lr = 5e-4,betas=(0.9, 0.999))  # Feel free to use any optimizer
#optim.Adagrad(model.parameters(), lr = 1e-3)



    for epoch in range(15,50): #119

            features = torch.load("/media/joey/Elements/data_values/trainingdata"+ str(epoch) + ".pt")
            targets = torch.load("/media/joey/Elements/data_values/trainingtargets"+ str(epoch) + ".pt")
            mask = torch.load("/media/joey/Elements/data_values/trainingdata_mask"+ str(epoch) + ".pt")
            target_moves = torch.load("/media/joey/Elements/data_values/trainingtargets_moves"+ str(epoch) + ".pt")
            target_values = torch.load("/media/joey/Elements/data_values/trainingtags"+ str(epoch) + ".pt")
            my_dataset = MyDataset(data=features, targets=targets, mask= mask, targets_move = target_moves, values= target_values)
            data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

            batch_count = 0

            grad_steps = 8
            # if epoch >  15:
            #     grad_steps = 16
            # else:
            #     grad_steps = 4


            progress_bar = tqdm(range(len(data_loader)))

            avg = []
            avg2 = []


            for batch_idx, (data, target, mask, targets_move, values) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)
                mask = mask.to(device)
                targets_move = targets_move.to(device)
                values = values.to(device)
                values = values.float()
                
                
                values = 2* (values-.5)



                # print(targets_move[0])
                # for thing in [target, mask, targets_move, values]:
                #      print(thing.shape)

                # bins = (convert_to_bin_indices(targets_move, num_bins= 64))
                # final_binned_masked_action_values = apply_mask_to_bins(bins, targets_move,ignored_index= 100)


                # final_values = convert_to_bin_indices(values, num_bins= 32)


                output1, output2, output3, dense, predicted_values, losses = model(data[:,0,:], data[:,1,:] , mask)

                # print(output1[:,-1,:].shape)
                # print(target[:,-1].shape)
                # model.close()

                if batch_size == 1:
                    output1 = output1.squeeze(0)
                    target = target.squeeze(0)
                    loss1 = criterion(output1, target)
                    # loss2 = criterion2(output2, targets_move.float())


                else:
                    # output1 = output1.view(-1, 4097)  # Flatten to [128*65, 4097]
                    # target = target[:,-1].view(-1)


                    # loss1 = criterion(output1, target)
                    loss1_temp = 1#loss1.item()


                    # output2 = output2.view(-1, 63)
                    # print(output2.shape)
                    # print(final_binned_masked_action_values.view(-1).shape)
                    # output2 = output2.view(-1, 8)
                    # mask = final_binned_masked_action_values.view(-1) < 80



                    # Applying the mask
                    # selected_logits = output2[mask]
                    # final_values_actions = final_binned_masked_action_values.view(-1)[mask]



                    # print(selected_logits.shape)
                    # print(final_values_actions.shape)
                    # final_binned_masked_action_values = final_binned_masked_action_values.view(-1)
                    # loss2 = criterion2(selected_logits, final_values_actions )
                    loss2_float = 1#loss2.item()


                    loss3 = criterion3(predicted_values, values) #values
                    # penalty = (1 - torch.abs(predicted_values).mean())
                    loss3_float = loss3.item()



                    output3 = output3.view(-1, 14)
                    target4 = data[:,1,:].reshape(-1)
                    loss4 = criterion4(output3,target4)

                    loss4_float = loss4.item()


                    # loss4 = .5* loss4
                    # adjusted_penalty = 1*penalty
                    # loss3 = loss3 + adjusted_penalty

                # total_loss = loss1  + loss2


                batch_count += 1


                if torch.isnan(loss3):
                        pass
                else:

                    # total_loss.backward()
                    # loss1.backward(retain_graph=True)
                    # loss2.backward(retain_graph=True)
                    loss3.backward(retain_graph=True)


                    for prob in losses:
                        if (prob[0].shape[0]) != 1:
                            load_loss = calculate_auxiliary_loss(prob[0])
                            z_loss = calculate_router_z_loss(prob[0])
                            total = load_loss + z_loss
                            total = total * 1e-4
                            total.backward(retain_graph=True)
                    
                    
                    # loss2.backward()
                    # torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                    avg.append(loss1_temp)
                    loss3_temp = loss3_float
                    avg2.append(loss3_float)

                    progress_bar.set_description("Epoch {0} Loss 1: {1:.5f} Loss 2: {2:.5f} Loss 3: {3:.5f} Loss 4: {4:.5f} Running Avg: {5:.4f} Running Avg2: {6:.4f}".format(epoch, loss1_temp, loss2_float, loss3_float,
                                loss4_float,
                                sum(avg[-100:])/len(avg[-100:]), sum(avg2[-100:])/len(avg2[-100:])))
                    progress_bar.update(1)

                    if batch_count % grad_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # if batch_count % 250 == 0:
                #     time.sleep(60)
                # if batch_count % 1000 == 0:
                #     torch.save(model.state_dict(), 'Actor_basic_'+str(epoch)+'.pth')


            torch.save(model.state_dict(), '/media/joey/Elements/Critic_medium_'+str(epoch)+'.pth')
    torch.save(model.state_dict(), '/media/joey/Elements/Critic_medium.pth')