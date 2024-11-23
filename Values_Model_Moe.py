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



        #Predicting Values

        values = self.tanh(self.reward_layer((torch.flatten(encoded, start_dim=1))))

        return output1_final ,  output2 , output3 ,encoded[:,-1,:], values, losses


from torch.utils.data import Dataset, DataLoader
import torch

class Dataset(Dataset):
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



    batch_size = 128  # Choose a batch size that fits your needs



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
            my_dataset = Dataset(data=features, targets=targets, mask= mask, targets_move = target_moves, values= target_values)
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





                    loss2_float = 1#loss2.item()


                    loss3 = criterion3(predicted_values, values) #values
                    # penalty = (1 - torch.abs(predicted_values).mean())
                    loss3_float = loss3.item()



                    output3 = output3.view(-1, 14)
                    target4 = data[:,1,:].reshape(-1)
                    loss4 = criterion4(output3,target4)

                    loss4_float = loss4.item()





                batch_count += 1


                if torch.isnan(loss3):
                        pass
                else:


                    loss3.backward(retain_graph=True)


                    for prob in losses:
                        if (prob[0].shape[0]) != 1:
                            load_loss = calculate_auxiliary_loss(prob[0])
                            z_loss = calculate_router_z_loss(prob[0])
                            total = load_loss + z_loss
                            total = total * 1e-4
                            total.backward(retain_graph=True)
                    
                    

                    torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
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
