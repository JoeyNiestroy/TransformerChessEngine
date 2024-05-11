print('Started')

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from tqdm import tqdm
import time
import torch.nn.functional as F
from st_moe_pytorch import MoE



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



class Chess_Model_Actor(nn.Module):
    def __init__(self, dim = 2048, num_layers = 24, nheads = 4):
        super().__init__()
        self.position_matrix = torch.nn.Embedding(65, dim)
        self.piece_matrix = torch.nn.Embedding(14, dim)
        self.layers = num_layers
        self.nheads = nheads
        self.transformer_encoder_layer = TransformerEncoderLayer( d_model=dim, nhead=self.nheads, dim_feedforward= 4*dim, dropout=0.1, activation= 'relu', batch_first=True,  norm_first = True)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers= self.layers, norm = RMSNorm(dim) )

        self.value_ff = nn.Linear(dim, dim)
        self.action_ff = nn.Linear(dim, dim)
        self.action_value_ff = nn.Linear(dim, dim)

        self.output_layer1 = nn.Linear(dim*65,1968)  # Output layer 1

        self.output_layer3 = nn.Linear(dim,14)
        self.reward_layer = nn.Linear(dim*65,1)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()



    def forward(self, input1, input2, mask):
        positions = self.position_matrix(input1) # Embed input tokens using matrix 1
        pieces = self.piece_matrix(input2)  # Embed input tokens using matrix 2
        embedded = positions +  pieces   # Add the embeddings element-wise

        encoded = self.transformer_encoder(embedded) # Pass added embeddings through Transformer encoder

        





        output1 = self.output_layer1(torch.flatten(encoded , start_dim=1)) #torch.flatten(encoded, start_dim=1)
        mask_unsqueezed = mask < -1


        

        _MASKING_VALUE = -1e+30 if output1.dtype == torch.float32 else -1e+4

        output1_final = output1.masked_fill(mask_unsqueezed, _MASKING_VALUE)

        dense_for_value_forward = encoded[:,-1,:].clone()


        action_values_output = None



        output3 = self.output_layer3(encoded)




        values = self.tanh(self.reward_layer(torch.flatten(encoded , start_dim=1)))
        
        return output1_final , action_values_output  , output3 ,encoded[:,-1,:], values


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
    model = Chess_Model_Actor(dim = 256, num_layers = 8, nheads = 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load('/media/joey/Elements/Actor_basic_small_22.pth'))





    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")

    model.to(device)
    model.train()


    batch_size = 128 

    # data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)


    criterion = nn.CrossEntropyLoss(ignore_index=1968) #, weight= normalized_weights)
    criterion2 = CornLoss(num_classes=64)#nn.CrossEntropyLoss(ignore_index=32) #pos_weight = normalized_weights)
    criterion3 = nn.MSELoss()#CornLoss(num_classes=128)#nn.CrossEntropyLoss(ignore_index=1968) #nn.MSELoss() #
    criterion4 = nn.CrossEntropyLoss(ignore_index=13)
    optimizer =  optim.AdamW(model.parameters(), lr = 1e-4,betas=(0.9, 0.999), weight_decay= 1e-2) 



    for epoch in range(15,27): #119

            features = torch.load("/media/joey/Elements/Data_Fen/trainingdata"+ str(epoch) + ".pt")
            targets = torch.load("/media/joey/Elements/Data_Fen/trainingtargets"+ str(epoch) + ".pt")
            mask = torch.load("/media/joey/Elements/Data_Fen/trainingdata_mask"+ str(epoch) + ".pt")
            target_moves = torch.load("/media/joey/Elements/Data_Fen/trainingtargets_moves"+ str(epoch) + ".pt")
            target_values = torch.load("/media/joey/Elements/Data_Fen/trainingtags"+ str(epoch) + ".pt")
            my_dataset = Dataset(data=features, targets=targets, mask= mask, targets_move = target_moves, values= target_values)
            data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

            batch_count = 0

            grad_steps = 16


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
                # final_binned_masked_action_values = apply_mask_to_bins(bins, targets_move, ignored_index= 100)

                final_values = convert_to_bin_indices(values, num_bins= 128)


                output1, output2, output3, dense, predicted_values = model(data[:,0,:], data[:,1,:] , mask)



                if batch_size == 1:
                    output1 = output1.squeeze(0)
                    target = target.squeeze(0)
                    loss1 = criterion(output1, target)
                    # loss2 = criterion2(output2, targets_move.float())


                else:
                    target = target[:,-1]



                    loss1 = criterion(output1, target.view(-1))
                    loss1_temp = loss1.item()



                    loss2_float = 1




                    loss3 = criterion3(predicted_values, values) #final_values.view(-1)
                    loss3_float = loss3.item()



                    output3 = output3.view(-1, 14)
                    target4 = data[:,1,:].reshape(-1)
                    loss4 = criterion4(output3,target4)

                    loss4_float = loss4.item()





                batch_count += 1


                if torch.isnan(loss1):
                        pass
                else:


                    loss1.backward()




                    avg.append(loss1_temp)
                    loss3_temp = loss3_float
                    avg2.append(loss3_float)
                    progress_bar.set_description("Epoch {0} Loss 1: {1:.5f} Loss 2: {2:.5f} Loss 3: {3:.5f} Loss 4: {4:.5f} Running Avg: {5:.4f} Running Avg2: {6:.4f}".format(epoch, loss1_temp, loss2_float, loss3_float,
                                loss4_float,
                                sum(avg[-100:])/len(avg[-100:]), sum(avg2[-100:])/len(avg2[-100:])))
                    progress_bar.update(1)

                    if batch_count % grad_steps == 0 and epoch != 26:
                        optimizer.step()
                        optimizer.zero_grad()


            torch.save(model.state_dict(), '/media/joey/Elements/Actor_basic_medium_'+str(epoch)+'.pth')
    torch.save(model.state_dict(), '/media/joey/Elements/Actor_basic_medium.pth')
