import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mat73
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import savgol_filter  # Import Savitzky-Golay filter
from scipy.io import savemat

torch.manual_seed(42)
np.random.seed(42)

# device = torch.device("cuda")
device = torch.device("cpu")

denoiserWeightsPath = 'denoiser_weights.pth'

mat = mat73.loadmat('Beam_ReducedFEModel.mat')
M = mat['MR_SEREP']
K = mat['KR_SEREP']
C = mat['CR_SEREP']
f = mat['g']


startTime = '0p011'
endTime = '2'
Red = '10'

# Load the data
mat = mat73.loadmat(f'NLDataForSIVA_Red={Red}_tStart={startTime}_tEnd={endTime}.mat')
disp, vel, acc = mat['DispTrain'], mat['VelTrain'], mat['AccTrain']
dispVal1, velVal1, accVal1 = mat['DispVal1'], mat['VelVal1'], mat['AccVal1']
dispVal2, velVal2, accVal2 = mat['DispVal2'], mat['VelVal2'], mat['AccVal2']
t = mat['t']


DOF = np.shape(disp,)[1]
numkNL = 5
numdNL = 0

# Convert initial data to tensors
dispTensor = torch.from_numpy(disp).float().to(device)
velTensor = torch.from_numpy(vel).float().to(device)
accTensor = torch.from_numpy(acc).float().to(device)
tTensor = torch.from_numpy(t).float().to(device)

dispTensorVal1 = torch.from_numpy(dispVal1).float().to(device)
velTensorVal1 = torch.from_numpy(velVal1).float().to(device)
accTensorVal1 = torch.from_numpy(accVal1).float().to(device)

dispTensorVal2 = torch.from_numpy(dispVal2).float().to(device)
velTensorVal2 = torch.from_numpy(velVal2).float().to(device)
accTensorVal2 = torch.from_numpy(accVal2).float().to(device)

Minv = torch.from_numpy(np.linalg.inv(M)).float().to(device)
Kt = torch.from_numpy(K).float().to(device)
Ct = torch.from_numpy(C).float().to(device)
ft = torch.from_numpy(f).float().to(device)


# Feature Extraction Network
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2*numkNL+2*numdNL)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = torch.mean(torch.squeeze(self.fc4(x)),0)
    
        
        dampPars = x[:2*numdNL]
        cnl = dampPars[:numdNL]*torch.pow(10,dampPars[numdNL:])
        
        stiffPars = x[2*numdNL:]
        knl = stiffPars[:numkNL]*torch.pow(10,stiffPars[numkNL:])
        
        return cnl, knl
    
class ConvDenoiser(nn.Module):
    def __init__(self, num_channels):
        super(ConvDenoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=num_channels, kernel_size=11, stride=2, padding=5, output_padding=1)
        )

    def forward(self, x):
        original_seq_len = x.shape[2]
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded[:, :, :original_seq_len]
        

# Generator (Physical Model)
class Generator(nn.Module):
    def __init__(self, Minv_mat, K_mat, C_mat):
        super(Generator, self).__init__()
        self.Minv = Minv_mat
        self.K = K_mat
        self.C = C_mat

        
    def forward(self, features, disp, vel):
        # Unpack features
        cnl = features[0]
        knl = features[1]
    
        Fsys = -torch.einsum("ij,jb->ib", self.K, disp)   
        Fsys -= torch.einsum("ij,jb->ib", self.C, vel)                                     
        
        Fnl = knl[0]*(disp[:,-1])+knl[1]*(disp[:,-1])**2+knl[2]*(disp[:,-1])**3+knl[3]*(disp[:,-1])**4 + \
                 knl[4]*(disp[:,-1])**5


        Fsys[..., -1] -= Fnl

        
        # Fnl2 = cnl[0]*vel[:,-1]*(disp[:,-1])**2
        
        # Fnoncons[:,-1] = Fnoncons[:,-1]+Fnl2

        
        accel = torch.einsum("ij,jb->ib", self.Minv, Fsys)

        
        return accel

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_size=DOF):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # Input size is 2 for 2-DOF
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def prepare_data(disp, vel, acc, force,
                 dispVal1, velVal1, accVal1, forceVal1,
                 dispVal2, velVal2, accVal2, forceVal2,
                 batch_size):
    # Create dataset
    dataset = TensorDataset(disp, vel, acc, force,
                            dispVal1, velVal1, accVal1, forceVal1, 
                            dispVal2, velVal2, accVal2, forceVal2)
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def pretrain_denoiser(noisy_disp, noisy_vel, epochs=200, lr=1e-3, batch_size=4):
    print("\n--- Starting Denoiser Pre-training ---")
    print(batch_size)
    # Create pseudo-clean targets using SG filter
    window_length, polyorder = 51, 3
    clean_disp = savgol_filter(noisy_disp, window_length, polyorder, axis=0)
    clean_vel = savgol_filter(noisy_vel, window_length, polyorder, axis=0)

    # Reshape for Conv1d: (batch, channels, sequence_length)
    # We treat each DOF as a "batch sample" for the DataLoader
    noisy_disp_T = torch.from_numpy(noisy_disp).float().permute(1, 0)
    noisy_vel_T = torch.from_numpy(noisy_vel).float().permute(1, 0)
    clean_disp_T = torch.from_numpy(clean_disp).float().permute(1, 0)
    clean_vel_T = torch.from_numpy(clean_vel).float().permute(1, 0)

    # Stack to create a 2-channel input: (DOF, 2, seq_len)
    noisy_multichannel = torch.stack([noisy_disp_T, noisy_vel_T], dim=1)
    clean_multichannel = torch.stack([clean_disp_T, clean_vel_T], dim=1)

    # DataLoader
    dataset = TensorDataset(noisy_multichannel, clean_multichannel)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    denoiser = ConvDenoiser(num_channels=2).to(device)
    optimizer = optim.Adam(denoiser.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    LDnStore = []
    for epoch in range(epochs):
        LDn = []
        for noisy_batch, clean_batch in dataloader:
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
            optimizer.zero_grad()
            denoised_output = denoiser(noisy_batch)
            loss = criterion(denoised_output, clean_batch)
            loss.backward()
            optimizer.step()
            LDn.append(loss.detach().item())
            
        LDnStore.append(np.mean(LDn))
        
        if (epoch + 1) % 1000 == 0:
            print(f"Denoiser Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}")
            with torch.no_grad():
                DN_tensor = denoiser(noisy_multichannel.to(device)) 

            DN = DN_tensor.cpu().numpy()
            fig, axs = plt.subplots(3, 1)
            axs[0].plot(LDnStore, label='Denoiser Loss')
            axs[0].set_yscale('log')
            axs[0].legend()
            axs[1].plot(t,noisy_multichannel.numpy()[-1,0,:],label='Real')
            axs[1].plot(t,DN[-1,0,:],label='Denoised',linestyle='--')
            axs[1].set_xlim(t[0],0.2)
            axs[1].legend()
            
            axs[2].plot(t,clean_multichannel.numpy()[-1,0,:],label='Clean')
            axs[2].plot(t,DN[-1,0,:],label='Denoised',linestyle='--')
            axs[2].set_xlim(t[0],0.2)
            axs[2].legend()
            
            
            plt.pause(0.1)
     

    torch.save(denoiser.state_dict(), denoiserWeightsPath)
    print(f"--- Denoiser training complete. Weights saved to '{denoiserWeightsPath}' ---\n")


def train(dataloader, num_epochs):
    
    # Initialize networks
    feature_extractor = FeatureExtractor().to(device)
    generator = Generator(Minv, Kt, Ct).to(device)
    discriminator = Discriminator(input_size=DOF).to(device)
    
    # Initialize and load pre-trained denoiser for the combined signals
    # The denoiser will process all DOFs at once, treating them as channels.
    denoiser = ConvDenoiser(num_channels=2).to(device)
    denoiser.load_state_dict(torch.load(denoiserWeightsPath))
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    
    # Set up optimizers with differential learning rates for fine-tuning
    lr_main = 1e-4
    lr_finetune = 1e-6 # Denoiser learns 100x slower
    optimizer_G = optim.Adam([
        {'params': feature_extractor.parameters(), 'lr': lr_main},
        {'params': generator.parameters(), 'lr': lr_main},
        {'params': denoiser.parameters(), 'lr': lr_finetune}
    ])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_main)
    
    LDStore, LGStore, LMStore = [], [], []
    
    print("--- Starting GAN Training with Fine-tuning ---")
    for epoch in range(num_epochs):
        # Set models to training mode
        feature_extractor.train()
        generator.train()
        discriminator.train()
        denoiser.train() # Allow denoiser to be fine-tuned
            
        LD, LG, LM = [], [], []
        print(epoch)
        for (batch_disp, batch_vel, batch_acc,
             batch_dispVal1, batch_velVal1, batch_accVal1,
             batch_dispVal2, batch_velVal2, batch_accVal2) in dataloader:


            batch_disp, batch_vel, batch_acc = batch_disp.to(device), batch_vel.to(device), batch_acc.to(device)
            batch_dispVal1, batch_velVal1, batch_accVal1 = batch_dispVal1.to(device), batch_velVal1.to(device), batch_accVal1.to(device)
            batch_dispVal2, batch_velVal2, batch_accVal2 = batch_dispVal2.to(device), batch_velVal2.to(device), batch_accVal2.to(device)
            
            current_batch_size = batch_disp.size(0)  # Get current batch size
            z = torch.randn(current_batch_size, 1, 1)  # ~9: vars to id

            # Extract features
            features = feature_extractor(z)
 
            # --- Denoise Data ---
            # Prepare multichannel input for the denoiser: (batch, channels, sequence_length)
            # Training data
            denoiser_input_train = torch.stack([batch_disp.permute(1,0), batch_vel.permute(1,0)], dim=1)
            denoised_train = denoiser(denoiser_input_train.permute(2,1,0)) # This needs careful shape management
            dispT = denoised_train[:, 0, :].permute(1,0)
            velT = denoised_train[:, 1, :].permute(1,0)
            
            # Validation data 1
            denoiser_input_val1 = torch.stack([batch_dispVal1.permute(1,0), batch_velVal1.permute(1,0)], dim=1)
            denoised_val1 = denoiser(denoiser_input_val1.permute(2,1,0))
            dispV1 = denoised_val1[:, 0, :].permute(1,0)
            velV1 = denoised_val1[:, 1, :].permute(1,0)
            
            # Validation data 2
            denoiser_input_val2 = torch.stack([batch_dispVal2.permute(1,0), batch_velVal2.permute(1,0)], dim=1)
            denoised_val2 = denoiser(denoiser_input_val2.permute(2,1,0))
            dispV2 = denoised_val2[:, 0, :].permute(1,0)
            velV2 = denoised_val2[:, 1, :].permute(1,0)
                
            # Generate fake acc
            fake_acc = generator(features, dispT, velT).T
            fake_accVal1 = generator(features, dispV1, velV1)
            fake_accVal2 = generator(features, dispV2, velV2)
            
            realD = torch.cat([batch_accVal1,batch_accVal2],dim=0)
            fakeD = torch.cat([fake_accVal1,fake_accVal2],dim=1).T

            cnl = features[0].detach().numpy()
            knl = features[1].detach().numpy()
            
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(realD.size(0), 1).to(device)
            fake_labels = torch.zeros(fakeD.size(0), 1).to(device)
 
            real_loss = criterion(discriminator(realD), real_labels)
            fake_loss = criterion(discriminator(fakeD.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator            
            optimizer_G.zero_grad()
            adversarial_loss = criterion(discriminator(fakeD), real_labels)
            mse_content_loss = mse_loss(batch_acc, fake_acc)
            g_loss = adversarial_loss + mse_content_loss

            g_loss.backward()
            optimizer_G.step()

            LD.append(d_loss.detach().item())
            LG.append(criterion(discriminator(fakeD.detach()), torch.ones(fakeD.size(0), 1)).item())
            LM.append(mse_loss(batch_acc,fake_acc).item())
        
        LDStore.append(np.mean(LD))
        LGStore.append(np.mean(LG))
        LMStore.append(np.mean(LM))
    
        
        z_acc = torch.randn(len(t), 1, 1)
        features = feature_extractor(z_acc)

        cnl = features[0].detach().numpy()
        knl = features[1].detach().numpy()
        
  
        print(f"Epoch [{epoch+1}/{num_epochs}], D_loss: {np.mean(LD):.4f}, G_loss: {np.mean(LG):.4f}, MSE_loss: {np.mean(LM):.4f}")
  
        # Print progress and plot results
        if epoch % 100 == 0:
                        

            
            # Plot results 
            z_acc = torch.randn(len(t), 1, 1)
            features = feature_extractor(z_acc)
            
            denoiser_input = torch.stack([dispTensor.permute(1,0), velTensor.permute(1,0)], dim=1)
            denoised_train = denoiser(denoiser_input.permute(2,1,0)) # This needs careful shape management
            dispT = denoised_train[:, 0, :].permute(1,0)
            velT = denoised_train[:, 1, :].permute(1,0)
            
            # Validation data 1
            denoiser_input_val1 = torch.stack([dispTensorVal1.permute(1,0), velTensorVal1.permute(1,0)], dim=1)
            denoised_val1 = denoiser(denoiser_input_val1.permute(2,1,0))
            dispV1 = denoised_val1[:, 0, :].permute(1,0)
            velV1 = denoised_val1[:, 1, :].permute(1,0)
            
            # Validation data 2
            denoiser_input_val1 = torch.stack([dispTensorVal2.permute(1,0), velTensorVal2.permute(1,0)], dim=1)
            denoised_val1 = denoiser(denoiser_input_val1.permute(2,1,0))
            dispV2 = denoised_val1[:, 0, :].permute(1,0)
            velV2 = denoised_val1[:, 1, :].permute(1,0)
                
            
            
            fake_acc = generator(features, dispT, velT).detach().numpy().T
            fake_accVal1 = generator(features, dispV1, velV1).detach().numpy().T
            fake_accVal2 = generator(features, dispV2, velV2).detach().numpy().T
            
            cnl = features[0].detach().numpy()
            knl = features[1].detach().numpy()
            
            # data = {'M': M,
            #         'K': K,
            #         'C': C,
            #         'knl': knl,
            #         'cnl': cnl
            #         }

            # savemat(f'NonlinearID_Red={Red}_tStart={startTime}_tEnd={endTime}.mat',data)
            
            
            # plt.close('all')
            fig, axs = plt.subplots(4, 2, figsize=(10, 10))
            axs[0,0].plot(LDStore, label='Discriminator Loss')
            axs[0,0].plot(LGStore, label='Generator Loss')
            axs[0,0].legend()
            
            axs[0,1].plot(LMStore, label='MSE Loss')
            axs[0,1].set_title('MSE Loss')
            axs[0,1].set_yscale('log')
            
            
            axs[1,0].plot(t,acc[:,0],label='Data - Train')
            axs[1,0].plot(t,fake_acc[:,0],label='Model - Train', linestyle='--')
            axs[1,0].set_xlim(t[0],t[-1])
            axs[1,0].legend()
            
            axs[1,1].plot(t,acc[:,-1],label='Data - Train')
            axs[1,1].plot(t,fake_acc[:,-1],label='Model - Train', linestyle='--')
            axs[1,1].set_xlim(t[0],t[-1])
            axs[1,1].legend()
            
            axs[2,0].plot(t,acc[:,0],label='Data - Train')
            axs[2,0].plot(t,fake_acc[:,0],label='Model - Train', linestyle='--')
            axs[2,0].set_xlim(t[0],0.2)
            axs[2,0].legend()
            
            axs[2,1].plot(t,acc[:,-1],label='Data - Train')
            axs[2,1].plot(t,fake_acc[:,-1],label='Model - Train', linestyle='--')
            axs[2,1].set_xlim(t[0],0.2)
            axs[2,1].legend()
            
            axs[3,0].plot(t,accVal1[:,-1],label='Data - Val 1')
            axs[3,0].plot(t,fake_accVal1[:,-1],label='Model - Val 1', linestyle='--')
            axs[3,0].set_xlim(t[0],t[-1])
            axs[3,0].legend()
            
            axs[3,1].plot(t,accVal2[:,-1],label='Data - Val 2')
            axs[3,1].plot(t,fake_accVal2[:,-1],label='Model - Val 2', linestyle='--')
            axs[3,1].set_xlim(t[0],t[-1])
            axs[3,1].legend()
            
            plt.pause(0.1)
            
# --- Execution ---
# 1. Pre-train the denoiser
pretrain_denoiser(disp, vel, epochs=20000, batch_size=1000) # Using training displacement and velocity

# 2. Create DataLoader for GAN training
batch_size = 1000
# Note: DataLoader will batch along the first dimension (time steps)
train_dataset = TensorDataset(dispTensor, velTensor, accTensor,
                              dispTensorVal1, velTensorVal1, accTensorVal1,
                              dispTensorVal2, velTensorVal2, accTensorVal2)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 3. Run the main GAN training
train(train_dataloader, num_epochs=100000)




# %%
# z = torch.randn(len(t), 1, 1, device=device)
# features = feature_extractor(z)
            
# features = feature_extractor(z)
# fake_acc = generator(features, tTensor, dispTensor, velTensor).detach().numpy()

# cnl = features[0].detach().cpu().numpy()
# knl = features[1].detach().cpu().numpy()

# print(knl)

# %%


# data = {'M': M,
#         'K': K,
#         'C': C,
#         'knl': knl
#         }

# savemat(f'NonlinearID_Red={Red}_tStart={startTime}_tEnd={endTime}.mat',data)

