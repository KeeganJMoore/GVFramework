import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mat73
import matplotlib.pyplot as plt
import torchode as to
from scipy.io import savemat

torch.manual_seed(42)
np.random.seed(42)

# device = torch.device("cuda")
device = torch.device("cpu")
mat = mat73.loadmat('Beam_ReducedFEModel.mat')
M = mat['MR_SEREP']
K = mat['KR_SEREP']
C = mat['CR_SEREP']
As = mat['As']
f = mat['g']


startTime = '0p011'
endTime = '2'
Red = '10'

# Load the data
mat = mat73.loadmat(f'NLDataForSIVA_Red={Red}_tStart={startTime}_tEnd={endTime}.mat')
dispVal1 = mat['DispVal1']
velVal1 = mat['VelVal1']
accVal1 = mat['AccVal1']

disp = mat['DispTrain']
vel = mat['VelTrain']
acc = mat['AccTrain']

dispVal2 = mat['DispVal2']
velVal2 = mat['VelVal2']
accVal2 = mat['AccVal2']

t = mat['t']

DOF = np.shape(disp,)[1]
numd = 2
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
Mt = torch.from_numpy(M).float().to(device)
Kt = torch.from_numpy(K).float().to(device)
Ct = torch.from_numpy(C).float().to(device)
ft = torch.from_numpy(f).float().to(device)
Ast = torch.from_numpy(As).float().to(device)

LDStore, LGStore, LMStore, LRStore = [], [], [], []

# --- NEW DATA PREPARATION: Create Non-Overlapping Segments ---

segment_length = 300


def create_non_overlapping_segments(disp, vel, acc, t, seg_len):
    """Chops time series data into non-overlapping segments."""
    n_samples = disp.shape[0]
    num_segments = n_samples // seg_len
    
    # Truncate to the largest number of full segments
    new_len = num_segments * seg_len
    disp = disp[:new_len, :]
    vel = vel[:new_len, :]
    acc = acc[:new_len, :]
    t = t[:new_len]
    
    # Reshape into (num_segments, segment_length, DOFs)
    disp_reshaped = disp.reshape(num_segments, seg_len, -1)
    vel_reshaped = vel.reshape(num_segments, seg_len, -1)
    acc_reshaped = acc.reshape(num_segments, seg_len, -1)
    
    # Extract initial conditions for each segment
    y0_disp = disp_reshaped[:, 0, :]
    y0_vel = vel_reshaped[:, 0, :]
    
    # The time span is the same for all segments
    t_span = t.reshape(num_segments, seg_len)[0, :]
    
    return acc_reshaped, y0_disp, y0_vel, t_span

# Process all datasets
target_acc_train, y0_disp_train, y0_vel_train, t_span = create_non_overlapping_segments(
    dispTensor, velTensor, accTensor, tTensor, segment_length)

target_acc_val1, y0_disp_val1, y0_vel_val1, _ = create_non_overlapping_segments(
    dispTensorVal1, velTensorVal1, accTensorVal1, tTensor, segment_length)

target_acc_val2, y0_disp_val2, y0_vel_val2, _ = create_non_overlapping_segments(
    dispTensorVal2, velTensorVal2, accTensorVal2, tTensor, segment_length)

# Combine validation data into a single set for training
target_acc_val = torch.cat([target_acc_val1, target_acc_val2], dim=0)
y0_disp_val = torch.cat([y0_disp_val1, y0_disp_val2], dim=0)
y0_vel_val = torch.cat([y0_vel_val1, y0_vel_val2], dim=0)

print(f"Created {y0_disp_train.shape[0]} training segments.")
print(f"Created {y0_disp_val.shape[0]} validation segments.")
print(f"Shape of initial conditions for ODE solver: {y0_disp_train.shape}")
print(f"ODE time span length: {t_span.shape[0]}")

# Feature Extraction Network
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2*numd+2*numkNL+2*numdNL)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = torch.squeeze(self.fc4(x))
        
        dampPars = x[:2*numd+2*numdNL]
        c = dampPars[:numd+numdNL]*torch.pow(10,dampPars[numd+numdNL:])
        cprop = torch.abs(c[:2*numd])
        cnl = c[2*numd:]
        
        stiffPars = x[2*numd+2*numdNL:]
        knl = stiffPars[:numkNL]*torch.pow(10,stiffPars[numkNL:])
        
        return cprop, cnl, knl

# Generator (Physical Model)
class Generator(nn.Module):
    def __init__(self, Minv_mat, M_mat, K_mat, C_mat, f_vec, dof):
        super(Generator, self).__init__()
        # We store the system matrices in the generator to avoid using global
        # variables.
        self.Minv = Minv_mat
        self.Mt = M_mat
        self.Ct = C_mat
        self.Kt = K_mat
        self.ft = f_vec
        self.DOF = dof

    def system(self, t, y, args):
        cprop, cnl, knl = args
        # y has shape (num_segments, 2 * self.DOF), e.g., (42, 20)
        
        numBatches = y.shape[0]

        # Extract displacements and velocities. Each of these is size (numBatches,DOF) because we
        # simulate each batch simultaneously. 
        
        d = y[..., :self.DOF]
        v = y[..., self.DOF:]
        d_last_dof = d[..., -1]

        # Compute the nonlinear force due to the flexure
        Fnl_scalar = (knl[0] * d_last_dof + knl[1] * d_last_dof**2 + knl[2] * d_last_dof**3 +
                      knl[3] * d_last_dof**4 + knl[4] * d_last_dof**5)

        # Create the total force vector:
        # 1. Copy a vector of zeros to create the correct shape and expand 
        #    it to match number of batches.
        F_total = self.ft.expand(numBatches, -1).clone()
        
        # 2. Subtract the nonlinear force vector, elastic forces from stiffness
        #    matrix, and the non-conservative forces from the damping matrix.
        #    
        #    We employ Einstein summation to multiply the stiffness matrix and 
        #    displacement matrix. This operation improves performance and 
        #    lowers memory usage. The string "ij,bj->bi" means that the first
        #    tensor has dimensions (i,j) while the second one has dimensions 
        #    (b,j), such that the multiplication results in a tensor of dimensions
        #    (b,i). This is equivalent to torch.matmul(K,d.T).T, but has better
        #    performance.
        F_total[..., -1] -= Fnl_scalar
        F_total -= torch.einsum("ij,bj->bi", self.Kt, d)
        F_total -= torch.einsum("ij,bj->bi", self.Ct+cprop[0]*self.Mt+cprop[1]*self.Kt, v)

        # Compute the accelerations using Einstein summation just like above.
        accel = torch.einsum("ij,bj->bi", self.Minv, F_total)
        
        # We return the state vector of velocities and accelerations
        return torch.cat([v, accel], dim=-1)

    def forward(self, features, t_span, disp_initial_batch, vel_initial_batch):
        cprop, cnl, knl = features
        numBatches = disp_initial_batch.size(0)

        # Assemble initial conditions using both displacements and velocities.
        y0 = torch.cat([disp_initial_batch, vel_initial_batch], dim=1)
        
        # Define the differential equation and arguments that we want to solve. 
        # to = torchode.
        term = to.ODETerm(self.system, with_args=True)
        args = (cprop, cnl, knl)

        # Prepare time-related inputs for the ode solver in batched format.
        t_eval_batched = t_span.unsqueeze(0).expand(numBatches, -1) # ODE is evaluated at these times
        t_start_batched = t_span[0].expand(numBatches) # Start time, same for every batch
        t_end_batched = t_span[-1].expand(numBatches) # End time, same for every batch
        
        # Assemble all pieces into a formal IVP object that the solver can understand
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval_batched, 
                                         t_start=t_start_batched, t_end=t_end_batched)

        # Select the numerical integration algorithm. 
        step_method = to.Tsit5(term=term) # Tsit5 is Tsistouras 5/4 method that is 
                                          # similar to Runge-Kutta methods like in ODE45
        
        # Create logic for adjusting time steps by defining tolerances.
        step_size_controller = to.IntegralController(atol=1e-5, rtol=1e-4, term=term)
        
        # Create the final solver object. AutoDiffAdjoint allows the solver to 
        # calculate graidents backwards through the entire integration process. 
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        
        # Run the simulation, then extract the state vector and separate into 
        # displacement and velocities
        solution = solver.solve(problem, args=args)
        y_sim_batched = solution.ys

        all_d = y_sim_batched[..., :self.DOF]
        all_v = y_sim_batched[..., self.DOF:]
        
        

        Fnl_scalar = (knl[0] * all_d[..., -1] + knl[1] * all_d[..., -1]**2 +
                      knl[2] * all_d[..., -1]**3 + knl[3] * all_d[..., -1]**4 +
                      knl[4] * all_d[..., -1]**5)

        Fnl_vec = torch.zeros_like(all_d)
        Fnl_vec[..., -1] = -Fnl_scalar
        K_d = torch.einsum("ij,btj->bti", self.Kt, all_d)
        C_v = torch.einsum("ij,btj->bti", self.Ct+cprop[0]*self.Mt+cprop[1]*self.Kt, all_v)

        F_total = self.ft.view(1, 1, -1) + Fnl_vec - K_d - C_v
        accel_gen = torch.einsum("ij,btj->bti", self.Minv, F_total)
        
        return accel_gen

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_size=DOF):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def train(num_epochs):
    for epoch in range(num_epochs):

        if epoch == 0:
            print(f"Epoch {epoch}/{num_epochs}")

        # --- Single Training Step on ALL segments ---
        
        # Generate one set of physical parameters for this epoch
        z = torch.randn(1, 1, 1, device=device)
        features = feature_extractor(z)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        
        # 1. Real Data is the full set of validation segments
        real_data = target_acc_val.view(-1, DOF)
        
        # 2. Fake Data is generated from all validation initial conditions
        with torch.no_grad():
            fake_data = generator(features, t_span, y0_disp_val, y0_vel_val).view(-1, DOF)
        
        real_loss = criterion(discriminator(real_data), torch.ones(real_data.size(0), 1, device=device))
        fake_loss = criterion(discriminator(fake_data), torch.zeros(fake_data.size(0), 1, device=device))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        
        # 1. GAN Loss: Try to fool the discriminator
        fake_data_for_g = generator(features, t_span, y0_disp_val, y0_vel_val).view(-1, DOF)
        g_loss_gan = criterion(discriminator(fake_data_for_g), torch.ones(fake_data_for_g.size(0), 1, device=device))

        # 2. MSE Loss: Match the ground truth on the TRAINING segments
        fake_acc_train = generator(features, t_span, y0_disp_train, y0_vel_train)
        mse = mse_loss(fake_acc_train, target_acc_train)
        
        g_loss = g_loss_gan + mse
        g_loss.backward()
        optimizer_G.step()
        
        # --- Logging and Plotting ---
        LDStore.append(d_loss.item())
        LGStore.append(g_loss_gan.item())
        LMStore.append(mse.item())
        
        print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {d_loss.item():.4f} | G Loss (GAN): {g_loss_gan.item():.4f} | G Loss (MSE): {mse.item():.4f}")

        if epoch % 5 == 0:
            cnl, knl = features
            cnl = cnl.detach().cpu().numpy()
            knl = knl.detach().cpu().numpy()
            print(knl)
            
            with torch.no_grad():
                # For plotting, we can still simulate the original, full-length data from t=0
                features_plot = (torch.from_numpy(cnl).float().to(device), torch.from_numpy(knl).float().to(device))
                full_sim_acc = generator(features_plot, tTensor, dispTensor[0].unsqueeze(0), velTensor[0].unsqueeze(0)).squeeze(0).cpu().numpy()
                full_sim_acc_val1 = generator(features_plot, tTensor, dispTensorVal1[0].unsqueeze(0), velTensorVal1[0].unsqueeze(0)).squeeze(0).cpu().numpy()
                full_sim_acc_val2 = generator(features_plot, tTensor, dispTensorVal2[0].unsqueeze(0), velTensorVal2[0].unsqueeze(0)).squeeze(0).cpu().numpy()

            data = {'M': M, 'K': K, 'C': C, 'knl': knl, 'cnl': cnl}
            savemat(f'NonlinearID_Red={Red}_tStart={startTime}_tEnd={endTime}.mat', data)
            
            plt.close('all')
            fig, axs = plt.subplots(4, 2, figsize=(10, 10))
            axs[0,0].plot(LDStore, label='D Loss'); axs[0,0].plot(LGStore, label='G Loss (GAN)'); axs[0,0].legend()
            axs[0,1].plot(LMStore, label='G Loss (MSE)'); axs[0,1].set_title('MSE Loss'); axs[0,1].set_yscale('log'); axs[0,1].legend()
            axs[1,0].plot(t, acc[:,0], label='Data'); axs[1,0].plot(t, full_sim_acc[:,0], label='Model', linestyle='--'); axs[1,0].set_xlim(t[0], t[-1]); axs[1,0].legend()
            axs[1,1].plot(t, acc[:,-1], label='Data'); axs[1,1].plot(t, full_sim_acc[:,-1], label='Model', linestyle='--'); axs[1,1].set_xlim(t[0], t[-1]); axs[1,1].legend()
            axs[2,0].plot(t, acc[:,0], label='Data'); axs[2,0].plot(t, full_sim_acc[:,0], label='Model', linestyle='--'); axs[2,0].set_xlim(t[0], 0.2); axs[2,0].legend()
            axs[2,1].plot(t, acc[:,-1], label='Data'); axs[2,1].plot(t, full_sim_acc[:,-1], label='Model', linestyle='--'); axs[2,1].set_xlim(t[0], 0.2); axs[2,1].legend()
            axs[3,0].plot(t, accVal1[:,-1], label='Val 1'); axs[3,0].plot(t, full_sim_acc_val1[:,-1], label='Model', linestyle='--'); axs[3,0].set_xlim(t[0], t[-1]); axs[3,0].legend()
            axs[3,1].plot(t, accVal2[:,-1], label='Val 2'); axs[3,1].plot(t, full_sim_acc_val2[:,-1], label='Model', linestyle='--'); axs[3,1].set_xlim(t[0], t[-1]); axs[3,1].legend()
            plt.tight_layout(); plt.pause(0.1)

# Initialize networks and optimizers
feature_extractor = FeatureExtractor().to(device)
generator = Generator(Minv_mat=Minv, M_mat = Mt, K_mat=Kt, C_mat=Ct, f_vec=ft, dof=DOF).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
mse_loss = nn.MSELoss()
optimizer_G = optim.Adam(feature_extractor.parameters(), lr=0.0001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

# --- Training ---
train(num_epochs=10000)

# %%
z = torch.randn(len(t), 1, 1, device=device)
features = feature_extractor(z)
            
features = feature_extractor(z)
fake_acc = generator(features, tTensor, dispTensor, velTensor).detach().numpy()

cnl = features[0].detach().cpu().numpy()
knl = features[1].detach().cpu().numpy()

print(knl)

# %%


data = {'M': M,
        'K': K,
        'C': C,
        'knl': knl
        }

savemat(f'NonlinearID_Red={Red}_tStart={startTime}_tEnd={endTime}.mat',data)

