import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

#Run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Declare the NN
class PINNsolver(nn.Module):
    def __init__(self):
        super(PINNsolver, self).__init__()
        self.fc1 = nn.Linear(2, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)
        self.fc4 = nn.Linear(80, 80)
        self.fc5 = nn.Linear(80, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x

#Beam Parameters
w = 0.0254 * 1.50
thickness = 0.0254 * np.mean([0.253, 0.252, 0.256, 0.252, 0.255]) 
I = w * thickness**3 / 12 
Area = w * thickness       
rho = 3.475 / (Area * 0.0254 * 72.5)  
rhoA = rho * Area  
L = 0.0254 * 24
Fexp = np.array([11.7333, 75.35, 212.683, 413.233, 704.7])
betas = np.array([1.875, 4.694, 7.855, 10.996])
beta = betas[0]
f = Fexp[0]
E = ((2 * np.pi * f)**2 * rhoA * L**4) / (beta**4 * I)
E = E * (11.6833 / 10.735)**2  


tMax = 40.0
F0 = 800
k1 = 0
k3 = 0

AccPos_in = [0, 2.4, 4.8, 7.2, 9.6, 12, 14.4, 16.8, 19.2, 21.6, 24]

def physics_loss(model, x, t, E, I, rhoA, F0, sigma_t = 0.01, w0=1, wend=1):

    nodes = x.shape[0]

    t_repeat = t.repeat(1, nodes).view(-1,1).requires_grad_(True).to(device)
    x_repeat = x.repeat(t.shape[0], 1).requires_grad_(True).to(device)
    
    u_pred = model(torch.cat([t_repeat, x_repeat], dim=1))
    # Derivatives
    u_t, u_x = torch.autograd.grad(u_pred, [t_repeat, x_repeat],
                                   grad_outputs=torch.ones_like(u_pred),
                                   create_graph=True)

    u_tt = torch.autograd.grad(u_t, t_repeat,
                               grad_outputs=torch.ones_like(u_t),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_repeat,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x_repeat,
                                grad_outputs=torch.ones_like(u_xx),
                                create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx, x_repeat,
                                 grad_outputs=torch.ones_like(u_xxx),
                                 create_graph=True)[0]

    residual = rhoA * u_tt + E * I * u_xxxx
    L_phys = torch.mean(residual**2)

    # Initial conditions loss
    # First node should be fixed
    x_first = x[0].view(1,1).to(device)         
    t_first = t.view(-1,1).to(device)      

    inputs_first = torch.cat([t_first, x_first.repeat(t.shape[0],1)], dim=1).to(device)

    u_first_pred = model(inputs_first)

    L_bc_first = torch.mean(u_first_pred**2)

    # Impulse force at tip of beam
    x_last = x[-1].view(1,1)         
    t_ic = torch.zeros_like(x_last).requires_grad_(True).to(device)  
    inputs_last = torch.cat([t_ic, x_last], dim=1).requires_grad_(True).to(device)
    u_last_pred = model(inputs_last)
    u_last_t_pred = torch.autograd.grad(u_last_pred, t_ic,
                                    grad_outputs=torch.ones_like(u_last_pred),
                                    create_graph=True)[0]

    # Treat impulse like an initial velocity
    m_tip = rhoA * L / (nodes - 1)
    v0 = (F0 * sigma_t)/ m_tip
    v0 = torch.tensor([[v0]], dtype=torch.float64, device=device)
    L_bc_last = torch.mean((u_last_t_pred - v0)**2)

    return L_phys + w0 * L_bc_first + wend * L_bc_last


#Training Loop
tMax = 15
timePoints = 50000
batch_size = 500
x = torch.tensor(np.linspace(0,L,12),dtype=torch.float64,device=device).view(-1,1)
t = torch.tensor(np.linspace(0,tMax,timePoints),dtype=torch.float64,device=device).view(-1,1)
start_time = time.time()

epochs = 1000
lr = 1e-4

model = PINNsolver().to(device).double()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_loss = float("inf")
best_model_state = None
loss_history = []
updateDIH = 0
for epoch in range(epochs):
    perm = torch.randperm(timePoints)  # shuffle time points
    t_shuffled = t[perm]
    epoch_start = time.time()
    epoch_loss = 0.0
    for i in range(0, timePoints, batch_size):
        t_batch = t_shuffled[i:i+batch_size]

        optimizer.zero_grad()
        # Compute physics loss for this batch
        L_phys_batch = physics_loss(model, x, t_batch, E, I, rhoA, F0)
        L_phys_batch.backward()
        optimizer.step()

        epoch_loss += L_phys_batch.item() * t_batch.shape[0]  # weighted sum
    epoch_loss /= timePoints
    loss_history.append(epoch_loss)
    
    if epoch_loss < best_loss:
        updateDIH = epoch
        best_loss = epoch_loss
        best_model_state = model.state_dict().copy()
    if (epoch - updateDIH) > 20:
        break
    elapsed_epoch = time.time() - epoch_start
    total_elapsed = time.time() - start_time

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, "
          f"Epoch Time: {elapsed_epoch:.2f}s, Total Time: {total_elapsed:.2f}s")

torch.save(best_model_state, "best_PINN_model.pth")
print(f"Best model saved with loss = {best_loss:.6f}")