# %%
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# %%
# x = [p,v,a]
# u = [a]
# y = [p]

A = [[0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]]
B = [[0],
     [0],
     [0]]
C = [[1, 0, 0],
     [0, 0, 1]]
D = [[0], [0]]

sys = ct.ss(A, B, C, D)
obs = ct.obsv(sys.A, sys.C)
print("Obs:", np.linalg.matrix_rank(obs))

comb = np.concatenate([A, B], axis=1)
comb = np.concatenate([comb, np.zeros((1, 4))], axis=0)

# generate trajectory p(t)
end = 10
samples = end*200
t = np.linspace(0, end, samples)
p = np.sin(t)
v = np.cos(t)
a = -np.sin(t)
a_bias = 0.0
ya = a + a_bias + 0.1*np.random.randn(samples)
yp = p + 0.1*np.random.randn(samples)

# discretize
dt = t[1] - t[0]
print("dt:", dt)
la.expm(comb*dt)
Ad = la.expm(comb*dt)[:3, :3]
Bd = la.expm(comb*dt)[:3, 3:]
Ad, Bd

# %%
# Define the system matrices
F = Ad

H = np.array(C)

# Process noise covariance
Q = np.diag([.1, .2, .1])

# Measurement noise covariance
R = np.diag([[0.1]])

# Initial state estimate
x_hat = np.array([[0], [1], [0]])

# Initial error covariance
P = np.eye(3) * 100

# record = np.zeros((len(p), 3))
# pos_modulo = int(np.ceil(0.033/dt))
# # Kalman Filter loop
# for i in range(len(p)):
#     # Predict step
#     x_hat = F @ x_hat
#     P = F @ P @ F.T + Q

#     # Position update
#     if i % pos_modulo == 0 and i < len(p)//2:
#         Hp = H[0, :].reshape(1,3)
#         K = P @ Hp.T @ np.linalg.inv(Hp @ P @ Hp.T + R)
#         x_hat = x_hat + K @ (yp[i] - Hp @ x_hat)
#         P = (np.eye(3) - K @ Hp) @ P

#     # Acceleration update
#     Ha = H[1, :].reshape(1,3)
#     K = P @ Ha.T @ np.linalg.inv(Ha @ P @ Ha.T + R)
#     x_hat = x_hat + K @ (ya[i] - Ha @ x_hat)
#     P = (np.eye(3) - K @ Ha) @ P

#     record[i, :] = x_hat.T


# %%
# # plt.figure(dpi=200)
# plt.plot(t,p, label='position')
# plt.plot(t,record[:, 0], label='estimated pos', linestyle='dashed')
# # plt.scatter(np.arange(0, len(p), end), yp[::end], label='measured', marker='x', c='g')
# plt.legend()
# # velocity
# plt.figure()
# plt.plot(t, record[:, 1], label='velocity')
# plt.plot(t, v, label='true velocity')
# plt.legend()
# plt.figure()
# plt.plot(t, record[:, 2], label='acc')
# plt.plot(t, a, label='true acc')
# plt.legend()

# %%
import okf
import torch

# %%
H

# %%
def get_F():
    return torch.tensor(Ad, dtype=torch.double)


def get_H():
    return torch.tensor(H, dtype=torch.double)


def initial_observation_to_state(z):
    # z = [p,a]
    return torch.tensor([z[0], 0, z[1]], dtype=torch.double)


def loss_fun():
    return lambda pred, x: ((pred-x)**2).sum()


def model_args():
    return dict(
        dim_x=3,
        dim_z=2,
        init_z2x=initial_observation_to_state,
        F=get_F(),
        H=get_H(),
        loss_fun=loss_fun(),
    )

# %%
# Define model
lidar_model_args = model_args()
print('---------------\nModel arguments:\n', lidar_model_args)
# baseline_model = okf.OKF(**lidar_model_args, optimize=False, model_name='KF')
model = okf.OKF(**lidar_model_args, optimize=True, model_name='OKF')

# %%
X = [np.vstack((p,v,a)).astype(np.float64).T] * 100
Z = [np.vstack((yp, ya)).astype(np.float64).T] * 100

print('Data:')
print(f'Simulated states:\ta {type(X)} of {len(X):d} targets, each is a {type(X[0])} of shape (n_time_steps, {X[0].shape[1]}).')
print(f'Simulated observations:\ta {type(Z)} of {len(Z):d} targets, each is a {type(Z[0])} of shape (n_time_steps, {Z[0].shape[1]}).')

# %%
print(model.get_Q())
print(model.get_R())

res,_ = okf.train(model, Z, X, verbose=1, n_epochs=3, batch_size=64)

print(res)

print(model.get_Q())
print(model.get_R())

