import numpy as np
import jax
import jax.numpy as jnp
from orthogonalx_augm import IO_augmentation, xavier_init
from matplotlib import pyplot as plt


@jax.jit
def ANN_function(u, params):
    # input: u (N x nu)
    # output: y (N x ny)
    Wu = params[0]
    Wy = params[1]
    bu = params[2]
    by = params[3]
    return jnp.tanh(u @ Wu.T + bu) @ Wy.T + by


def gen_input_sequence(N, symm, low, high):
    if symm:
        half = int(N / 2)
        r = np.random.normal(loc=0, scale=high, size=half)
        u = np.empty(N)
        u[:half] = r
        u[half:] = -r
    else:
        u = np.random.uniform(low, high, N)
    return u


def delta(v):
    Fc = 0.8
    Fs = 1.2
    vs = 0.02
    return Fc * np.sign(v) + (Fs - Fc) * np.exp(-np.abs(v) / vs) * np.sign(v)


def generate_data_set(theta_true, N=600, symm=False, umin=-1., umax=1., noise=False):

    # generate input
    u = gen_input_sequence(N, symm, umin, umax)

    # simulate output
    y = np.zeros(N)
    e = np.zeros(N)
    for k in range(N-2):
        y[k+2] = theta_true[0] * y[k+1] + theta_true[1] * y[k] + theta_true[2] * u[k+1] - theta_true[2] * delta((y[k+1]-y[k])/Ts)
        if noise:
            ek = np.random.normal(scale=0.00018)
            e[k+2] = ek
            y[k+2] += ek
    # build X for k=2..N-1
    X = np.zeros((N, 3))
    for k in range(N-2):
        X[k+2, :] = [y[k+1], y[k], u[k+1]]
    return X[2:, :], y[2:].reshape(-1, 1), e[2:].reshape(-1)


# dummy phi function as X already contains all elements
@jax.jit
def phi_fun(x):
    return jnp.array(x)


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

    # User definitions
    N = 10000  # training data length
    seed = 0  # for Monte Carlo study
    lambda_orth = 1.  # trade-off parameter for orthogonal regularization
    lambda_reg = 0.  # trade-off parameter for baseline regularization

    # true baseline parameters
    m = 1.
    c = 100.
    k = 1000.
    Ts = 0.01
    Gamma = m + c * Ts + k * Ts**2
    theta_true = np.array([(2*m-c*Ts)/Gamma, -m/Gamma, Ts**2 / Gamma])

    # data generation
    np.random.seed(12345)  # for reproducibility
    X_train, y_train, e_k = generate_data_set(theta_true, umin=-100., umax=100., N=N, symm=True, noise=True)
    X_test, y_test, _ = generate_data_set(theta_true, umin=-100., umax=100., N=1000)

    # Model initialization
    theta_base = np.array([0.61, -0.52, 4.7e-5])

    # 1 hidden layer ANN
    nn = 16
    nu = 3
    ny = 1
    np.random.seed(seed)
    Wu_init = xavier_init(n_in=nu, n_out=nn, act_fun="tanh")
    Wy_init = xavier_init(n_in=nn, n_out=ny)
    bu_init = np.zeros(nn)
    by_init = np.zeros(ny)

    thetaInit = [Wu_init, Wy_init, bu_init, by_init, theta_base]


    model = IO_augmentation(ANN_function=ANN_function, phi_function=phi_fun)
    model.initialize(thetaInit)
    model.set_training_params(adam_epochs=2500, lbfgs_epochs=500, lbfgs_memory=20)
    model.fit(X_train, y_train, orth_regul_coeff=lambda_orth, simple_reg_coeff=lambda_reg)

    Y_pred_train, _, _ = model.predict(X_train)
    Y_pred_test, _, _ = model.predict(X_test)

    RMSE_train = np.sqrt(((y_train.reshape(-1) - Y_pred_train.reshape(-1))**2).mean())
    RMSE_test = np.sqrt(((y_test.reshape(-1) - Y_pred_test.reshape(-1))**2).mean())
    print(f"Training RMSE = {RMSE_train}")
    print(f"Test RMSE = {RMSE_test}\n")

    theta_base_final = np.array(model.params[-1])

    print(f"Initial baseline params: [{theta_base[0]}, {theta_base[1]}, {theta_base[2]}]")
    print(f"True baseline params: [{theta_true[0]}, {theta_true[1]}, {theta_true[2]}]")
    print(f"Tuned baseline params: [{theta_base_final[0]}, {theta_base_final[1]}, {theta_base_final[2]}]")
    print(f"Error of baseline params (scaled): {np.linalg.norm((theta_true-theta_base_final)/theta_true, 2)}")

    plt.figure()
    plt.plot(y_test, label="True")
    plt.plot(y_test - Y_pred_test, label="Pred. error")
    plt.legend()
    plt.show()
