import jax
from jax import numpy as jnp
import numpy as np
from orthogonalx_augm import IO_orthogonal_augmentation, xavier_init
from matplotlib import pyplot as plt
import seaborn as sns
import os


@jax.jit
def ANN_function(u, params):
    # input: u (N x nu)
    # output: y (N x ny)
    Wu = params[0]
    Wy = params[1]
    bu = params[2]
    by = params[3]
    return jnp.tanh(u @ Wu.T + bu) @ Wy.T + by


@jax.jit
def generate_phi(U):
    return jnp.hstack((U, U ** 3))


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

    # User definitions
    N = 512  # half of the training data length (data is 2N long)
    seed = 0  # for Monte Carlo study
    noise = False  # if True, adds noise to achieve 30 dB SNR
    estim_data = 0  # 0 - symmetric distribution with "trick", 1 - symmetric distribution (only with N->inf), 2 - asymm. distr.
    estimate_covariance = False  # if True, the script calculates and plots the estimated covariance matrix

    # data generation
    np.random.seed(10)  # for data generation: keep it constant for reproducibility!
    if estim_data == 0:
        u_train = 0.3 * np.random.randn(N, 1)
        u_train = np.vstack((u_train, -u_train))
    elif estim_data == 1:
        u_train = 0.3 * np.random.randn(2 * N, 1)
    elif estim_data == 2:
        u_train = 0.3 * np.random.randn(2 * N, 1) - 0.01
    else:
        raise ValueError(f"Invalid estim_data: {estim_data}")

    u_test = 0.3 * np.random.randn(1024, 1)

    theta_ast = np.array([0.01, 1., -0.5, 0.1])  # true baseline params

    y_train = theta_ast[0] + theta_ast[1] * u_train + theta_ast[2] * u_train ** 2 + theta_ast[3] * u_train ** 3
    y_test = theta_ast[0] + theta_ast[1] * u_test + theta_ast[2] * u_test ** 2 + theta_ast[3] * u_test ** 3

    # adding additional noise (if required)
    if noise:
        sigma_n = 0.009
        noise = np.random.normal(0, sigma_n, y_train.shape)
        Py = np.sum(np.square(y_train), axis=0)
        Pn = np.sum(np.square(noise), axis=0)
        y_train += noise
        SNR_train = 10 * np.log10(Py / Pn)
        print(f"Training SNR: {SNR_train}")


    # Model initialization
    theta_base = np.array([0.8, 0.03])
    theta_true = np.array([1., 0.1])

    # 1 hidden layer ANN
    nn = 16
    nu = 1
    ny = 1
    np.random.seed(seed)
    Wu_init = xavier_init(n_in=nu, n_out=nn, act_fun="tanh")
    Wy_init = xavier_init(n_in=nn, n_out=ny)
    bu_init = np.zeros(nu)
    by_init = np.zeros(ny)

    thetaInit = [Wu_init, Wy_init, bu_init, by_init, theta_base]

    model = IO_orthogonal_augmentation(ANN_function=ANN_function, phi_function=generate_phi)
    model.initialize(thetaInit)
    model.set_training_params(adam_epochs=500, lbfgs_epochs=1000, lbfgs_memory=20)
    model.fit(u_train, y_train)

    orth_eval_train = model.training_orthogonality
    print(f"Orthogonality check on training data ([u u**3].T @ y_aug]):  [{orth_eval_train[0, 0]}, {orth_eval_train[1, 0]}]\n")

    Y_pred_train, orth_ANN, y_base_train, raw_ANN = model.predict(u_train)
    Y_pred_test, _, y_base_test, _ = model.predict(u_test)

    RMSE_train = np.mean(np.sum((y_train - Y_pred_train) ** 2))
    RMSE_test = np.mean(np.sum((y_test - Y_pred_test) ** 2))
    print(f"Training RMSE = {RMSE_train}")
    print(f"Test RMSE = {RMSE_test}\n")

    theta_base_final = np.array(model.params[-1])

    print(f"Initial baseline params: [{theta_base[0]}, {theta_base[1]}]")
    print(f"True baseline params: [{theta_ast[1]}, {theta_ast[3]}]")
    print(f"Tuned baseline params: [{theta_base_final[0]}, {theta_base_final[1]}]")
    print(f"Error of baseline params: {np.linalg.norm(theta_true-theta_base_final, 2)}")

    if estimate_covariance:
        H, _ = model.approx_covariance_mx(u_train, y_train, 2*N, 1)
        H_masked = H.copy()
        H_masked[np.abs(H) < 1e-6] = np.nan

        cmap = sns.diverging_palette(220, 10, as_cmap=True)  # normal colormap
        # Use matplotlib's colormap with NaN handling
        cmap.set_bad(color='k')  # color for NaNs

        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)

        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        sns.heatmap(H_masked, linewidth=0.5, square=True, cmap=cmap, ax=ax, cbar_kws={"shrink": 0.73})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel(r'$\theta_j$')
        ax.set_ylabel(r'$\theta_i$')
        plt.tight_layout()
        plt.show(block=False)

    plt.figure(layout="tight")
    plt.title("Training data: baseline + orth. augm. decomposition")
    plt.plot(u_train, y_base_train, '.', label="Baseline output")
    plt.plot(u_train, orth_ANN, '.', label="Orth. augm. output")
    plt.xlabel("u")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show(block=False)

    plt.figure(layout="tight")
    plt.title("Training data: learning component before and after orthogonalization")
    plt.plot(u_train, raw_ANN, '.', label="Raw ANN")
    plt.plot(u_train, orth_ANN, '.', label="orth. ANN")
    plt.grid()
    plt.legend()
    plt.show(block=False)

    u_test_2 = np.linspace(-1, 1, 100).reshape(-1, 1)
    unmodeled_terms = theta_ast[0] + theta_ast[2] * u_test_2 ** 2
    _, orth_ANN, _, _ = model.predict(u_test_2)

    plt.figure()
    plt.plot(u_test_2, unmodeled_terms, label="Unmodeled terms")
    plt.plot(u_test_2, orth_ANN, '--', label="Learning comp.")
    plt.grid()
    plt.legend()
    plt.show()
