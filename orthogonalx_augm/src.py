import jax
from jax import numpy as jnp
import numpy as np
import jaxopt
import time
from jax_sysid.models import lbfgs_options, adam_solver
from typing import Callable


def xavier_init(n_in: int, n_out: int, act_fun="linear"):
    """
    Implements the well-known Xavier initialization for the weigths of a given ANN layer. The implementation follows the PyTorch version.

    Args:
        n_int (int): Input dimension of the ANN layer.
        n_out (int): Output dimension of the ANN layer.
        act_fun (str, optional): Activation function for the ANN layer. Defaults to "linear".

    Returns:
        W (array-like): Initialized weight matrix. 
    """
    if act_fun == "linear":
        gain = 1.
    elif act_fun == "relu":
        gain = np.sqrt(2.)
    elif act_fun == "tanh":
        gain = 5. / 3.
    else:
        raise NotImplementedError("Initialization is not implemented for further act. functions.")

    a = gain * np.sqrt(6. / (n_in + n_out))
    W = np.random.uniform(low=-a, high=a, size=(n_out, n_in))
    return W


class IO_augmentation(object):
    """
    Standard additive model augmentation structure for IO baseline models.
    """
    def __init__(self, ANN_function: Callable, phi_function: Callable):
        """
        Initialization of the IO_augmentation model class.

        Args:
            ANN_function (Callable): Implemented ANN function for the learning component with user-specified architecture.
            phi_function (Callable): Implemented function that returns the baseline regressor matrix-function.
        """
        super().__init__()
        self.ANN_fun = ANN_function
        self.phi = phi_function
        self.params = None

        # default training parameters (if not specified otherwised)
        self.adam_epochs = 1000
        self.adam_lr = 1e-3
        self.lbfgs_epochs = 0
        self.lbfgs_options = None
        self.lbfgs_tol = 1e-8
        self.lbfgs_memory = 10

    def initialize(self, params: list):
        """
        Initializes the tunable parameters of the structure.

        Args:
            params (list): Initial values of the optimized parameters. List elements should be numpy arrays with the last 1D numpy array containing
                the baseline parameters.
        """
        self.params = params

    def set_training_params(self, adam_epochs=None, lbfgs_epochs=None, adam_lr=None, lbfgs_tol=None, lbfgs_memory=None):
        """
        Sets the optimization-related hyperparameters.

        Args:
            adam_epochs (int, optional): Number of Adam epochs applied at the beginning of optimization. If None, 1000 epochs are used, Defaults to None.
            lbfgs_epochs (int, optional): Number of iterations for the L-BFGS optimization after the Adam algorithm. If None, 0 is applied. Defaults
                to None.
            adam_lr (float, optional): Learning rate for the Adam algorithm. If None, it is set to 1e-3. Defaults to None.
            lbfgs_tol (float, optional): Tolerance for the L-BFGS method. If None, it is set to 1e-8. Defaults to None.
            lbfgs_memory (int, optional): Memory for the L-BFGS method. If None, it is set to 10. Defaults to None.
    """
        if adam_epochs is not None:
            self.adam_epochs = adam_epochs
        if lbfgs_epochs is not None:
            self.lbfgs_epochs = lbfgs_epochs
        if adam_lr is not None:
            self.adam_lr = adam_lr
        if lbfgs_tol is not None:
            self.lbfgs_tol = lbfgs_tol
        if lbfgs_memory is not None:
            self.lbfgs_memory = lbfgs_memory

        self.lbfgs_options = lbfgs_options(50, self.lbfgs_epochs, self.lbfgs_tol, self.lbfgs_memory)

    def fit(self, Xi_train: np.ndarray, Y_train: np.ndarray):
        """
        Trains the standard additive augmentation model.

        Args:
            Xi_train (array-like): Contains the baseline regression points of the training data. Its shape should be compatible with the
                provided phi_function.
            Y_train (array-like): Contains the measured output values of the training set.
        """

        Phi_train = self.phi(Xi_train)

        @jax.jit
        def J(params):
            learning_component = self.ANN_fun(Xi_train, params)
            theta_base = params[-1]
            baseline_component = Phi_train @ theta_base.reshape(-1, 1)
            Y_pred = baseline_component + learning_component
            Y_error = jnp.mean(jnp.sum((Y_train - Y_pred) ** 2, axis=0))
            return Y_error

        def JdJ(th):
            return jax.value_and_grad(J)(th)

        # ADAM optimization
        t0 = time.time()
        p_opt, J_opt = adam_solver(JdJ, self.params, self.adam_epochs, self.adam_lr, 1)

        if self.lbfgs_epochs > 0:
            solver = jaxopt.ScipyMinimize(fun=J, tol=1e-8, method="L-BFGS-B", maxiter=self.lbfgs_epochs,
                                          options=self.lbfgs_options)
            p_opt, state = solver.run(p_opt)
            iter_num = state.iter_num
            print('L-BFGS-B done in %d iterations.' % iter_num)
            # save optimization information
            self.lbfgs_info = state
        tend = time.time()
        print(f"Training finished in {tend - t0} seconds.\n")

        # save model parameters
        self.params = p_opt

    def predict(self, Xi_test: np.ndarray):
        """
        Predicts the system responses on a test data set with the trained model.

        Args:
            Xi_test (array-like): Contains the baseline regression points of the test data. Its shape should be compatible with the provided phi_function.

        Returns:
            tuple (Y_pred, baseline_component, learning_component)
            - Y_pred (array-like): The predicted model outputs with shape (Nt, ny) with Nt being the number of data points in the test set, ny is
            the output dimension.
            - baseline_component (array-like): The baseline part of the predicted output with shape (Nt, ny). Corresponds to the prediction of
            the baseline model with the tuned baseline parameters.
            - learning_component (array-like): The ANN part of the predicted output with shape (Nt, ny).
        """
        Phi_test = self.phi(Xi_test)
        baseline_component = Phi_test @ self.params[-1].reshape(-1, 1)
        learning_component = self.ANN_fun(Xi_test, self.params)
        Y_pred = baseline_component + learning_component
        return Y_pred, baseline_component, learning_component


class IO_orthogonal_augmentation(object):
    """
    Orthogonal-by-construction model augmentation structure for IO baseline models.
    """
    def __init__(self, ANN_function: Callable, phi_function: Callable):
        """
        Initialization of the IO_orthogonal_augmentation model class.

        Args:
            ANN_function (Callable): Implemented ANN function for the learning component with user-specified architecture. Simple ANN, without
                orthogonal projector.
            phi_function (Callable): Implemented function that returns the baseline regressor matrix-function.
        """
        super().__init__()
        self.ANN_fun = ANN_function
        self.phi = phi_function
        self.params = None
        self.theta_aux = None
        self.training_orthogonality = None

        # default training parameters
        self.adam_epochs = 1000
        self.adam_lr = 1e-3
        self.lbfgs_epochs = 0
        self.lbfgs_options = None
        self.lbfgs_tol = 1e-8
        self.lbfgs_memory = 10

    def initialize(self, params: list):
        """
        Initializes the tunable parameters of the structure.

        Args:
            params (list): Initial values of the optimized parameters. List elements should be numpy arrays with the last 1D numpy array containing
                the baseline parameters.
        """
        self.params = params

    def set_training_params(self, adam_epochs=None, lbfgs_epochs=None, adam_lr=None, lbfgs_tol=None, lbfgs_memory=None):
        """
        Sets the optimization-related hyperparameters.

        Args:
            adam_epochs (int, optional): Number of Adam epochs applied at the beginning of optimization. If None, 1000 epochs are used, Defaults to None.
            lbfgs_epochs (int, optional): Number of iterations for the L-BFGS optimization after the Adam algorithm. If None, 0 is applied. Defaults 
                to None.
            adam_lr (float, optional): Learning rate for the Adam algorithm. If None, it is set to 1e-3. Defaults to None.
            lbfgs_tol (float, optional): Tolerance for the L-BFGS method. If None, it is set to 1e-8. Defaults to None.
            lbfgs_memory (int, optional): Memory for the L-BFGS method. If None, it is set to 10. Defaults to None.
    """
        if adam_epochs is not None:
            self.adam_epochs = adam_epochs
        if lbfgs_epochs is not None:
            self.lbfgs_epochs = lbfgs_epochs
        if adam_lr is not None:
            self.adam_lr = adam_lr
        if lbfgs_tol is not None:
            self.lbfgs_tol = lbfgs_tol
        if lbfgs_memory is not None:
            self.lbfgs_memory = lbfgs_memory

        self.lbfgs_options = lbfgs_options(50, self.lbfgs_epochs, self.lbfgs_tol, self.lbfgs_memory)

    def fit(self, Xi_train: np.ndarray, Y_train: np.ndarray):
        """
        Trains the orthogonal-by-construction additive augmentation model.

        Args:
            Xi_train (array-like): Contains the baseline regression points of the training data. Its shape should be compatible with the
                provided phi_function.
            Y_train (array-like): Contains the measured output values of the training set.
        """

        Phi_train = self.phi(Xi_train)
        K_phi = jnp.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T

        @jax.jit
        def J(params):
            learning_component = self.ANN_fun(Xi_train, params)
            theta_base = params[-1]
            baseline_component = Phi_train @ theta_base.reshape(-1, 1)
            theta_aux = K_phi @ learning_component
            orth_component = learning_component - Phi_train @ theta_aux
            Y_pred = baseline_component + orth_component
            Y_error = jnp.mean(jnp.sum((Y_train - Y_pred) ** 2, axis=0))
            return Y_error

        def JdJ(th):
            return jax.value_and_grad(J)(th)

        # ADAM optimization
        t0 = time.time()
        p_opt, J_opt = adam_solver(JdJ, self.params, self.adam_epochs, self.adam_lr, 1)

        if self.lbfgs_epochs > 0:
            solver = jaxopt.ScipyMinimize(fun=J, tol=1e-8, method="L-BFGS-B", maxiter=self.lbfgs_epochs,
                                          options=self.lbfgs_options)
            p_opt, state = solver.run(p_opt)
            iter_num = state.iter_num
            print('L-BFGS-B done in %d iterations.' % iter_num)
        tend = time.time()
        print(f"Training finished in {tend - t0} seconds.\n")

        # save model parameters
        self.params = p_opt
        learning_component_train = self.ANN_fun(Xi_train, self.params)
        self.theta_aux = K_phi @ learning_component_train

        # orthogonality check
        self.training_orthogonality = Phi_train.T @ (learning_component_train - Phi_train @ self.theta_aux)

    def approx_Hessian(self, Xi_train: np.ndarray, Y_train: np.ndarray, N: int):
        """
        Approximates the hessian of the cost function at the optimized parameters.

        Args:
            Xi_train (array-like): Contains the baseline regression points of the training data. Its shape should be compatible with the
                provided phi_function.
            Y_train (array-like): Contains the measured output values of the training set.
            N (int): Length of the training data set.
        
        Returns:
            The approximated Hessian matrix.
        """

        def error(params):
            Phi_train = self.phi(Xi_train)
            K_phi = jnp.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T
            learning_component = self.ANN_fun(Xi_train, params)
            theta_base = params[-1]
            baseline_component = Phi_train @ theta_base.reshape(-1, 1)
            theta_aux = K_phi @ learning_component
            orth_component = learning_component - Phi_train @ theta_aux
            Y_pred = baseline_component + orth_component
            Y_error = Y_train - Y_pred
            return Y_error

        jac = jax.jacrev(error)(self.params)
        i_max = len(jac) - 1

        for i in range(len(jac)):
            Ji = np.squeeze(np.array(jac[i_max-i])).reshape(N, -1)
            if i == 0:
                J = Ji
            else:
                J = np.hstack((J, Ji))

        return 2 / N * J.T @ J

    def predict(self, Xi_test: np.ndarray):
        """
        Predicts the system responses on a test data set with the trained model.

        Args:
            Xi_test (array-like): Contains the baseline regression points of the test data. Its shape should be compatible with the provided phi_function.

        Returns:
            tuple (Y_pred, orth_component, baseline_component, raw_ANN_component)
            - Y_pred (array-like): The predicted model outputs with shape (Nt, ny) with Nt being the number of data points in the test set, ny is
            the output dimension.
            - orth_component (array-like): Returns the projected ANN part of the prediction that is interpreted as the learning component of the
            model prediction with shape (Nt, ny).
            - baseline_component (array-like): The baseline part of the predicted output with shape (Nt, ny). Corresponds to the prediction of
            the baseline model with the tuned baseline parameters.
            - raw_ANN_component (array-like): The raw ANN part of the learning component, before the orthogonal projection with shape (Nt, ny).
        """

        Phi_test = self.phi(Xi_test)
        baseline_component = Phi_test @ self.params[-1].reshape(-1, 1)
        raw_ANN_component = self.ANN_fun(Xi_test, self.params)
        orth_component = raw_ANN_component - Phi_test @ self.theta_aux
        Y_pred = baseline_component + orth_component
        return Y_pred, orth_component, baseline_component, raw_ANN_component
