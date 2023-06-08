'''Tests of krr module

'''

from krr import forward_substitution, backward_substitution, train_ridge, linear, gaussian, laplacian
import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import cdist


def train_ridge_sklearn(x, y, λ, my_kernel, *par):
    '''
    Fitting of Kernelised Ridge Regression model  with sklearn's KernelRidge.

    Parameters:
    -----------
    x (2D array of shape (d,n)): each row is a training input sample of length d,
    where d is the dimension of the model
    y (1D array of length n): training output (label)
    λ (float): regularization parameter
    my_kernel (function type): type of kernel
    *par (iterable): parameters needed for the kernel

    Returns:
    --------
    alpha (1D array of length n): optimized regression coefficients

    Notes
    -----
    Comparison with these results is needed to test our implementation
    '''
    K=cdist(x, x, metric= lambda x, y: my_kernel(x, y, *par))
    #K = my_kernel(x, x, *par)
    print("kernel", K)
    ridge = KernelRidge(alpha=λ, kernel='precomputed')
    ridge.fit(K, y)
    alpha = ridge.dual_coef_

    return alpha


@pytest.fixture
def L():
    decomposable = np.array([[4, 12, -16],
                             [12, 37, -43],
                             [-16, -43, 98]])
    kernel = np.linalg.cholesky(decomposable)
    return kernel

@pytest.fixture
def y():
    return np.array([1, 2, 3])


def test_forward(L,y):
    '''
    Tests forward substitution

    GIVEN: an (n,n) upper triangular matrix L and an n-vector y
    WHEN: I apply forward_substitution to them
    THEN: I check that L times the result gives y
    '''
    beta = forward_substitution(L,y)
    residual = np.dot(L, beta) - y
    assert np.allclose(residual, np.zeros_like(residual))


def helper_backward(m, v):
    '''
    Helper function to test backward_substitution

    Parameters:
    -----------
    m (2D array of shape (n,n)): upper triangular matrix
    v (1D array of length n)

    Returns:
    --------
    residual (1D array of length n)

    Notes:
    ------
    Applying backward_substitution, m multiplied by the result should give back v
    '''
    alpha = backward_substitution(m, v)
    residual = np.dot(m, alpha) - v
    print("dot",np.dot(m, alpha))
    print("v", v)
    return residual



def test_backward(L,y):
    '''
    Test backward_substitution

    GIVEN: an (n,n) upper triangular matrix L.T and an n-vector my_beta, result of forward_substitution
    WHEN: I apply backward_substitution
    THEN: I check that the residual error is close to zero

    See also:
    ---------
    helper_backward, krr.forward_substitution
    '''
    my_beta=forward_substitution(L,y)
    residual = helper_backward(L.T, my_beta)
    assert np.allclose(residual, np.zeros_like(residual))
    #is_backward(L.T,y)
    #print("Computed beta:", beta)
    #print("dot:", np.dot(m, beta)) # Check if computed beta satisfies the equation m * beta = y


@pytest.mark.parametrize("my_kernel", [linear, gaussian, laplacian])
def test_train_ridge(my_kernel):
    '''
    Compares parameters of KRR trained with our implementation to sklearn.KernelRidge

    GIVEN: A kernel matrix
    WHEN: I apply train_ridge to it
    THEN: I test that the trained model is compatible with sklearn implementation
    '''
    # Set up input data
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 20, 30])
    λ = 0.1
    sigma = 1.0

    # Call my implementation of train_ridge
    alpha = train_ridge(x, y, λ, my_kernel, sigma)

    # Call scikit-learn's KernelRidge
    alpha_sklearn = train_ridge_sklearn(x, y, λ, my_kernel, sigma)

    # Compare the results
    np.testing.assert_allclose(alpha, alpha_sklearn, atol=1e-6)
