import logging
#import traceback
#import pprint
import numpy as np
from scipy.spatial.distance import cdist
logger = logging.getLogger(__name__)

'''
Kernelised Ridge Regression (KRR). Train and predict a KRR model, using
a linear/gaussian/laplacian kernel.
'''

def linear(xi,xj, sigma = None):
    '''
    Linear kernel function
    PARAMETERS:
    xi (1D array of length d)
    xj (1D array of length d)
    sigma: only present for formal compatibility with the other kernels,
           must be set to None

    RETURNS:
    k (float): value of the linear kernel between xi and xj
    '''
    return np.dot(xi,xj)#np.einsum('ij, kj -> ik', xi, xj)

def gaussian(xi,xj, sigma):
    '''
    Gaussian kernel function
    PARAMETERS:
    xi (1D array of length d)
    xj (1D array of length d)
    sigma (float)
    RETURNS:
    k (float): value of the gaussian kernel between xi and xj
    '''
    #logger.debug("arguments received by gaussian \n xi %s xj \n %s sigma \n %s ", pprint.pformat(xi),pprint.pformat(xj),sigma)
    #logger.debug ("norm %s", np.linalg.norm(xi- xj)**2)
    k = np.exp( -1 * np.linalg.norm(xi- xj)**2 / (2*sigma**2) )
    #logger.debug("gaussian kernel %s", k)
    return k

def laplacian(xi,xj, sigma):
    '''
    Laplacian kernel function
    PARAMETERS:
    xi (1D array of length d)
    xj (1D array of length d)
    sigma (float)
    RETURNS:
    k (float): value of the laplacian kernel between xi and xj
    '''
    #logger.debug("arguments received by gaussian \n xi %s xj \n %s sigma \n %s ", pprint.pformat(xi),pprint.pformat(xj),sigma)
    #logger.debug ("norm %s", np.linalg.norm(xi- xj)**2)
    k = np.exp( -1 * np.linalg.norm(xi- xj, ord = 1) / sigma )
    #logger.debug("gaussian kernel %s", k)
    return k

def forward_substitution(m,v):
    '''
    For solution of linear systems for lower triangular matrix
    PARAMETERS:
    m (2D array of shape (n,n)):
    v (1D array of length n):
    RETURNS:
    beta (1D array of length n):
    '''
    n= len(v)
    beta = np.zeros(n)

    for i in range(n):
        beta[i] = v[i]
        for j in range(i):
            beta[i] -= m[i,j]* beta[j]
        beta[i]/=m[i,i]
    #print("y",y)
    return beta

def backward_substitution(m,v):
    '''
    For solution of linear systems for upper triangular matrix
    PARAMETERS:
    m (2D array of shape (n,n)):
    v (1D array of length n):
    RETURNS:
    alpha (1D array of length n):
    '''
    n=len(v)
    alpha = np.empty_like(v)

    for i in range(n-1,-1,-1):
        alpha[i] = v[i]
        for j in range(i+1,n):#(n-1,i,-1):
            alpha[i] -=m[i,j]* alpha[j]
        alpha[i]/=m[i,i]
    return alpha

def train_ridge(x,y,λ, my_kernel, *par):
    '''
    Fitting of Kernelised Ridge Regression model using Cholesky decomposition.
    PARAMETERS:
    x (2D array of shape (d,n)): each row is a training input sample of length d,
    where d is the dimension of the model
    y (1D array of length n): training output (label)
    λ (float): regularization parameter
    my_kernel (function type): type of kernel (linear, gaussian, laplacian)
    *par : parameters needed to compute the kernel.
    RETURNS:
    alpha (1D array of length n): optimized regression coefficients

    The regression coefficients alpha are obtained by solving the linear system
    of equations (K + lambda * I) * alpha = y, where K + lambda * I is symmetric
    and strictly positive definite. One way to do this for numerical stability,
    is to use Cholesky decomposition K + lambda * I = U.T * U,
    where U is upper triangular (.T indicates the transpose).
    One then solves  U.T * U * alpha = y by solving two linear systems of equations,
    first U.T * beta = y, then U * alpha = beta. Since U.T is lower triangular
    and U is upper triangular, this requires only two straightforward passes
    over the data called forward and backward substitution, respectively.
    '''

    K=cdist(x, x, metric= lambda x, y: my_kernel(x, y, *par))#my_kernel(x, x, *par)
    decomposable = K+λ * np.identity(len(y))
    U= np.linalg.cholesky(decomposable)
    beta = forward_substitution(U, y)
    #print("beta",beta.shape)
    alpha = backward_substitution(U.T, beta)
    #print("alpha",alpha.shape)
    #logger.info("(K+lambda I)*alpha \n%s",np.matmul(decomposable, alpha))
    #logger.info("y \n%s",y)
    logger.info("y: %s", y)
    logger.info("K: %s", K)
    logger.info("decomposable: %s", decomposable)
    logger.info("U: %s", U)
    logger.info("beta: %s", beta)
    logger.info("alpha: %s", alpha)
    logger.info("decomposable*alpha: %s",np.matmul(decomposable, alpha))
    return alpha


def predict_ridge(alpha, x,xbar, my_kernel, *par):
    '''
    Prediction with Kernelised Ridge Regression model
    PARAMETERS:
    alpha (1D array of length n): computed regression coefficients
    x (2D array of shape (d,n)): every row is a training input sample
    xbar (2D array of shape (d,n_bar)): every row is a prediction input sample
    my_kernel (function type): type of kernel
    *par : parameters needed to compute the kernel matrix; must be the same used in training.

    RETURNS:
    f_xbar (1D array of length n_bar): prediction output for n_bar samples
    '''
    logger.debug("Entered predict_ridge")
    #logger.debug("x %s, \n xbar %s", x, xbar)
    try:
        L = cdist(x, xbar, metric= lambda x, y: my_kernel(x, y, *par)) #shape (n,nbar)
        #L= my_kernel(x[:, np.newaxis, :], xbar[np.newaxis, :,:],  *par) #L= my_kernel(my_xbar[:,:,None],my_xbar[:,None,:]) #kernel matrix between training inputs and prediction inputs my_kernel(xi,xjbar)
    except Exception as e:
        #logger.error(traceback.format_exc())
        logger.exception("An exception occurred: %s", str(e))
        raise
    logger.debug("shape of L.T %s \n shape of alpha %s", L.T.shape, alpha.shape)
    f_xbar= np.matmul(L.T, alpha)
    #trythis = (L.T * alpha)
    #logger.debug("shape of result %s \n shape of old result %s ", f_xbar.shape, trythis.shape)
    return f_xbar


#my_kernel(my_x[:,:,None],my_x[:,None,:]) #kernel matrix between training samples my_kernel(xi,xj)
