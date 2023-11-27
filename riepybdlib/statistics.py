'''
Riemannian statistics module of riepybdlib package

Writing code takes time. Polishing it and making it available to others takes longer! 
If some parts of the code were useful for your research of for a better understanding 
of the algorithms, please reward the authors by citing the related publications, 
and consider making your own research available in this way.

@article{Zeestraten2017,
  	title = {An Approach for Imitation Learning on Riemannian Manifolds},
	author = {Zeestraten, M.J.A. and Havoutis, I. and Silverio, J. and Calinon, S. and Caldwell, D. G.},
	journal={{IEEE} Robotics and Automation Letters ({RA-L})},
	year = {2017},
	month={January},
}

 
Copyright (c) 2017 Istituto Italiano di Tecnologia, http://iit.it/
Written by Martijn Zeestraten, http://www.martijnzeestraten.nl/

This file is part of RiePybDlib, http://gitlab.martijnzeestraten.nl/martijn/riepybdlib

RiePybDlib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.
 
RiePybDlib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RiePybDlib. If not, see <http://www.gnu.org/licenses/>.
'''

from loguru import logger
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from functools import reduce
import numpy as np
import pbdlib as pbd
import riepybdlib.manifold as rm
import riepybdlib.plot as fctplt
from scipy.linalg import block_diag

from copy import deepcopy

from enum import Enum

realmin = pbd.functions.realmin
realmax = pbd.functions.realmax

def multiply_iterable(l):
    return reduce(lambda x, y: x*y, l)

class RegularizationType(Enum):
    NONE      = None
    SHRINKAGE = 1
    DIAGONAL  = 2
    COMBINED  = 3
    DIAGONAL_ONLY = 4
    ADD_CONSTANT = 5
    # NOTE: there's also spherical, which is an diagonal scaled by a constant
    # for all dimensions. Ie. like diagonal, but with a single value.
    # Didn't see a reasoon to implement this, as it is very restrictive.

colors = tuple(('tab:red', 'tab:green', 'tab:blue'))
labels = ['x', 'y', 'z']

# Taken from https://adamj.eu/tech/2021/10/13/how-to-create-a-transparent-attribute-alias-in-python/
class Alias:
    def __init__(self, source_name):
        self.source_name = source_name

    def __get__(self, obj, objtype=None):
        if obj is None:
            # Class lookup, return descriptor
            return self
        return getattr(obj, self.source_name)

    def __set__(self, obj, value):
        setattr(obj, self.source_name, value)

# For hashing
# https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
def array_to_tuple(np_array):
    """Iterates recursivelly."""
    try:
        return tuple(array_to_tuple(x) for x in np_array)
    except TypeError:
        return np_array


# Statistical function:
class Gaussian(object):
    def __init__(self, manifold, mu=None, sigma=None):
        '''Initialize Gaussian on manifold'''
        self.manifold = manifold

        if mu is None:
            # Default initialization
            self.mu = manifold.id_elem
        else:
            self.mu = mu 

        if sigma is None:
            # Default initialization
            self.sigma = np.eye(manifold.n_dimT)
        elif (sigma.shape[0] != manifold.n_dimT or
            sigma.shape[1] != manifold.n_dimT):
            raise RuntimeError('Dimensions of sigma do not match Manifold')
        else:
            self.sigma = sigma

    def prob_from_np(self, npdata, log=False):
        data = self.manifold.np_to_manifold(npdata)

        return self.prob(data, log=log)
        
    def prob(self,data, log=False):
        '''Evaluate probability of sample
        data can either be a tuple or a list of tuples
        '''

        # print(data)
        # print(self.mu)
        # print(self.sigma)

        # Regularization term
        #d = len(self.mu) # Dimensions
        d = self.manifold.n_dimT
        reg = np.sqrt( ( (2*np.pi)**d )*np.linalg.det(self.sigma) ) + 1e-200
        
        # Mahanalobis Distance:

        # Batch:
        dist = self.manifold.log(data, self.mu)

        # Correct size:
        if dist.ndim==2:
            # Correct dimensions
            pass
        elif dist.ndim==1 and dist.shape[0]==self.manifold.n_dimT:
            # Single element
            dist=dist[None,:]
        elif dist.ndim==1 and self.manifold.n_dimT==1:
            # Multiple elements
            dist=dist[:,None]

        try:
            dist = ( dist * np.linalg.solve(self.sigma,dist.T).T ).sum(axis=(dist.ndim-1))
        except np.linalg.LinAlgError:
            logger.warning('Singular matrix, adding diag constant', filter=False)
            try:
                diag = np.diag([1e-20] * self.sigma.shape[0])
                dist = ( dist * np.linalg.solve(self.sigma + diag,dist.T).T ).sum(axis=(dist.ndim-1)) 
            except np.linalg.LinAlgError:
                logger.warning('Singular matrix, using lstsq', filter=False)
                dist = ( dist * np.linalg.lstsq(self.sigma,dist.T,rcond=None)[0].T ).sum(axis=(dist.ndim-1))
        # probs =  np.exp( -0.5*dist )/reg 

        log_lik = -0.5*dist - np.log(reg)

        return log_lik if log else np.exp(log_lik)

        # Iterative
        #probs = []
        #for i, x in enumerate(data):
        #    dist = self.manifold.log(self.mu,x)
        #    dist = np.sum(dist*np.linalg.solve(self.sigma,dist),0)
        #    probs.append( np.exp( -0.5*dist )/reg )
        
        # Return results
        #print(probs)
        # return probs
        #if len(probs) == 1:
        #    return probs[0]
        #else:
        #    return np.array(probs)

    def margin(self,i_in):
        '''Compute the marginal distribution'''
        
        # Compute index:
        if type(i_in) is list:
            mu_m = tuple([ self.mu[i] for _,i in enumerate(i_in)])
        else:
            mu_m = self.mu[i_in]
        
        ran  = self.manifold.get_tangent_indices(i_in)
        sigma_m = self.sigma[np.ix_(ran,ran)]
        
        return Gaussian(self.manifold.get_submanifold(i_in),mu_m, sigma_m  )

    def mle(self, x, h=None, reg_lambda=1e-3, reg_lambda2=1e-3,
            reg_type=RegularizationType.SHRINKAGE, plot_process=False):
        '''Maximum Likelihood Estimate
        x         : input data
        h         : optional list of weights
        reg_lambda: Regularization factor
        reg_type  : Covariance RegularizationType         
        '''
        x = self.manifold.swapto_tupleoflist(x)

        self.mu    = self.__empirical_mean(x, h, plot_process=plot_process)
        self.sigma = self.__empirical_covariance(x, h, reg_lambda, reg_lambda2,
                                                 reg_type)
        return self


    def __empirical_mean(self, x, h=None, plot_process=False):
        '''Compute Emperical mean
        x   : (list of) manifold element(s)
        h   : optional list of weights, if not specified weights we be taken equal
        '''
        mu = self.mu
        diff =1.0
        it = 0;
        if plot_process:
            from matplotlib.animation import ArtistAnimation
            from matplotlib.lines import Line2D

            # Size of x is number of manifolds. First mani might be time, then
            # per frame pos, rot, (pos delta, rot delta).
            # Not interested in time and pos (and pos delta), so // 2.
            n_frames = len(x) // 2
            incl_time = len(x) % 2 == 1
            fig, ax = plt.subplots(1, n_frames)
            fig.set_size_inches(n_frames*3, 5)
            if n_frames == 1:
                ax = [ax]
            artists = []

            handles = [Line2D([0], [0], color=c, label=l)
                       for c, l in zip(colors, labels)] + \
                      [Line2D([0], [0], color='black', linestyle='solid',
                                label='data'),
                       Line2D([0], [0], color='black', linestyle='dashed',
                                label='base')]
            fig.legend(handles=handles, ncols=2)

            def update(dtmp, base_mani, iter):
                # fragment_length = 20
                # n_fragmens = dtmp.shape[0] // fragment_length
                # assert dtmp.shape[0] % fragment_length == 0

                it_artists = []
                for c in range(n_frames):
                    j = int(incl_time) + 6*c + 3   # time, 6D per frame, then skip over 3 pos dims
                    # for f in range(n_fragmens):
                    for i in range(3):
                        # fragment_slice = slice(f*fragment_length, (f+1)*fragment_length)
                        # it_artists.extend(
                        #     ax[c].plot(dtmp[fragment_slice, j+i]*360/np.pi,
                        #                 label=labels[i], c=colors[i]))
                        it_artists.extend(
                            ax[c].plot(dtmp[:, j+i]*360/np.pi,c=colors[i]))
                    for i in range(3):  # for nicer legend order
                        it_artists.append(
                            ax[c].axhline(base_mani[j+i]*360/np.pi, color=colors[i],
                                          linestyle = 'dashed'))
                it_artists.append(ax[0].annotate(f'Iteration {iter}', (0,0),
                                  xycoords='figure fraction'))
                return it_artists
        else:
            update = None
            artists = None

        while (diff > 1e-6): 
            # logger.info(f'Mean computation iter {it} current diff: {diff}')
            delta = self.__get_weighted_distance(x, mu, h, plot_cb=update,
                                                 plot_artists=artists, it=it)
            mu = self.manifold.exp(delta, mu)
            diff = sum(delta*delta)
            
            it+=1
            if it >50:
                logger.warning('Gaussian mle not converged in 50 iterations.')
                # raise RuntimeWarning('Gaussian mle not converged in 50 iterations.')
                break
        #print('Converged after {0} iterations.'.format(it))

        if plot_process:
            ani = ArtistAnimation(fig, artists, interval=200, repeat=True)
            html_widget = HTML(ani.to_jshtml())
            display(html_widget)
            plt.close(fig)

            # import datetime
            # file_name = f'/home/hartzj/Desktop/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif'
            # ani.save(file_name)  #, writer="imagemagick")
            # raise KeyboardInterrupt
            # HTML(ani.to_jshtml())
            # plt.show()


        return mu
        
    def __get_weighted_distance(self, x, base, h=None, plot_cb=None,
                                plot_artists=None, it=None):
        ''' Compute the weighted distance between base and the elements of X
        base: The base of the distance measure usedmanifold element
        x   : (list of) manifold element(s)
        h   : optional list of weights, if not specified weights we be taken equal
        '''
        # Create weights if not supplied:
        if h is None:
            # No weights given, equal weight for all points
            if type(x) is np.ndarray:
                n_data = x.shape[0]
            elif type(x) is list:
                n_data = len(x)
            elif type(x[0]) is list:
                n_data = len(x[0])
            elif type(x[0]) is np.ndarray :
                n_data = x[0].shape[0]
            else:
                raise RuntimeError('Could not determine dimension of input')
            h = np.ones(n_data)/n_data

        # Compute weighted distance
        #d = np.zeros(self.manifold.n_dimT)
        dtmp = self.manifold.log(x, base)
        base_mani = self.manifold.log(self.manifold.id_elem, base)

        # print(base[-1])
        #print('dtmp.shape: ' ,dtmp.shape)
        #print('h.shape: ', h.shape)
        d = h.dot(self.manifold.log(x, base))
        #print('d.shape: ', d.shape)

        #for i,val in enumerate(x):
        #    d += h[i]*self.manifold.log(base, val)

        if plot_cb is not None:
            it_artists = plot_cb(dtmp, base_mani, it)
            plot_artists.append(it_artists)

        return d

    def __empirical_covariance(self, x, h=None, reg_lambda=1e-3, reg_lambda2=1e-3,
                               reg_type=RegularizationType.SHRINKAGE):
        '''Compute emperical mean
        x         : input data
        h         : optional list of weights
        reg_lambda: Regularization factor
        reg_type  : Covariance RegularizationType         
        '''

        # Create weights if not supplied:
        if h is None:
            # No weights given, equal weight for all points
            # Determine dimension of input
            if type(x) is np.ndarray:
                n_data = x.shape[0]
            elif type(x) is list:
                n_data = len(x)
            elif type(x[0]) is list:
                n_data = len(x[0])
            elif type(x[0]) is np.ndarray :
                n_data = x[0].shape[0]
            else:
                raise RuntimeError('Could not determine dimension of input')
            h = np.ones(n_data)/n_data
        
        # Compute covariance:
        # Batch:
        tmp   = self.manifold.log(x, self.mu)   # Project data in tangent space
        sigma = tmp.T.dot( np.diag(h).dot(tmp)) # Compute covariance
        # Iterative: 
        #sigma = np.zeros( (self.manifold.n_dimT, self.manifold.n_dimT) )
        #tmplist = []
        #for i,val in enumerate(x):
        #    tmp = self.manifold.log(self.mu, val)[:,None]
        #    sigma += h[i]*tmp.dot(tmp.T)
            
        if reg_type is RegularizationType.SHRINKAGE:
            return reg_lambda*np.diag(np.diag(sigma)) + (1-reg_lambda)*sigma
        elif reg_type is RegularizationType.DIAGONAL:
            return sigma + reg_lambda*np.eye(len(sigma))
        elif reg_type is RegularizationType.COMBINED:
            return reg_lambda*np.diag(np.diag(sigma)) + (1-reg_lambda)*sigma + reg_lambda2*np.eye(len(sigma))
        elif reg_type is RegularizationType.DIAGONAL_ONLY:
            return sigma * np.eye(len(sigma))
        elif reg_type is RegularizationType.ADD_CONSTANT:
            return sigma + reg_lambda
        elif reg_type is RegularizationType.NONE or reg_type is None:
            return sigma
        else:
            raise ValueError('Unknown regularization type for covariance regularization')
        
    def _update_empirical_covariance(self, x, h=None, reg_lambda=1e-3, reg_lambda2=1e-3,
                               reg_type=RegularizationType.SHRINKAGE):
        self.sigma = self.__empirical_covariance(x, h, reg_lambda, reg_lambda2, reg_type)

    def condition(self, val, i_in=0, i_out=1):
        '''Gaussian Conditioniong
        val  : (list of) elements on submanifold i_in
        i_in : index of  the input sub-manifold
        i_out: index of the sub-manifold to output
        '''

        ## ----- Pre-Processing:
        # Convert indices to list:
        if type(i_in) is not list:
            i_in = [i_in]
        if type(i_out) is not list:
            i_out = [i_out]

        # Marginalize Gaussian to related manifolds
        # Here we create a structure [i_in, iout]
        i_total = [i for ilist in [i_in,i_out] for i in ilist]
        gtmp = self.margin(i_total)

        # Construct new indices:
        i_in  = list(range(len(i_in)))
        i_out = list(range(len(i_in),len(i_in) + len(i_out)))

        # Get related manifold
        man     = gtmp.manifold
        man_in  = man.get_submanifold(i_in)
        man_out = man.get_submanifold(i_out)

        # Seperate Mu:
        # Compute index:
        mu_i = gtmp.margin(i_in).mu
        mu_o = gtmp.margin(i_out).mu

        # Compute lambda and split:
        Lambda    = np.linalg.inv(gtmp.sigma)

        ran_in  = man.get_tangent_indices(i_in)
        ran_out = man.get_tangent_indices(i_out)
        Lambda_ii = Lambda[np.ix_(ran_in,ran_in)]
        Lambda_oi = Lambda[np.ix_(ran_out,ran_in)]
        Lambda_oo = Lambda[np.ix_(ran_out,ran_out)]


        # Convert input values to correct values:
        val = man_in.swapto_listoftuple(val)
            
        condres = []
        for x_i in val:
            x_o = mu_o # Initial guess
            it=0; diff = 1
            while (diff > 1e-6):
                # Parallel transport of only the output:
                Ro = man_out.parallel_transport(np.eye(man_out.n_dimT), mu_o, x_o).T
                lambda_oi = Ro.dot(Lambda_oi)
                lambda_oo = Ro.dot(Lambda_oo.dot(Ro.T))

                # Compute update
                delta = (man_out.log(mu_o, x_o) 
                         - man_in.log(x_i,mu_i).dot( (np.linalg.inv(lambda_oo).dot(lambda_oi)).T )  )
                x_o   = man_out.exp(delta, x_o)

                diff = sum(delta*delta)
                # Max iterations
                it+=1
                if it >50:
                    logger.warning(
                        'Conditioning did not converge in {0} its, Delta: {1}'.format(it, delta))
                    break
                    
            sigma_xo = np.linalg.inv(lambda_oo)
            condres.append( Gaussian(man_out, x_o, sigma_xo) )
        
        # Return result depending input value:
        if len(condres)==1:
            return condres[-1]
        else:
            return condres       
            

    # @logger.contextualize(filter=False)
    def __mul__(self,other):
        '''Compute the product of Gaussian''' 
        max_it      = 50
        conv_thresh = 1e-5

        # Get manifolds:
        man = self.manifold
        Log = man.log
        Exp = man.exp

        # Function for covariance transport
        fR = lambda g,h: man.parallel_transport(np.eye(man.n_dimT), g, h).T
        
        sigma_s = self.sigma
        sigma_o = other.sigma

        # Decomposition of covariance:
        try:
            lambda_s = np.linalg.inv(sigma_s)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix, adding diag constant", filter=False)
            lambda_s = np.linalg.inv(sigma_s + np.eye(sigma_s.shape[0])*1e-20)
        try:
            lambda_o = np.linalg.inv(sigma_o)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix, adding diag constant", filter=False)
            lambda_o = np.linalg.inv(sigma_o + np.eye(sigma_o.shape[0])*1e-20)
       
        mu  = self.mu # Initial guess
        it=0; diff = 1
        while (diff > conv_thresh):
            # Transport precision to estimated mu
            Rs = fR(self.mu, mu)
            lambda_sn = Rs.dot( lambda_s.dot( Rs.T) )
            Ro = fR(other.mu, mu)
            lambda_on = Ro.dot( lambda_o.dot( Ro.T) )

            # Compute new covariance:
            try:
                sigma = np.linalg.inv( lambda_sn + lambda_on)  # TODO: add regularization?
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix, adding diag constant", filter=False)
                sigma = np.linalg.inv( lambda_sn + lambda_on + np.eye(lambda_sn.shape[0])*1e-20 )

            # Compute weighted distances:
            d_self  = lambda_sn.dot( Log(self.mu , mu) )
            d_other = lambda_on.dot( Log(other.mu, mu) )

            # update mu:
            delta = sigma.dot(d_self + d_other)
            mu = Exp(delta+0e-4,mu)  # TODO: Should this be 1e-4?

            # Handle convergence
            diff = sum(delta*delta)
            it+=1
            if it >max_it:
                logger.warning(
                    'Product did not converge in {0} iterations.'.format(max_it))
                break

        return Gaussian(self.manifold, mu,sigma)

    def tangent_action(self,A):
        ''' Perform Transformation A, to the tangent space of the Gaussian'''
        # Apply A, to covariance
        self.sigma = A.dot(self.sigma.dot(A.T))

    def parallel_transport(self, h):
        ''' Move Gaussian to h using parallel transport.'''
        man = self.manifold

        # Compute rotation for covariance matrix:
        R = man.parallel_transport(np.eye(man.n_dimT), self.mu, h).T
        self.sigma = R.dot( self.sigma.dot(R.T) )
        self.mu = h

    def inv_trans_s(self, A, b):
        # print(A)
        # print(b)
        # print(A.shape)
        # print(b.shape)
        raise NotImplementedError
        # for compatibility with pbdlib's LQR
        model = self.copy()
        # inverse transform of homogegenous_trans
        model.parallel_transport(-b)
        model.tangent_action(A.T)

        return model

    def plot_2d(self, ax, base=None, ix=0, iy=1,**kwargs):
        ''' Plot Gaussian'''

        if base is None:
            base = self.manifold.id_elem
            mu = self.manifold.log(self.mu, base)[ [ix,iy] ]
            sigma = self.sigma[ [ix, iy], :][:, [ix, iy] ]
        else:
            # Mu:
            mu = self.manifold.log(self.mu, base)[ [ix,iy] ]

            # Select elements
            sigma = self.sigma[ [ix, iy], :][:, [ix, iy] ]

        return fctplt.GaussianPatch2d(ax=ax,mu=mu, sigma=sigma, **kwargs)
        #return fctplt.plot_gaussian_2d(mu, sigma, ax=ax, **kwargs)
        
    def plot_3d(self, ax=None, base=None, ix=0, iy=1, iz=2, **kwargs):
        ''' Plot Gaussian'''

        if base is None:
            base = self.manifold.id_elem
            
        mu = self.manifold.log(self.mu, base)[ [ix,iy,iz] ]
        sigma = self.sigma[ [ix, iy, iz], :][:, [ix, iy, iz] ]

        return fctplt.GaussianPatch3d(ax=ax, mu=mu, sigma=sigma, **kwargs)

    def get_mu_sigma(self, base=None, idx=None, mu_on_tangent=True,
                     as_np=False, raw_tuple=False):
        if base is None:
            base = self.manifold.id_elem

        if mu_on_tangent:
            mu = self.manifold.log(self.mu, base, stack=not raw_tuple)
        elif as_np:
            mu = list(self.mu)
            for i, m in enumerate(mu):
                if type(m) is not np.ndarray:
                    mu[i] = m.to_nparray()
            mu = np.concatenate(mu)
        else:
            mu = self.mu

        sigma = self.sigma
        if idx is not None:
            mu = mu[ [*idx] ]
            sigma = sigma[ [*idx], :][:, [*idx] ]

        return mu, sigma

    def copy(self):
        '''Get copy of Gaussian'''
        g_copy = Gaussian(self.manifold, deepcopy(self.mu),deepcopy(self.sigma))
        return g_copy

    def save(self,name):
        '''Write Gaussian parameters to files: name_mu.txt, name_sigma.txt'''
        np.savetxt('{0}_mu.txt'.format(name),self.manifold.manifold_to_np(self.mu) )
        np.savetxt('{0}_sigma.txt'.format(name),self.sigma)

    def sample(self):
        A = np.linalg.cholesky(self.sigma)
        samp = A.dot(np.random.randn(self.manifold.n_dimT))
        return self.manifold.exp(samp,self.mu)

    @staticmethod
    def load(name,manifold):
        '''Load Gaussian parameters from files: name_mu.txt, name_sigma.txt'''
        try:
            mu    = np.loadtxt('{0}_mu.txt'.format(name))
            sigma = np.loadtxt('{0}_sigma.txt'.format(name))
        except Exception as err:
            print('Was not able to load Gaussian {0}.txt:'.format(name,err))

        try:
            mu = manifold.np_to_manifold(mu)
        except Exception as err:
            print('Specified manifold is not compatible with loaded mean: {0}'.format(err))
        return Gaussian(manifold, mu,sigma)

    def __key(self):
        return (
            self.manifold, array_to_tuple(self.mu), array_to_tuple(self.sigma))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Gaussian):
            return self.__key() == other.__key()
        return NotImplemented


class GMM:
    ''' Gaussian Mixture Model class based on sci-learn GMM.
    This child implements additional initialization algorithms

    '''
    def __init__(self, manifold, n_components, base=None):
        '''Create GMM'''
        self.manifold = manifold

        self.set_n_components(n_components, warn=False)

        if base is None:
            self.base = manifold.id_elem
        else:
            self.base = base

        self._last_reg_type = None

    def set_n_components(self, n_components, warn=True):
        logger.info(f"Changing number of components to {n_components}")

        if warn:
            logger.warning("Changing number of components resets the GMM.")

        mu0 = self.manifold.id_elem
        sigma0 = np.eye(self.manifold.n_dimT)

        self.n_components = n_components
        self.priors = np.ones(n_components)/n_components

        self.gaussians = []
        for _ in range(n_components):
            self.gaussians.append( Gaussian(self.manifold, mu0, sigma0) )

    def expectation(self,data):
        # Expectation:
        lik = []
        for i,gauss in enumerate(self.gaussians):
              lik.append(gauss.prob(data)*self.priors[i])
        lik = np.vstack(lik)
        return lik

    def predict(self, data):
        '''Classify to which datapoint each kernel belongs'''
        lik = self.expectation(data)
        return np.argmax(lik, axis=0)

    # @logger.contextualize(filter=False)
    def fit(self, data, convthres=1e-5, maxsteps=100, minsteps=5, reg_lambda=1e-3,
            reg_type= RegularizationType.SHRINKAGE):
        '''Initialize trajectory GMM using a time-based approach'''
            
        # Make sure that the data is a tuple of list:
        data = self.manifold.swapto_tupleoflist(data)
        n_data = len(data)
        
        prvlik = 0
        avg_loglik = []
        for st in range(maxsteps):
            # Expectation:
            lik = self.expectation(data)
            gamma0 = (lik/ (lik.sum(axis=0) + 1e-200) )# Sum over states is one
            gamma1 = (gamma0.T/gamma0.sum(axis=1)).T # Sum over data is one

            # Maximization:
            # - Update Gaussian:
            for i,gauss in enumerate(self.gaussians):
                gauss.mle(data,gamma1[i,], reg_lambda, reg_type)
            # - Update priors: 
            self.priors = gamma0.sum(axis=1)   # Sum probabilities of being in state i
            self.priors = self.priors/self.priors.sum() # Normalize

            # Check for convergence:
            avglik = -np.log(lik.sum(0)+1e-200).mean()
            if abs(avglik - prvlik) < convthres and st > minsteps:
                logger.info('EM converged in %i steps'%(st))
                break
            else:
                avg_loglik.append(avglik)
                prvlik = avglik
        if (st+1) >= maxsteps:
             logger.info('EM did not converge in {0} steps'.format(maxsteps))

        self._last_reg_type = reg_type
            
        return lik, avg_loglik

    # @logger.contextualize(filter=False)
    def fit_from_np(self, npdata, convthres=1e-5, maxsteps=100, minsteps=5, reg_lambda=1e-3, 
                    reg_lambda2=1e-3, reg_type= RegularizationType.SHRINKAGE,
                    plot=False, fix_last_component=False, fix_first_component=False):
        '''Initialize trajectory GMM using a time-based approach'''

        data = self.manifold.np_to_manifold(npdata)

        # Make sure that the data is a tuple of list:
        n_data = len(data)

        gaussians = self.gaussians

        if fix_last_component:
            gaussians = gaussians[:-1]
        if fix_first_component:
            gaussians = gaussians[1:]

        offset = 1 if fix_first_component else 0
        
        prvlik = 0
        avg_loglik = []
        for st in tqdm(range(maxsteps), desc='EM'):
            # Expectation:
            lik = self.expectation(data)
            gamma0 = (lik/ (lik.sum(axis=0) + 1e-200) )# Sum over states is one
            gamma1 = (gamma0.T/gamma0.sum(axis=1)).T # Sum over data is one

            # Maximization:
            # - Update Gaussian:
            for i,gauss in enumerate(gaussians):
                gauss.mle(data,gamma1[i+offset,], reg_lambda, reg_lambda2, reg_type,
                          plot_process=plot)
            # - Update priors: 
            self.priors = gamma0.sum(axis=1)   # Sum probabilities of being in state i
            self.priors = self.priors/self.priors.sum() # Normalize

            # NOTE: for debugging
            # print(self.mu)

            # Check for convergence:
            avglik = -np.log(lik.sum(0)+1e-200).mean()
            if abs(avglik - prvlik) < convthres and st > minsteps:
                logger.info('EM converged in %i steps'%(st))
                break
            else:
                avg_loglik.append(avglik)
                prvlik = avglik
        if (st+1) >= maxsteps:
             logger.warning('EM did not converge in {0} steps'.format(maxsteps))

        self._last_reg_type = reg_type

        return lik, avg_loglik

    def bci_from_lik(self, lik):
        # lik is the likelihood per state, weighted by the priors
        # so aggregate over states
        return -2 * np.log(lik.sum(0)+1e-200).mean() * lik.shape[1] + \
            self._n_parameters * np.log(lik.shape[1])
    
    @property
    def _n_parameters(self, reg_type=None):

        if reg_type is None:
            reg_type = self._last_reg_type

        n_features = self.mu.shape[1]
        n_components = self.n_components

        mean_params = self.n_components * n_features

        if reg_type in [RegularizationType.NONE,
                        RegularizationType.SHRINKAGE,
                        RegularizationType.DIAGONAL,
                        RegularizationType.COMBINED,
                        RegularizationType.ADD_CONSTANT]:
            cov_params = n_components * n_features * (n_features + 1) / 2.
        elif reg_type == RegularizationType.DIAGONAL_ONLY:
            cov_params = n_components * n_features
        elif reg_type is None:
            raise ValueError("Regularization type not set. Did perform fit?"
                             "In fitting calls, uses RegularizationType.NONE "
                             "instead of None!")
        else:
            raise ValueError('Unkown regularization type.')

        return self.n_components + mean_params + cov_params - 1

    def init_time_based(self,t,data, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):

        if t.ndim==2:
            t = t[:,0] # Drop last dimension


        # Ensure data is tuple of lists
        data = self.manifold.swapto_tupleoflist(data)
        
        # Timing seperation:
        timing_sep = np.linspace(t.min(), t.max(),self.n_components+1)
        man = self.gaussians[0].manifold
        npdata = man.manifold_to_np(data)
        
        for i, g in enumerate(self.gaussians):
            # Select elements:
            idtmp = (t>=timing_sep[i])*(t<timing_sep[i+1]) 
            sl =  np.ix_( idtmp, range(npdata.shape[1]) )
            tmpdata = self.manifold.np_to_manifold( npdata[sl] )

            # Perform mle:
            g.mle(tmpdata, reg_lambda=reg_lambda, reg_type=reg_type)
            self.priors[i] = len(idtmp)
        self.priors = self.priors / self.priors.sum()

        self._last_reg_type = reg_type


    def init_time_based_from_np(self, npdata, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE, plot=False,
                                drop_time=False, fix_first_component=False, fix_last_component=False,
                                fixed_first_component_n_steps=2, fixed_last_component_n_steps=2):
        # Assuming that data is first manifold dimension
        t = npdata[:,0]

        if drop_time:
            npdata = npdata[:,1:]

        t_delta = t[1] - t[0]

        t_min = t.min()
        t_max = t.max()

        t_start = t_min + (fixed_first_component_n_steps * t_delta) if fix_first_component else t_min
        t_stop = t_max - (fixed_last_component_n_steps * t_delta) if fix_last_component else t_max

        n_dyn_components = self.n_components - int(fix_first_component) - int(fix_last_component)

        # Timing seperation:
        timing_sep = np.linspace(t_start, t_stop, n_dyn_components+1)

        if fix_first_component:
            timing_sep = np.concatenate(([t_min], timing_sep))

        if fix_last_component:
            timing_sep = np.concatenate([timing_sep, [t_max+t_delta]])

        # print(timing_sep)

        for i, g in tqdm(enumerate(self.gaussians), desc='Time-based init',
                         total=self.n_components):
            if plot:
                logger.info('Fitting GMM component %i/%i'%(i+1, self.n_components),
                            filter=False)
            # Select elements:
            idtmp = (t>=timing_sep[i])*(t<timing_sep[i+1])
            sl =  np.ix_( idtmp, range(npdata.shape[1]) )

            tmpdata = self.manifold.np_to_manifold( npdata[sl] )

            # from lovely_numpy import lo
            # # print(len(tmpdata))
            # # print(len(tmpdata[0]), type(tmpdata[0]))
            # # print(lo(tmpdata[0]))
            # # print(lo(tmpdata[-1]))
            # print('==================')
            # print(npdata[sl][:, 11:])
            # print(tmpdata[-1])

            # print(len(tmpdata), tmpdata[0].shape)
            # print(sl)
            # raise KeyboardInterrupt

            # Perform mle:
            g.mle(tmpdata, reg_lambda=reg_lambda, reg_type=reg_type,
                  plot_process=plot)
            self.priors[i] = len(idtmp)
        self.priors = self.priors / self.priors.sum()

        self._last_reg_type = reg_type

        data = self.manifold.np_to_manifold(npdata)
        lik = self.expectation(data)
        avglik = -np.log(lik.sum(0)+1e-200).mean()

        return lik, avglik, timing_sep


    def sammi_init(self, npdata, includes_time=False, max_local_components=3,
                   debug_borders=False, plot_cb=None, debug_multimodal=False,
                   em_kwargs=None, kmeans_kwargs=None,
                   fixed_component_from_last_step=False,
                   fixed_component_from_first_step=False,
                   fixed_component_n_steps=2):

        from scipy.ndimage import gaussian_filter1d
        from sklearn.cluster import DBSCAN

        def filter_zeromask(array):
            # Filter out consecutive zeros. Only keep leftmost and rightmost.
            filtered = np.copy(array)
            D, TR, TI = array.shape
            for d in range(D):
                for tr in range(TR):
                    for t in range(TI):
                        if array[d, tr, t-2] and array[d, tr, t]:
                            filtered[d, tr, t-1] = False

            return filtered
        
        def get_cluster_means(data, labels):
            means = []

            for l in np.unique(labels):
                if l == -1:
                    continue
                means.append(np.mean(np.concatenate(data)[labels==l], axis=0))

            return means

        def get_components(log_data, local_eps=2, local_min_samples=2,
                           global_eps=3, global_min_samples=3,
                           edge_margin=9, zero_thresh=0.001):
            # zp_data = np.stack([log_data[d] for d in dims], axis=0)
            zero_mask = np.abs(log_data) < zero_thresh
            filtered = filter_zeromask(zero_mask)

            # Cluster zero points per dimension
            dbs = DBSCAN(eps=local_eps, min_samples=local_min_samples)
            dim_zeroes = [np.concatenate([np.argwhere(tr) for tr in d])
                        for d in filtered]
            dim_cluster_labels = [dbs.fit_predict(d) for d in dim_zeroes]

            # Compute the mean of all clusters
            dim_cluster_means = []
            for dim in range(len(dim_zeroes)):
                dim_cluster_means.append(
                    get_cluster_means(dim_zeroes[dim], dim_cluster_labels[dim]))

            # Cluster the means of all dimensions
            dbs = DBSCAN(eps=global_eps, min_samples=global_min_samples)

            global_zero_labels = dbs.fit_predict(
                np.concatenate(dim_cluster_means).reshape(-1, 1))
            global_zero_means = get_cluster_means(
                dim_cluster_means, global_zero_labels)

            global_zero_means = sorted(global_zero_means)

            # Filter out borders that are too close to the edges
            while global_zero_means[0] < edge_margin:
                global_zero_means = global_zero_means[1:]
            while global_zero_means[-1] > n_time_steps - edge_margin:
                global_zero_means = global_zero_means[:-1]

            return global_zero_means, dim_cluster_means

        def fit_multimodal_components(local_data, max_components):

            candidate_gmms = []
            bci_scores = []
            for i in range(1, max_components + 1):
                candidate = GMM(self.manifold, n_components=i,
                                base=self.base)
                candidate.kmeans_from_np(local_data, **kmeans_kwargs)
                b, _ = candidate.fit_from_np(local_data, **em_kwargs)
                if debug_multimodal:
                    plot_cb(plot_traj=True, plot_gaussians=True,
                            model=candidate)
                candidate_gmms.append(candidate)
                bci_scores.append(candidate.bci_from_lik(b))

            inc_idx = np.argmin(bci_scores)
            incumbent = candidate_gmms[inc_idx]

            bci_str = ', '.join(
                [f'{c+1}: {bci:.2f}' for c, bci in enumerate(bci_scores)])
            logger.info(f'BCI scores: {bci_str}')
            logger.info(f'Selecting incumbent: {inc_idx+1} components.')

            return incumbent

        n_time_steps = npdata.shape[1]

        logger.info('Estimating component borders ...')
        log_data = np.stack(
            [self.np_to_manifold_to_np(traj) for traj in npdata])

        grad = np.gradient(log_data, axis=1).transpose(2, 0, 1)
        grad = gaussian_filter1d(grad, 2)

        if includes_time:
            # orig = log_data.transpose(2, 0, 1)[1:]
            grad = grad[1:]

        borders, dim_borders = get_components(grad)

        if debug_borders:
            fctplt.plot_component_time_series(grad, (24, 20), borders,
                                              dim_borders)

        borders = [0] + borders

        if fixed_component_from_first_step:
            borders = [fixed_component_n_steps] + borders
        if fixed_component_from_last_step:
            borders.append(n_time_steps - 1 - fixed_component_n_steps)

        borders.append(n_time_steps - 1)

        t = np.arange(n_time_steps)

        component_gmms = []
        global_priors = []

        logger.info('Estimating component  modalities ...')
        for i in tqdm(range(len(borders) - 1), desc='Fitting components'):
            # Select elements:
            idtmp = (t>=borders[i])*(t<borders[i+1])
            sl =  np.ix_(range(npdata.shape[0]), idtmp, range(npdata.shape[2]))
            local_data = npdata[sl].reshape(-1, npdata.shape[2])
            
            if (fixed_component_from_last_step and i == len(borders) - 2) or \
                (fixed_component_from_first_step and i == 0):
                n_max_components = 1  # fixed component should be unimodal
            else:
                n_max_components = max_local_components

            component_gmms.append(
                fit_multimodal_components(local_data, n_max_components))
            global_priors.append(len(idtmp))

        self.gaussians = [g for c in component_gmms for g in c.gaussians]
        self.n_components = len(self.gaussians)

        # join global and local priors
        global_priors = np.array(global_priors) / np.sum(global_priors)
        self.priors = [gp*lp for c, gp in zip(component_gmms, global_priors)
                       for lp in c.priors]

        self._last_reg_type = em_kwargs['reg_type']

        data = self.manifold.np_to_manifold(npdata.reshape(-1, npdata.shape[2]))
        lik = self.expectation(data)
        avglik = -np.log(lik.sum(0)+1e-200).mean()

        return lik, avglik, borders

    # @logger.contextualize(filter=False)
    def kmeans(self,data, maxsteps=100,reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE ):

        # Make sure that data is tuple of lists
        data = self.manifold.swapto_tupleoflist(data)

        # Init means
        npdata = self.manifold.manifold_to_np(data)
        n_data = npdata.shape[0]

        id_tmp = np.random.permutation(n_data)
        for i, gauss in enumerate(self.gaussians):
            gauss.mu = self.manifold.np_to_manifold( npdata[id_tmp[i],:])
        
        dist = np.zeros( (n_data, self.n_components) )
        dist2 = np.zeros( (n_data, self.n_components) )
        id_old = np.ones(n_data) + self.n_components
        for it in range(maxsteps):
            # E-step 
            # batch:
            for i, gauss in enumerate(self.gaussians):
                dist[:,i] = (gauss.manifold.log(data, gauss.mu)**2).sum(axis=1)
            # single::
            #    for n in range(n_data):
            #        dist[n,i] = sum(gauss.manifold.log(gauss.mu, data[n])**2)
            id_min = np.argmin(dist, axis=1)        

            # M-step
            # Batch:
            for i, gauss in enumerate(self.gaussians):
                sl = np.ix_( id_min==i , range(npdata.shape[1]))
                dtmp = gauss.manifold.np_to_manifold(npdata[sl])
                gauss.mle(dtmp, reg_lambda=reg_lambda, reg_type=reg_type)

            #for i, gauss in enumerate(self.gaussians):
            #    tmp = [data[j] for j, x in enumerate(id_min) if x == i]
            #    gauss.mle(tmp)
            #    self.priors[i] = len(tmp)
            self.priors = self.priors/sum(self.priors)

            # Stopping criteria:
            if (sum(id_min != id_old) == 0):
                # No datapoint changes:
                logger.info('K-means converged in {0} iterations'.format(it))
                break;
            else:
                id_old = id_min

        self._last_reg_type = reg_type


    # @logger.contextualize(filter=False)
    def kmeans_from_np(self,npdata, maxsteps=100,reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE, plot=False, fix_first_component=False, fix_last_component=False):

        if fix_first_component or fix_last_component:
            print(npdata.shape)
            raise NotImplementedError

        data = self.manifold.np_to_manifold(npdata)
        n_data = npdata.shape[0]

        id_tmp = np.random.permutation(n_data)
        for i, gauss in enumerate(self.gaussians):
            gauss.mu = self.manifold.np_to_manifold( npdata[id_tmp[i],:])
        
        dist = np.zeros( (n_data, self.n_components) )
        dist2 = np.zeros( (n_data, self.n_components) )
        id_old = np.ones(n_data) + self.n_components
        for it in tqdm(range(maxsteps), desc='K-means'):
            # E-step 
            # batch:
            for i, gauss in enumerate(self.gaussians):
                dist[:,i] = (gauss.manifold.log(data, gauss.mu)**2).sum(axis=1)
            # single::
            #    for n in range(n_data):
            #        dist[n,i] = sum(gauss.manifold.log(gauss.mu, data[n])**2)
            id_min = np.argmin(dist, axis=1)        

            # M-step
            # Batch:
            for i, gauss in enumerate(self.gaussians):
                sl = np.ix_( id_min==i , range(npdata.shape[1]))
                dtmp = gauss.manifold.np_to_manifold(npdata[sl])
                gauss.mle(dtmp, reg_lambda=reg_lambda, reg_type=reg_type,
                          plot_process=plot)

            #for i, gauss in enumerate(self.gaussians):
            #    tmp = [data[j] for j, x in enumerate(id_min) if x == i]
            #    gauss.mle(tmp)
            #    self.priors[i] = len(tmp)
            self.priors = self.priors/sum(self.priors)

            # Stopping criteria:
            if (sum(id_min != id_old) == 0):
                # No datapoint changes:
                logger.info('K-means converged in {0} iterations'.format(it))
                break;
            else:
                id_old = id_min

        self._last_reg_type = reg_type

        lik = self.expectation(data)
        avglik = -np.log(lik.sum(0)+1e-200).mean()

        return lik, avglik, None

    def log_from_np(self, npdata, base=None):
        data = self.manifold.np_to_manifold(npdata)

        # TODO: need to use (respective) mu?
        if base is None:
            base = self.base

        proj = self.manifold.log(data, base)

        return proj

    def tangent_action(self,A):
        ''' Perform A to the tangent space of the GMM 
        At   : Tangent space transformation matrix (e.g. Rotation matrix)
        
        '''
        man = self.gaussians[0].manifold

        # Apply A:
        # Get mu's in the tangent space
        mu_tan = np.zeros( (self.n_components, man.n_dimT)) 
        for i, g in enumerate(self.gaussians):
            mu_tan[i,:] = man.log(g.mu, self.base)
            
        # Apply A to change location of means:
        mu_tan  = mu_tan.dot(A.T) 
        mus_new = man.exp(mu_tan, self.base)

        # Update gaussian
        mus_new = man.swapto_listoftuple(mus_new)
        for i, _ in enumerate(self.gaussians):
            self.gaussians[i].mu = mus_new[i]
            self.gaussians[i].tangent_action(A)


    def parallel_transport(self, h):
        ''' Parallel transport GMM from current base to h'''
        
        man = self.gaussians[0].manifold
        # Compute new mu's
        # Collect mu's and put them in proper data structure
        mulist = []
        for i, g in enumerate(self.gaussians):
            mulist.append(g.mu)

        # Perform action:
        mulist = man.swapto_tupleoflist(mulist)
        mus_new = man.action(mulist, self.base, h)
        self.base = h # Set new base for GMM

        mus_new = man.swapto_listoftuple(mus_new)
        for i,_ in enumerate(self.gaussians):
            # assign new base
            # To compensate for the mis-alignment of the tangent bases cause by 
            # moving the origin of the GMM from e to h, we need to apply a correction each
            # Gaussian. 
            # We do this by parallel transporting each Gaussian from h (the new base), to the new
            # mean location. 
            self.gaussians[i].mu = self.base 
            self.gaussians[i].parallel_transport(mus_new[i])

    def homogeneous_trans(self, A, b):
        model = self.copy()
        model.tangent_action(A)  # Apply A in tangent space of origin.
        # TODO: is using b as argument correct?
        # mu: should be mapped from e to e+b?
        # Sigma: ???
        model.parallel_transport(b)   # Move origin to new mean.

        return model
       
    def margin(self, i_in):
        # Construct new GMM:
        newgmm = type(self)(self.manifold.get_submanifold(i_in), self.n_components)

        # Copy priors
        newgmm.priors = self.priors

        # Add marginalized Gaussian
        for i,gauss in enumerate(self.gaussians):
            newgmm.gaussians[i] = gauss.margin(i_in)
        return newgmm
            
    
    def copy(self):
        copygmm = GMM(self.manifold, self.n_components)
        for i,gauss in enumerate(self.gaussians):
            copygmm.gaussians[i] = gauss.copy()
        copygmm.priors = deepcopy(self.priors)
        copygmm.base = deepcopy(self.base)
        return copygmm

    def __mul__(self,other):
        prodgmm = self.copy()

        for i, _ in enumerate(prodgmm.gaussians):
            prodgmm.gaussians[i] = self.gaussians[i]*other.gaussians[i]
            
        return prodgmm

    def gmr_from_np(self, npdata_in, i_in=0, i_out=1, initial_obs=None):
        data = self.manifold.np_to_manifold(npdata_in, dim=i_in)
        return self.gmr(data, i_in, i_out)

    # @logger.contextualize(filter=False)
    def gmr(self ,data_in, i_in=0, i_out=1):
        '''Perform Gaussian Mixture Regression
        x    : liar of elements on the input manifold
        i_in : Index of input manifold
        i_out: Index of output manifold
        '''
        m_out  = self.gaussians[0].manifold.get_submanifold(i_out)
        m_in   = self.gaussians[0].manifold.get_submanifold(i_in)
        
        # Check swap input data to list:
        ldata_in = m_in.swapto_listoftuple(data_in)
        n_data = len(ldata_in)
            
        # Compute weights:
        h = np.zeros( (n_data,self.n_components) )
        h = self.margin(i_in).expectation(data_in)
        h = (h/h.sum(0)).T # Normalize w.r.t states
        
        gmr_list = []
        for n, point in enumerate(ldata_in):
            # Compute conditioned elements:
            gc_list = []
            muc_list = []
            for i, gauss in enumerate(self.gaussians):
                # Only compute conditioning for states that have a significant influence
                # Do this to prevent convergence errors
                if h[n,i]>1e-5:
                    # weight is larger than 1e-3
                    gc_list.append(gauss.condition(point, i_in=i_in, i_out=i_out))
                else:
                    # No signnificant weight, just store the margin 
                    gc_list.append(gauss.margin(i_out))
                # Store the means in np array:
                muc_list.append(gc_list[-1].mu)

            # Convert to tuple of  lists, such that we can apply log and exp to them
            # TODO: This should be done by the log and exp functions of the manifold
            muc_list = m_out.swapto_tupleoflist(muc_list) 
            
            
            # compute Mu of GMR:
            mu_gmr = deepcopy(gc_list[0].mu)
            diff = 1.0; it = 0
            
            diff == 1

            while (diff > 1e-5):
                delta = h[n,:].dot( m_out.log(muc_list, mu_gmr) )
                mu_gmr = m_out.exp(delta, mu_gmr)
                
                # Compute update difference
                diff = (delta*delta).sum()
                
                if it>50:
                    logger.warning('GMR did not converge in {0} iterations.'.format(n))
                    break;
                it+=1
                    
             # Compute covariance: 
            sigma_gmr = np.zeros( (m_out.n_dimT, m_out.n_dimT))
            for i in range(self.n_components):
                R = m_out.parallel_transport(np.eye(m_out.n_dimT), 
                                            gc_list[i].mu, mu_gmr).T

                dtmp = np.zeros(m_out.n_dimT)[:,None] #m_out.log(gc_list[i].mu, mu_gmr)[:,None] #
                sigma_gmr += h[n,i]*( R.dot(gc_list[i].sigma.dot(R.T))
                                     + dtmp.dot(dtmp.T) 
                                    )           
            # Compute covariance: 
            #sigma_gmr = np.zeros( (m_out.n_dimT, m_out.n_dimT))
            #for i in range(self.n_components):
            #    dtmp = m_out.log(gc_list[i].mu, mu_gmr)[:,None]
            #    sigma_gmr += h[n,i]*(  gc_list[i].sigma 
            #                         + dtmp.dot(dtmp.T) 
            #                        )

            gmr_list.append( Gaussian(m_out, mu_gmr, sigma_gmr))
            
        return gmr_list

    def plot_2d(self, ax, base=None, ix=0, iy=1, **kwargs):
        ''' Plot Gaussian'''

        if base is None:
            base = self.manifold.id_elem
            
        l_list = fctplt.GaussianGraphicList()
        for _,gauss in enumerate(self.gaussians):
            l_list.append( gauss.plot_2d(base=base,ax=ax,ix=ix,iy=iy,**kwargs) )

        return l_list
        
    def plot_3d(self, ax, base=None, ix=0, iy=1, iz=2, **kwargs):
        ''' Plot Gaussian'''

        if base is None:
            base = self.manifold.id_elem
            
        l_list =fctplt.GaussianGraphicList()
        for _,gauss in enumerate(self.gaussians):
            l_list.append( gauss.plot_3d(base=base,ax=ax,ix=ix,iy=iy, iz=iz, **kwargs) )

        return l_list

    def get_mu_sigma(self, base=None, idx=None, stack=False, mu_on_tangent=True,
                     as_np=False, raw_tuple=False):
        comp = [g.get_mu_sigma(base=base, idx=idx, mu_on_tangent=mu_on_tangent,
                               as_np=as_np, raw_tuple=raw_tuple)
                for g in self.gaussians]

        mu = [c[0] for c in comp]

        sigma = [c[1] for c in comp]

        if stack:
            assert as_np
            mu = np.stack(mu)
            sigma = np.stack(sigma)
        else:
            mu = tuple(mu)
            sigma = tuple(sigma)

        return mu, sigma

    @property
    def mu(self):  # for compatibility with pbdlib
        mu, _ = self.get_mu_sigma(stack=True, as_np=True)
        return mu

    @property
    def mu_raw(self):
        mu, _ = self.get_mu_sigma(stack=False, as_np=False, raw_tuple=True,
                                  mu_on_tangent=False)
        return mu

    @property
    def mu_tangent(self):
        mu, _ = self.get_mu_sigma(stack=False, as_np=False, raw_tuple=True,
                                  mu_on_tangent=True)
        return mu

    @property
    def sigma(self):
        _, sigma = self.get_mu_sigma(stack=True, as_np=True)
        return sigma

    def np_to_manifold_to_np(self, npdata_in, i_in=None, base=None):
        data = self.manifold.np_to_manifold(npdata_in, dim=i_in)
        # if base is None:
        #     base = self.base
        return self.manifold.log(data, base, dim=i_in)

    def save(self,name):
        for i,g in enumerate(self.gaussians):
            g.save('{0}{1}'.format(name,i) )
        np.savetxt('{0}_priors.txt'.format(name),self.priors)

    @staticmethod
    def load(name,n_components,manifold):
        mygmm = GMM(manifold, n_components)
        for i in range(n_components):
            tmpg = Gaussian.load('{0}{1}'.format(name,i),manifold)
            mygmm.gaussians[i]=tmpg
        mygmm.priors = np.loadtxt('{0}_priors.txt'.format(name))
        return mygmm

    def key(self):
        return self.__key()

    def __key(self):
        return (self.manifold.name, hash(array_to_tuple(self.priors)),
                hash(tuple(self.gaussians)))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, GMM):
            return self.__key() == other.__key()
        return NotImplemented

    # Aliases for compatibility with pbdlib
    nb_states = Alias("n_components")
    # def marginal_model(self, slc):
    #     idx = list(range(slc.stop)[slc])
    #     print(idx)
    #     print(self.gaussians)
    #     return self.margin(idx)

    def concatenate_gaussian(self, q, get_mvn=True, reg=None):
        logger.warning("Code not tested yet.")
        # print(q.shape)
        # manis = [g.manifold for g in gaussians]
        # joint_manifold = multiply_iterable(manis)
        # print(type(self.gaussians[0].mu), type(self.gaussians[0].sigma))
        # print(len(self.gaussians[0].mu), self.gaussians[0].sigma.shape)
        # raise KeyboardInterrupt
        # joint_mu = np.concatenate([g.mu for g in gaussians])
        # mvn = MVNRBD(joint_manifold, joint_mu, joint_sigma)

        if reg is None:
            if not get_mvn:
                # return np.concatenate([self.mu[i] for i in q]), block_diag(*[self.sigma[i] for i in q])
                raise NotImplementedError
            else:
                mani = multiply_iterable([self.manifold for _ in q])
                mus = tuple([self.mu[i] for i in q])
                sigmas = block_diag(*[self.sigma[i] for i in q])

                mvn = Gaussian(mani, mu=mus, sigma=sigmas)
                # mvn.mu = np.concatenate([self.mu[i] for i in q])
                # mvn._sigma = block_diag(*[self.sigma[i] for i in q])
                # mvn._lmbda = block_diag(*[self.lmbda[i] for i in q])

                return mvn
        else:
            raise NotImplementedError

            # if not get_mvn:
            #     return np.concatenate([self.mu[i] for i in q]), block_diag(
            #         *[self.sigma[i] + reg for i in q])
            # else:
            #     mvn = MVN()
            #     mvn.mu = np.concatenate([self.mu[i] for i in q])
            #     mvn._sigma = block_diag(*[self.sigma[i] + reg for i in q])
            #     mvn._lmbda = block_diag(*[np.linalg.inv(self.sigma[i] + reg) for i in q])

            # return mvn

class HMM(GMM):
    def __init__(self, manifold, n_components, base=None):
        GMM.__init__(self, manifold, n_components, base=base)

        self._trans = None
        self._init_priors = None

    @property
    def init_priors(self):
        if self._init_priors is None:
            logger.info("HMM init priors not defined, initializing to uniform")
            self._init_priors = np.ones(self.nb_states) / self.nb_states

        return self._init_priors
    
    @init_priors.setter
    def init_priors(self, value):
        self._init_priors = value

    @property
    def trans(self):
        if self._trans is None:
            logger.info("HMM transition matrix not defined, initializing to uniform")
            self._trans = np.ones((self.nb_states, self.nb_states)) / self.nb_states
        return self._trans

    @trans.setter
    def trans(self, value):
        self._trans = value


    @property
    def Trans(self):
        return self.trans

    @Trans.setter
    def Trans(self, value):
        self.trans = value

    def make_finish_state(self, demos, dep_mask=None):
        raise NotImplementedError
        self.has_finish_state = True
        self.nb_states += 1

        data = np.concatenate([d[-3:] for d in demos])

        mu = np.mean(data, axis=0)

        # Update covariances
        if data.shape[0] > 1:
            sigma = np.einsum('ai,aj->ij',data-mu, data-mu)/(data.shape[0] - 1) + self.reg
        else:
            sigma = self.reg

        # if cov_type == 'diag':
        # 	self.sigma *= np.eye(self.nb_dim)

        if dep_mask is not None:
            sigma *= dep_mask

        self.mu = np.concatenate([self.mu, mu[None]], axis=0)
        self.sigma = np.concatenate([self.sigma, sigma[None]], axis=0)
        self.init_priors = np.concatenate([self.init_priors, np.zeros(1)], axis=0)
        self.priors = np.concatenate([self.priors, np.zeros(1)], axis=0)
        pass

    def viterbi(self, demo, reg=True):
        """
        Compute most likely sequence of state given observations

        :param demo: 	[np.array([nb_timestep, nb_dim])]
        :return:
        """

        nb_data, dim = demo.shape

        logB = np.zeros((self.nb_states, nb_data))
        logDELTA = np.zeros((self.nb_states, nb_data))
        PSI = np.zeros((self.nb_states, nb_data)).astype(int)

        _, logB = self.obs_likelihood(demo)

        # forward pass
        logDELTA[:, 0] = np.log(self.init_priors + realmin * reg) + logB[:, 0]

        for t in range(1, nb_data):
            for i in range(self.nb_states):
                # get index of maximum value : most probables
                PSI[i, t] = np.argmax(logDELTA[:, t - 1] + np.log(self.Trans[:, i] + realmin * reg))
                logDELTA[i, t] = np.max(logDELTA[:, t - 1] + np.log(self.Trans[:, i] + realmin * reg)) + logB[i, t]

        assert not np.any(np.isnan(logDELTA)), "Nan values"

        # backtracking
        q = [0 for i in range(nb_data)]
        q[-1] = np.argmax(logDELTA[:, -1])
        for t in range(nb_data - 2, -1, -1):
            q[t] = PSI[q[t + 1], t + 1]

        return q

    def split_kbins(self, demos):
        raise NotImplementedError

        t_sep = []
        t_resp = []

        for demo in demos:
            t_sep += [map(int, np.round(
                np.linspace(0, demo.shape[0], self.nb_states + 1)))]

            resp = np.zeros((demo.shape[0], self.nb_states))

            # print t_sep
            for i in range(self.nb_states):
                resp[t_sep[-1][i]:t_sep[-1][i+1], i] = 1.0
            # print resp
            t_resp += [resp]

        return np.concatenate(t_resp)

    def obs_likelihood(self, demo=None, dep=None, marginal=None):
        """
        marginal seems to be [slice] or [list of index] over the dimensions (change to manis?)
        """
        sample_size = demo.shape[0]
        # emission probabilities
        B = np.ones((self.nb_states, sample_size))

        if marginal is not None:
            marg_gmm = self.margin(marginal)
        else:
            marg_gmm = self

        if marginal != []:
            for i in range(self.nb_states):

                if dep is None:
                    # evaluate the MVN at index i of marg_gmm at demo
                    B[i, :] = marg_gmm.gaussians[i].prob_from_np(demo, log=True)

                else:  # block diagonal computation
                    raise NotImplementedError(
                        "Didn't implemnt as thought it was not used.")

        return np.exp(B), B

    def gmr_from_np(self, demo, i_in=0, i_out=1, initial_obs=False):
        '''Perform Gaussian Mixture Regression
        demo : single observation or sequence of observations
        i_in : Index of input manifold
        i_out: Index of output manifold
        '''

        m_out  = self.gaussians[0].manifold.get_submanifold(i_out)
        m_in   = self.gaussians[0].manifold.get_submanifold(i_in)

        data_in = self.manifold.np_to_manifold(demo, dim=i_in)

        # Check swap input data to list:
        ldata_in = m_in.swapto_listoftuple(data_in)
        n_data = len(ldata_in)
            
        # Compute weights:
        if n_data == 1:
            h = self.online_forward_message(demo, marginal=i_in,
                                            reset=initial_obs)
            h = np.expand_dims(h, 0)
        else:
            h = np.zeros((n_data, self.n_components))
            for step, point in enumerate(ldata_in):
                h[step, :] = self.online_forward_message(point, marginal=i_in,
                                                         reset=initial_obs)
            # NOTE: online_forward_message takes a new obs and computes the forward
            # message. It is not meant to be used for sequences of data.
            # Instead, it assumes the current obs is the next in the sequence.
            # To start a new sequence, set initial_obs=True (reset).
            # TODO: pass initial_obs to this method, but only at first step.
            raise NotImplementedError("Need to implement forward messages for "
                                      "sequences of data.")


        gmr_list = []
        for n, point in enumerate(ldata_in):
            # Compute conditioned elements:
            gc_list = []
            muc_list = []
            for i, gauss in enumerate(self.gaussians):
                # Only compute conditioning for states that have a significant influence
                # Do this to prevent convergence errors
                if h[n,i]>1e-5:
                    # weight is larger than 1e-3
                    gc_list.append(gauss.condition(point, i_in=i_in, i_out=i_out))
                else:
                    # No signnificant weight, just store the margin 
                    gc_list.append(gauss.margin(i_out))
                # Store the means in np array:
                muc_list.append(gc_list[-1].mu)

            # Convert to tuple of  lists, such that we can apply log and exp to them
            muc_list = m_out.swapto_tupleoflist(muc_list) 

            # compute Mu of GMR:
            mu_gmr = deepcopy(gc_list[0].mu)
            diff = 1.0; it = 0
            
            diff == 1

            while (diff > 1e-5):
                delta = h[n,:].dot( m_out.log(muc_list, mu_gmr) )
                mu_gmr = m_out.exp(delta, mu_gmr)
                
                # Compute update difference
                diff = (delta*delta).sum()
                
                if it>50:
                    logger.warning('GMR did not converge in {0} iterations.'.format(n))
                    break;
                it+=1
                    
             # Compute covariance: 
            sigma_gmr = np.zeros( (m_out.n_dimT, m_out.n_dimT))
            for i in range(self.n_components):
                R = m_out.parallel_transport(np.eye(m_out.n_dimT), 
                                            gc_list[i].mu, mu_gmr).T

                dtmp = np.zeros(m_out.n_dimT)[:,None] #m_out.log(gc_list[i].mu, mu_gmr)[:,None] #
                sigma_gmr += h[n,i]*( R.dot(gc_list[i].sigma.dot(R.T))
                                     + dtmp.dot(dtmp.T) 
                                    )           

            gmr_list.append( Gaussian(m_out, mu_gmr, sigma_gmr))
            
        return gmr_list

    def online_forward_message(self, x, marginal=None, reset=False):
        """
        Takes a single observation and computes the forward message.
        :param x:
        :param marginal: slice
        :param reset:
        :return:
        """
        if (not hasattr(self, '_marginal_tmp') or reset) and marginal is not None:
            self._marginal_tmp = self.margin(marginal)

        if marginal is not None:
            B, _ = self._marginal_tmp.obs_likelihood(x)
        else:
            B, _ = self.obs_likelihood(x)

        # print(B)
        # print(self.init_priors)

        # print(x.shape, B.shape)

        if not hasattr(self, '_alpha_tmp') or reset:
            self._alpha_tmp = self.init_priors * B[:, 0]
        else:
            self._alpha_tmp = self._alpha_tmp.dot(self.Trans) * B[:, 0]

        self._alpha_tmp /= np.sum(self._alpha_tmp, keepdims=True)

        return self._alpha_tmp

    def compute_messages(self, demo=None, dep=None, table=None, marginal=None):
        """

        :param demo: 	[np.array([nb_timestep, nb_dim])]
        :param dep: 	[A x [B x [int]]] A list of list of dimensions
            Each list of dimensions indicates a dependence of variables in the covariance matrix
            E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
            E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
            covariance with dim [2]
        :param table: 	np.array([nb_states, nb_demos]) - composed of 0 and 1
            A mask that avoid some demos to be assigned to some states
        :param marginal: [slice(dim_start, dim_end)] or []
            If not None, compute messages with marginals probabilities
            If [] compute messages without observations, use size
            (can be used for time-series regression)
        :return:
        """
        sample_size = demo.shape[0]

        B, _ = self.obs_likelihood(demo, dep, marginal)
        # if table is not None:
        # 	B *= table[:, [n]]
        self._B = B

        # forward variable alpha (rescaled)
        alpha = np.zeros((self.nb_states, sample_size))
        alpha[:, 0] = self.init_priors * B[:, 0]

        c = np.zeros(sample_size)
        c[0] = 1.0 / np.sum(alpha[:, 0] + realmin)
        alpha[:, 0] = alpha[:, 0] * c[0]

        for t in range(1, sample_size):
            alpha[:, t] = alpha[:, t - 1].dot(self.Trans) * B[:, t]
            # Scaling to avoid underflow issues
            c[t] = 1.0 / np.sum(alpha[:, t] + realmin)
            alpha[:, t] = alpha[:, t] * c[t]

        # backward variable beta (rescaled)
        beta = np.zeros((self.nb_states, sample_size))
        beta[:, -1] = np.ones(self.nb_states) * c[-1]  # Rescaling

        for t in range(sample_size - 2, -1, -1):
            beta[:, t] = np.dot(self.Trans, beta[:, t + 1] * B[:, t + 1])
            # catch NaNs caused by overflow
            beta[:, t] = np.where(np.isnan(beta[:, t]), realmax, beta[:, t])
            beta[:, t] = np.minimum(beta[:, t] * c[t], realmax)

        # Smooth node marginals, gamma
        gamma = (alpha * beta) / np.tile(np.sum(alpha * beta, axis=0) + realmin,
                                         (self.nb_states, 1))

        # Smooth edge marginals. zeta (fast version, considers the scaling factor)
        zeta = np.zeros((self.nb_states, self.nb_states, sample_size - 1))

        for i in range(self.nb_states):
            for j in range(self.nb_states):
                zeta[i, j, :] = self.Trans[i, j] * alpha[i, 0:-1] * B[j, 1:] * beta[
                                                                               j,
                                                                               1:]
        return alpha, beta, gamma, zeta, c

    def init_params_random(self, data, left_to_right=False, self_trans=0.9):
        """

        :param data:
        :param left_to_right:  	if True, init with left to right. All observations models
            will be the same, and transition matrix will be set to l_t_r
        :type left_to_right: 	bool
        :param self_trans:		if left_to_right, self transition value to fill
        :type self_trans:		float
        :return:
        """
        raise NotImplementedError
        # NOTE: primarily need to update the data-format
        mu = np.mean(data, axis=0)
        sigma = np.cov(data.T)
        if sigma.ndim == 0:
            sigma = np.ones((1,1))*sigma


        if left_to_right:
            self.mu = np.array([mu for i in range(self.nb_states)])
        else:
            self.mu = np.array([np.random.multivariate_normal(mu*1, sigma)
                 for i in range(self.nb_states)])

        self.sigma = np.array([sigma + self.reg for i in range(self.nb_states)])
        self.priors = np.ones(self.nb_states) / self.nb_states

        if left_to_right:
            self.Trans = np.zeros((self.nb_states, self.nb_states))
            for i in range(self.nb_states):
                if i < self.nb_states - 1:
                    self.Trans[i, i] = self_trans
                    self.Trans[i, i+1] = 1. - self_trans
                else:
                    self.Trans[i, i] = 1.

            self.init_priors = np.zeros(self.nb_states)/ self.nb_states
        else:
            self.Trans = np.ones((self.nb_states, self.nb_states)) * (1.-self_trans)/(self.nb_states-1)
            # remove diagonal
            self.Trans *= (1.-np.eye(self.nb_states))
            self.Trans += self_trans * np.eye(self.nb_states)
            self.init_priors = np.ones(self.nb_states)/ self.nb_states

    def gmm_init(self, data, **kwargs):
        logger.info("Initializing HMM with GMM EM.")
        GMM.fit_from_np(self, data, **kwargs)

        self.init_priors = np.ones(self.nb_states) / self.nb_states
        self.Trans = np.ones((self.nb_states, self.nb_states))/self.nb_states

    def init_loop(self, demos):
        raise NotImplementedError
        # NOTE: at least needs data format update.
        self.Trans = 0.98 * np.eye(self.nb_states)
        for i in range(self.nb_states-1):
            self.Trans[i, i + 1] = 0.02

        self.Trans[-1, 0] = 0.02

        data = np.concatenate(demos, axis=0)
        _mu = np.mean(data, axis=0)
        _cov = np.cov(data.T)

        self.mu = np.array([_mu for i in range(self.nb_states)])
        self.sigma = np.array([_cov for i in range(self.nb_states)])

        self.init_priors = np.array([1.] + [0. for i in range(self.nb_states-1)])

    def em(self, demos, dep=None, table=None, dep_mask=None,
           left_to_right=False, nb_max_steps=40, loop=False, obs_fixed=False,
           trans_reg=None, mle_kwargs=None, finish_kwargs=None,
           fix_last_component=False, fix_first_component=False):
        """

        :param demos:
        :param dep:		[A x [B x [int]]] A list of list of dimensions or slices
            Each list of dimensions indicates a dependence of variables in the covariance matrix
            !!! dimensions should not overlap eg : [[0], [0, 1]] should be [[0, 1]], [[0, 1], [1, 2]] should be [[0, 1, 2]]
            E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
            E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
            covariance with dim [2]
            E.g. [slice(0, 2), [2]] indicates a full covariance matrix between [0, 1] and no
            covariance with dim [2]
        :param table:		np.array([nb_states, nb_demos]) - composed of 0 and 1
            A mask that avoid some demos to be assigned to some states


        trans_reg: reg for transition matrix
        left_to_right: reg for transition matrix to be sequential model
        loop: reg for transition matrix to be loop model
            Both just mask the trans model and multiply other values by 0

        dep_mask: mask for covariance matrix. Sets other values to zero.
            Can use to remove unwanted covariances.


        :return:
        """
        # TODO: understand dep (not dep_mask)

        if mle_kwargs is None:
            mle_kwargs = {}

        nb_min_steps = 2  # min num iterations
        max_diff_ll = 1e-4  # max log-likelihood increase

        nb_samples = demos.shape[0]
        data = np.concatenate(demos).T

        data_rbd = self.manifold.np_to_manifold(data.T)

        s = [{} for _ in demos]
        LL = np.zeros(nb_max_steps)  # stored log-likelihood


        if dep is not None:
            dep_mask = self.get_dep_mask(dep)

        if self.mu is None or self.sigma is None:
            raise ValueError("HMM not initialized")

        # create regularization matrix

        if left_to_right or loop:
            mask = np.eye(self.Trans.shape[0])
            for i in range(self.Trans.shape[0] - 1):
                mask[i, i + 1] = 1.
            if loop:
                mask[-1, 0] = 1.

        if dep_mask is not None:
            raise NotImplementedError(
                "Need to update sigma for the gaussians individually, see below."
                "When implementing this, also update all other occurances below.")
            self.sigma *= dep_mask


        gaussians = self.gaussians
        if fix_last_component:
            gaussians = gaussians[:-1]
        if fix_first_component:
            gaussians = gaussians[1:]

        offset = 1 if fix_first_component else 0

        for it in tqdm(range(nb_max_steps), desc='HMM EM'):

            for n, demo in enumerate(demos):
                s[n]['alpha'], s[n]['beta'], s[n]['gamma'], s[n]['zeta'], s[n]['c'] = HMM.compute_messages(self, demo, dep, table)

            # concatenate intermediary vars
            gamma = np.hstack([s[i]['gamma'] for i in range(nb_samples)])
            zeta = np.dstack([s[i]['zeta'] for i in range(nb_samples)])
            gamma_init = np.hstack([s[i]['gamma'][:, 0:1] for i in range(nb_samples)])
            gamma_trk = np.hstack([s[i]['gamma'][:, 0:-1] for i in range(nb_samples)])

            gamma2 = gamma / (np.sum(gamma, axis=1, keepdims=True) + realmin)

            # M-step
            if not obs_fixed:
                for i,gauss in enumerate(gaussians):
                    gauss.mle(data_rbd, gamma2[i+offset], **mle_kwargs)

                if dep_mask is not None:
                    self.sigma *= dep_mask

            # Update initial state probablility vector
            self.init_priors = np.mean(gamma_init, axis=1)

            # Update transition probabilities
            if np.isnan(zeta).any():
                raise ValueError("Nan in zeta")
            if np.isnan(gamma_trk).any():
                raise ValueError("Nan in gamma_trk")
            self.Trans = np.sum(zeta, axis=2) / (np.sum(gamma_trk, axis=1) + realmin)

            if trans_reg is not None:
                self.Trans += trans_reg
                self.Trans /= np.sum(self.Trans, axis=1, keepdims=True)

            if left_to_right or loop:
                self.Trans *= mask
                self.Trans /= np.sum(self.Trans, axis=1, keepdims=True)

            # print self.Trans
            # Compute avarage log-likelihood using alpha scaling factors
            LL[it] = 0
            for n in range(nb_samples):
                LL[it] -= sum(np.log(s[n]['c']))
            LL[it] = LL[it] / nb_samples

            self._gammas = [s_['gamma'] for s_ in s]

            # Check for convergence
            if it > nb_min_steps and LL[it] - LL[it - 1] < max_diff_ll:
                logger.info("HMM EM converged")

                if finish_kwargs is not None:
                    for i, gauss in enumerate(gaussians):
                        gauss._update_empirical_covariance(
                            data_rbd, gamma2[i+offset], **finish_kwargs)

                if dep_mask is not None:
                    self.sigma *= dep_mask

                break

        else:
            logger.warning("HMM EM did not converge")

        return gamma, LL[it]

    def score(self, demos):
        """

        :param demos:	[list of np.array([nb_timestep, nb_dim])]
        :return:
        """
        # NOTE: seems to given likelihood of HMM for given demos
        ll = []
        for n, demo in enumerate(demos):
            _, _, _, _, c = HMM.compute_messages(self, demo)
            ll += [np.sum(np.log(c))]

        return ll

    def condition(self, data_in, dim_in, dim_out):

        print(type(data_in))
        print(len(data_in))
        print(type(dim_in[0]))
        print(len(dim_in[0]))

        return super.condition(self, data_in, dim_in, dim_out)

        # understand why for slice(0,1) the dim_in_msg are empty. Is this time?
        # if return_gmm:
        #     return super().condition(data_in, dim_in, dim_out, return_gmm=return_gmm)
        # else:
        #     if dim_in == slice(0, 1):
        #         dim_in_msg = []
        #     else:
        #         dim_in_msg = dim_in
        #     a, _, _, _, _ = self.compute_messages(data_in, marginal=dim_in_msg)

        #     return super().condition(data_in, dim_in, dim_out, h=a)
    
    def margin(self, i_in):
        marg = GMM.margin(self, i_in)

        marg._trans = np.copy(self._trans)
        marg._init_priors = np.copy(self._init_priors)

        return marg

    def copy(self):
        other = HMM(deepcopy(self.manifold), self.n_components)

        # Copy relevant parts from GMM
        other.gaussians = [g.copy() for g in self.gaussians]
        other.priors = np.copy(self.priors)  # not needed for HMM
        other.base = deepcopy(self.base)

        # Copy HMM-specific parts
        other._trans = np.copy(self._trans)
        other._init_priors = np.copy(self._init_priors)

        return other


class HSMM(GMM):
    def __init__(self, manifold, n_components, base=None):
        raise NotImplementedError