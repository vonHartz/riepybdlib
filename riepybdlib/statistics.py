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

import numpy as np
import pbdlib as pbd
import riepybdlib.manifold as rm
import riepybdlib.plot as fctplt

from copy import deepcopy

from enum import Enum

class RegularizationType(Enum):
    NONE      = None
    SHRINKAGE = 1
    DIAGONAL  = 2
    COMBINED  = 3

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

    def prob_from_np(self, npdata):
        data = self.manifold.np_to_manifold(npdata)

        return self.prob(data)
        
    def prob(self,data):
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
            logger.warning('Singular matrix, using lstsq', filter=False)
            dist = ( dist * np.linalg.lstsq(self.sigma,dist.T,rcond=None)[0].T ).sum(axis=(dist.ndim-1))
        probs =  np.exp( -0.5*dist )/reg 

        # Iterative
        #probs = []
        #for i, x in enumerate(data):
        #    dist = self.manifold.log(self.mu,x)
        #    dist = np.sum(dist*np.linalg.solve(self.sigma,dist),0)
        #    probs.append( np.exp( -0.5*dist )/reg )
        
        # Return results
        #print(probs)
        return probs
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
            logger.info(f'Mean computation iter {it} current diff: {diff}')
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
            
        # Perform Shrinkage regularizaton:
        if (reg_type == RegularizationType.SHRINKAGE):
            return reg_lambda*np.diag(np.diag(sigma)) + (1-reg_lambda)*sigma
        elif (reg_type == RegularizationType.DIAGONAL):
            return sigma + reg_lambda*np.eye(len(sigma))
        elif (reg_type == RegularizationType.COMBINED):
            return reg_lambda*np.diag(np.diag(sigma)) + (1-reg_lambda)*sigma + reg_lambda2*np.eye(len(sigma))
        elif reg_type==None:
            return sigma
        else:
            raise ValueError('Unknown regularization type for covariance regularization')

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
        
        # Decomposition of covariance:
        lambda_s = np.linalg.inv(self.sigma)
        lambda_o = np.linalg.inv(other.sigma)
        
        mu  = self.mu # Initial guess
        it=0; diff = 1
        while (diff > conv_thresh):
            # Transport precision to estimated mu
            Rs = fR(self.mu, mu)
            lambda_sn = Rs.dot( lambda_s.dot( Rs.T) )
            Ro = fR(other.mu, mu)
            lambda_on = Ro.dot( lambda_o.dot( Ro.T) )

            # Compute new covariance:
            sigma = np.linalg.inv( lambda_sn + lambda_on )

            # Compute weighted distances:
            d_self  = lambda_sn.dot( Log(self.mu , mu) )
            d_other = lambda_on.dot( Log(other.mu, mu) )

            # update mu:
            delta = sigma.dot(d_self + d_other)
            mu = Exp(delta+0e-4,mu)

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
                     as_np=False):
        if base is None:
            base = self.manifold.id_elem

        if mu_on_tangent:
            mu = self.manifold.log(self.mu, base)
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
        mu0 = manifold.id_elem
        sigma0 = np.eye(manifold.n_dimT)
        
        self.gaussians = []
        for i in range(n_components):
            self.gaussians.append( Gaussian(manifold,mu0,sigma0))
            
        self.n_components = n_components
        self.priors = np.ones(n_components)/n_components
        self.manifold = manifold

        if base is None:
            self.base = manifold.id_elem
        else:
            self.base = base

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
            
        return lik, avg_loglik

    # @logger.contextualize(filter=False)
    def fit_from_np(self, npdata, convthres=1e-5, maxsteps=100, minsteps=5, reg_lambda=1e-3, 
                    reg_lambda2=1e-3, reg_type= RegularizationType.SHRINKAGE):
        '''Initialize trajectory GMM using a time-based approach'''

        data = self.manifold.np_to_manifold(npdata)

        # Make sure that the data is a tuple of list:
        n_data = len(data)
        
        prvlik = 0
        avg_loglik = []
        for st in tqdm(range(maxsteps), desc='EM'):
            # Expectation:
            lik = self.expectation(data)
            gamma0 = (lik/ (lik.sum(axis=0) + 1e-200) )# Sum over states is one
            gamma1 = (gamma0.T/gamma0.sum(axis=1)).T # Sum over data is one

            # Maximization:
            # - Update Gaussian:
            for i,gauss in enumerate(self.gaussians):
                gauss.mle(data,gamma1[i,], reg_lambda, reg_lambda2, reg_type)
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
            
        return lik, avg_loglik

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

    def init_time_based_from_np(self, npdata, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):
        # Assuming that data is first manifold dimension
        t = npdata[:,0]

        # Timing seperation:
        timing_sep = np.linspace(t.min(), t.max(),self.n_components+1)

        for i, g in tqdm(enumerate(self.gaussians), desc='Time-based init',
                         total=self.n_components):
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
                  plot_process=True)
            self.priors[i] = len(idtmp)
        self.priors = self.priors / self.priors.sum()

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
#    def action(self,h):
#        newgmm = GMM(self.n_components, self.manifold, base=h)
#        for i,gauss in enumerate(self.gaussians):
#            newmu = gauss.manifold.action(gauss.mu, self.base, h)
#            newgmm.gaussian[i] = gauss.action(newmu)

    # @logger.contextualize(filter=False)
    def kmeans_from_np(self,npdata, maxsteps=100,reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE ):

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
        newgmm = GMM(self.manifold.get_submanifold(i_in), self.n_components)

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

    def gmr_from_np(self, npdata_in, i_in=0, i_out=1):
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
        h= self.margin(i_in).expectation(data_in)
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
                     as_np=False):
        comp = [g.get_mu_sigma(base=base, idx=idx, mu_on_tangent=mu_on_tangent,
                               as_np=as_np)
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
    def sigma(self):
        _, sigma = self.get_mu_sigma(stack=True, as_np=True)
        return sigma

    def np_to_manifold_to_np(self, npdata_in, i_in=None, base=None):
        data = self.manifold.np_to_manifold(npdata_in, dim=i_in)
        # base = self.mu??
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