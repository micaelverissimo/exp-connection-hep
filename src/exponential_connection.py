
__all__ = ['non_parametric_conection', 'calc_kl']

import scipy


import numpy as np

from loguru import logger
from scipy.spatial.distance import cdist
from .histograms_functions import *

def calc_kl(pk, qk):
    '''
    A function to calculate the Kullback-Libler divergence between p and q distribution.
    Arguments:
    pk: pdf values from p distribution
    qk: pdf values from q distribution
    '''
    return scipy.special.kl_div(pk, qk)


class non_parametric_conection:
    def __init__(self, n_params, pdf1, pdf2, support):
        '''
            This implementation will create connections (e and m) between two PDFs estimated from a Gaussian KDE with a adjustable bandwidth.
            Arguments:
                - n_params: number of t paramters obtained from the connections.
                - pdf1: the pdf used to be the intial point for build the connections.
                - pdf2: the pdf used to be the final point for build the connections.
                - support: the binning i.e. the values that should be used to integrate
                NOTE: The pdf1 and pdf2 must have the same support.

        '''
        
        self.n_params = n_params
        self.p1 = pdf1
        self.p2 = pdf2
        self.support = support
        self.dx = np.diff(self.support)[:-1]
        self.t_params = np.linspace(0, 1, n_params)  # n_params equally spaced points between 0 and 1
        self.exp_connection = self.get_exp_connection()
        self.kl_matrix = self.get_kl_between_e_geo()
        # Get minimum KL divergence excluding diagonal (where KL = 0)
        mask = ~np.eye(self.kl_matrix.shape[0], dtype=bool)  # Create mask to exclude diagonal
        self.min_kl_value = np.min(self.kl_matrix[mask])
        # Log the creation of the exponential connection
        logger.info(f'Exponential connection created with {len(self.exp_connection)} points.')
        logger.info(f'Minimum KL divergence value (excluding diagonal): {self.min_kl_value:.9e}')
    
    def get_exp_connection(self):
        '''
        This function will compute the exponential connection as described in the Amari's Information Geometry and Its Applications book.
        '''
        self.e_geo = []
        for it in self.t_params:
            con   = (1-it)*np.log(self.p1) + it*np.log(self.p2)
            phi_t = np.log(np.trapz(np.exp(con), dx=self.dx))
            self.e_geo.append(np.exp(con - phi_t))
        return self.e_geo
        
    def get_mix_connection(self):
        '''
        This function will compute the exponential mixture as described in the Amari's Information Geometry and Its Applications book.
        '''
        self.mix = []
        for it in self.t_params:
            self.mix.append((1-it)*self.p1 + it*self.p2)
        return self.mix
    
    def get_kl_between_e_geo(self):
        '''
        This function will compute the KL divergence between consecutive elements in the exponential connection in both directions using scipy cdist for parallelization.
        Returns:
            total_kl_values_forward: list of integrated KL divergence values from i to i+1
            total_kl_values_backward: list of integrated KL divergence values from i+1 to i
            kl_dist_matrix: full KL divergence distance matrix between all e_geo elements
        '''
        # Convert exp_connection to numpy array for efficient computation
        e_geo_array = np.array(self.exp_connection)
        
        # Custom KL divergence function for cdist
        def kl_divergence_metric(p, q):
            """Custom KL divergence metric for cdist"""
            kl_div = calc_kl(p, q)
            return np.trapz(kl_div, dx=self.dx)
        
        # Compute full KL divergence distance matrix
        kl_dist_matrix = cdist(e_geo_array, e_geo_array, metric=kl_divergence_metric)

        return kl_dist_matrix

    def get_tangent_vectors(self, pdf, projection_index):
        '''
        This function will compute the tangent vectors for the e-connection and m-connection as described in the Amari's Information Geometry and Its Applications book given a 3th point.
        Argument:
            - pdf: a pdf with the same support.
            - projection_index: the index used to be a projection on exp connection.
        '''
        # calculate the tangent vectors
        return np.log(self.p2)-np.log(self.p1), self.e_geo[projection_index]-pdf
    
    def get_projection_angle(self, tang_u, tang_v):
        '''
        This function will compute the angle between two tangent vectors.
        Arguments:
            - tang_u: the first tangent vector.
            - tang_v: the second tangent vector.
        '''
        # calculate tinner product
        norm_u    = np.sqrt(np.trapz(tang_u*tang_u, dx=self.dx))
        norm_v    = np.sqrt(np.trapz(tang_v*tang_v, dx=self.dx))
        u_inner_v = np.trapz(tang_u*tang_v, dx=self.dx)
        return np.degrees(np.arccos(u_inner_v/(norm_u*norm_v)))

    def get_extended_projection(self, kl_value, angle):
        '''
        This function will compute the extended projection of the KL divergence and the angle.
        Arguments:
            - kl_value: the Kullback-Leibler divergence value.
            - angle: the angle between the tangent vectors.
        '''
        return kl_value * np.cos(np.radians(angle))

    def get_nearest_point(self, pdf3):
        '''
        This function will compute the nearest point to the connection given a 3rd PDF in terms of Kullback-Leibler divergence.
        Argument:
            - pdf3: a new pdf that remains in the same manifold but probably isn't in the connection.
        '''

        best_match = 100
        best_idx   = -1
        con3   = np.log(pdf3)
        phi_t3 = np.log(np.trapz(np.exp(con3), dx=self.dx))
        pdf3   = np.exp(con3 - phi_t3)
        
        all_kl     = []
        all_angles = [] 
        for idx in range(len(self.exp_connection)):
            kl_values = calc_kl(self.exp_connection[idx], pdf3)
            total_kl  = np.trapz(kl_values, dx=self.dx)
            # get tangent vectors
            e_tan_vec, m_tan_vec = self.get_tangent_vectors(pdf3, idx)
            all_angles.append(self.get_projection_angle(e_tan_vec, m_tan_vec))
            all_kl.append(total_kl)
            if total_kl < best_match: 
                best_match = total_kl
                best_idx   = idx
        return best_match, best_idx, np.array(all_kl), np.array(all_angles)