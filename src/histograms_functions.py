__all__ = ['rebin_histogram', 'combine_edges',
           'estimate_pds_over_same_support_from_hist',
           'estimate_pdf_from_count_and_edges',
           'estimate_pds_over_same_support_from_kde',
           'estimate_pdf_from_hist']
import scipy
import numpy as np 

def rebin_histogram(old_bin_edges, old_bin_count, new_bin_edges):
    x1 = np.asarray(old_bin_edges)
    y1 = np.asarray(old_bin_count)
    x2 = np.asarray(new_bin_edges)

    # the fractional bin locations of the new bins in the old bins
    i_place = np.interp(x2, x1, np.arange(len(x1)))

    cum_sum = np.r_[[0], np.cumsum(y1)]

    # calculate bins where lower and upper bin edges span
    # greater than or equal to one original bin.
    # This is the contribution from the 'intact' bins (not including the
    # fractional start and end parts.
    whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.
    start = cum_sum[np.ceil(i_place[:-1]).astype(int)]
    finish = cum_sum[np.floor(i_place[1:]).astype(int)]

    y2 = np.where(whole_bins, finish - start, 0.)

    bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1) - 1)

    # fractional contribution for bins where the new bin edges are in the same
    # original bin.
    same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
    frac = i_place[1:] - i_place[:-1]
    contrib = (frac * y1[bin_loc[:-1]])
    y2 += np.where(same_cell, contrib, 0.)

    # fractional contribution for bins where the left and right bin edges are in
    # different original bins.
    different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
    frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
    contrib = (frac_left * y1[bin_loc[:-1]])

    frac_right = i_place[1:] - np.floor(i_place[1:])
    contrib += (frac_right * y1[bin_loc[1:]])

    y2 += np.where(different_cell, contrib, 0.)

    return y2

def combine_edges( counts, edges, threshold ):
    '''
    This function will merge the bins into a numpy histogram given a threshold

    Arguments:

    - counts: the histogram bin counts.
    - edges: the histogram bin edges.
    - threshold: the minimum counts for each bin.
    '''

    max_ix = counts.argmax()
    c_list = list( counts )   # Lists can be popped from
    e_list = list( edges )    # Lists can be popped from

    def eliminate_left( ix ):
        # Sum the count and eliminate the edge relevant to ix
        # Before the peak (max_ix)
        nonlocal max_ix
        max_ix -= 1         # max_ix will change too.
        c_list[ix+1]+=c_list[ix]
        c_list.pop(ix)
        e_list.pop(ix+1)

    def eliminate_right( ix ):
        # Sum the count and eliminate the edge relevant to ix
        # after the peak (max_ix) 
        c_list[ix-1]+=c_list[ix]
        c_list.pop(ix)
        e_list.pop(ix)

    def first_lt():
        # Find the first ix less than the threshold
        for ix, ct in enumerate( c_list[:max_ix] ):
            if ct < threshold:
                return ix  # if ct < threshold return the index and exit the function
        # The function only reaches here if no ct values are less than the threshold
        return -1  # If zero items < threshold return -1

    def last_lt():
        # Find the last ix less than the threshold
        for ix, ct in zip( range(len(c_list)-1, max_ix, -1), c_list[::-1]):
            # ix reduces from len(c_list)-1, c_list is accessed in reverse order.
            if ct < threshold:
                return ix
        return -1  # If no items < threshold return -1

    cont = True
    while cont:
        # Each iteration removes any counts less than threshold
        # before the peak.  This process would combine e.g. counts of [...,1,2,3,...] into [..., 6, ...]
        ix = first_lt()
        if ix < 0:
            cont = False   # If first_lt returns -1 stop while loop
        else:
            eliminate_left( ix )

    cont = True
    while cont:
        ix = last_lt()
        if ix < 0:
            cont = False   # If last_lt returns -1 stop while loop
        else:
            eliminate_right( ix )

    return np.array( c_list ), np.array( e_list )

def estimate_pds_over_same_support_from_hist(p_counts, p_edges, q_counts, q_edges):
    '''
    This function will rebin the histogram in order to have the same support and estimate the pdfs;

    Arguments:
    p_counts: count from p histogram
    p_edges: edges from p histogram
    q_counts: count from q histogram
    q_edges: edges from q histogram
    '''
    # compare the histogram resolution
    if len(p_edges) > len(q_edges):
        p_counts = rebin_histogram(old_bin_edges=p_edges, old_bin_count=p_counts, new_bin_edges=q_edges)
        p_edges  = q_edges.copy()
    elif len(p_edges) < len(q_edges):     
        q_counts = rebin_histogram(old_bin_edges=q_edges, old_bin_count=q_counts, new_bin_edges=p_edges)
        q_edges  = p_edges.copy()
        
    if  np.array_equal(p_edges, q_edges) != True:
        q_counts = rebin_histogram(old_bin_edges=q_edges, old_bin_count=q_counts, new_bin_edges=p_edges)
        q_edges  = p_edges.copy()

    # get pdfs
    p_db = np.array(np.diff(p_edges), float)
    q_db = np.array(np.diff(q_edges), float)
    pdf_p = p_counts/p_db/p_counts.sum()
    pdf_q = q_counts/q_db/q_counts.sum()
    
    return pdf_p, pdf_q, p_edges

def estimate_pdf_from_count_and_edges(counts, edges, return_edges=False):
    db  = np.array(np.diff(edges), float)
    pdf = counts/db/counts.sum()
    if return_edges:
        return pdf, edges
    return pdf
    
def estimate_pds_over_same_support_from_kde(data_p1, data_p2, lim_inf, lim_sup, 
                                            n_points=100, bandwidth=0.5):
    '''
    This function will rebin the histogram in order to have the same support and estimate the pdfs;

    Arguments:
    data_p1:   data to be used for kde estimation
    data_p1:   data to be used for kde estimation
    bandwithd: bandwith method used in the kde process 
    '''
    data_p1 = data_p1[(data_p1>=lim_inf) & (data_p1<lim_sup)]
    data_p2 = data_p2[(data_p2>=lim_inf) & (data_p2<lim_sup)]

    pdf_model1 = scipy.stats.gaussian_kde(data_p1, bw_method=bandwidth)
    pdf_model2 = scipy.stats.gaussian_kde(data_p2, bw_method=bandwidth)
    
    support = np.linspace(lim_inf, lim_sup, n_points)

    pdf_p = lambda x : pdf_model1.evaluate([[x]])
    pdf_q = lambda x : pdf_model2.evaluate([[x]])
    print('Sannity test: \n Integraf p: %1.2f \n Integral q: %1.2f' %(scipy.integrate.quad(lambda x: pdf_p(x), a=-np.inf, b=np.inf)[0],  
                                                                      scipy.integrate.quad(lambda x: pdf_q(x), a=-np.inf, b=np.inf)[0]))
    
    return pdf_model1.pdf(support), pdf_model2.pdf(support), support 

def estimate_pdf_from_hist(data, binning):
    return np.histogram(data, bins=binning, density=True)