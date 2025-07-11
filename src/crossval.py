

__all__ = ['crossval_table']


from pybeamer import *
from pprint import pprint
from functools import reduce
from itertools import product
from tqdm import tqdm
from neuralnet import logger



import matplotlib as mpl
import matplotlib.pyplot as plt
import collections, os, glob, json, copy, re, pickle
import numpy as np
import pandas as pd
import os
import json
import joblib
import glob
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json
fire = '\U0001F525'


def load_file(m_file):
    if m_file.endswith('pic'):
        with open(m_file, 'rb') as f:
            tuned_file = [pickle.load(f)]
            version = 1 # enable Run 3 schemma
        return tuned_file
    # Run 2 files
    elif m_file.endswith('npz'):
        return dict(np.load(m_file, allow_pickle=True))['tunedData']
    else:
        print('Extension not supported.')
class crossval_table:
    #
    # Constructor
    #
    def __init__(self, config_dict, etbins=None, etabins=None ):
        '''
        The objective of this class is extract the tuning information from saphyra's output and
        create a pandas DataFrame using then.
        The informations used in this DataFrame are listed in info_dict, but the user can add more
        information from saphyra summary for example.


        Arguments:

        - config_dict: a dictionary contains in keys the measures that user want to check and
        the values need to be a empty list.

        Ex.: info = collections.OrderedDict( {

              "max_sp_val"      : 'summary/max_sp_val',
              "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
              "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
              "max_sp_op"       : 'summary/max_sp_op',
              "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
              "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
              'tight_pd_ref'    : "reference/tight_cutbased/pd_ref#0",
              'tight_fa_ref'    : "reference/tight_cutbased/fa_ref#0",
              'tight_pd_ref_passed'     : "reference/tight_cutbased/pd_ref#1",
              'tight_fa_ref_passed'     : "reference/tight_cutbased/fa_ref#1",
              'tight_pd_ref_total'      : "reference/tight_cutbased/pd_ref#2",
              'tight_fa_ref_total'      : "reference/tight_cutbased/fa_ref#2",
              'tight_pd_val_passed'     : "reference/tight_cutbased/pd_val#1",
              'tight_fa_val_passed'     : "reference/tight_cutbased/fa_val#1",
              'tight_pd_val_total'      : "reference/tight_cutbased/pd_val#2",
              'tight_fa_val_total'      : "reference/tight_cutbased/fa_val#2",
              'tight_pd_op_passed'      : "reference/tight_cutbased/pd_op#1",
              'tight_fa_op_passed'      : "reference/tight_cutbased/fa_op#1",
              'tight_pd_op_total'       : "reference/tight_cutbased/pd_op#2",
              'tight_fa_op_total'       : "reference/tight_cutbased/fa_op#2",

              } )

        - etbins: a list of et bins edges used in training;
        - etabins: a list of eta bins edges used in training;
        '''
        self.table = None

        # Check wanted key type
        self.__config_dict = collections.OrderedDict(config_dict) if type(config_dict) is dict else config_dict
        self.__etbins = etbins
        self.__etabins = etabins


    #
    # Fill the main dataframe with values from the tuning files and convert to pandas dataframe
    #
    def fill(self, path, tag):
        '''
        This method will fill the information dictionary and convert then into a pandas DataFrame.

        Arguments.:

        - path: the path to the tuned files;
        - tag: the training tag used;
        '''
        paths = glob.glob(path)
        logger.info( f"Reading file for {tag} tag from {path}" )

        # Creating the dataframe
        dataframe = collections.OrderedDict({
                              'train_tag'      : [],
                              'et_bin'         : [],
                              'eta_bin'        : [],
                              'model_idx'      : [],
                              'sort'           : [],
                              'init'           : [],
                              'file_name'      : [],
                              'op_name'        : [],
                          })


        logger.info( f'There are {len(paths)} files for this task...')
        logger.info( f'Filling the table... ')

        for ituned_file_name in tqdm( paths , desc=fire+' Reading %s...'%tag):
            #for ituned_file_name in paths:
            # Extract et and eta indices from the file name using regex
            match = re.search(r'_et(\d+)_eta(\d+)', ituned_file_name)
            if match:
                et = int(match.group(1))
                eta = int(match.group(2))
            else:
                logger.warning(f"Could not extract et/eta from file name {ituned_file_name}.")
                et = None
                eta = None
            version = 0 # Run 2 schemma
            try:
                # Run 3 files
                if ituned_file_name.endswith('pic'):
                    with open(ituned_file_name, 'rb') as f:
                        tuned_file = [pickle.load(f)]
                        version = 1 # enable Run 3 schemma
                # Run 2 files
                elif ituned_file_name.endswith('npz'):
                    tuned_file = dict(np.load(ituned_file_name, allow_pickle=True))['tunedData']
                else:
                    logger.error(f'Extension not supported. Skip file {itune_file_name}.')
                    continue
                
            except:
                logger.warning( f"File {ituned_file_name} not open. skip.")
                continue


            for idx, ituned in enumerate(tuned_file):

                history = ituned['history']

                for op, config_dict in self.__config_dict.items():
                    
                    dataframe['train_tag'].append(tag)
                    dataframe['file_name'].append(ituned_file_name)
                    dataframe['op_name'].append(op)

                    if version == 0: # Run 2 
                        dataframe['model_idx'].append(ituned['imodel'])
                        dataframe['sort'].append(ituned['sort'])
                        dataframe['init'].append(ituned['init'])
                        dataframe['et_bin'].append(self.get_etbin(ituned_file_name))
                        dataframe['eta_bin'].append(self.get_etabin(ituned_file_name))

                    elif version == 1: # Run 3
                        dataframe['sort'].append(ituned['folds'])
                        dataframe['init'].append(0)
                        dataframe['et_bin'].append(et)
                        dataframe['eta_bin'].append(eta)
                        dataframe['model_idx'].append(0)


                    # Get the value for each wanted key passed by the user in the contructor args.
                    for key, local  in config_dict.items():
                        if not key in dataframe.keys():
                            dataframe[key] = [self.__get_value( history, local )]
                        else:
                            dataframe[key].append( self.__get_value( history, local ) )
        
        # append tables if is need
        # ignoring index to avoid duplicated entries in 
        self.table = pd.concat( (self.table, pd.DataFrame(dataframe) ), ignore_index=True ) if self.table is not None else pd.DataFrame(dataframe)
        logger.info( 'End of fill step, a pandas DataFrame was created...')


    #
    # Convert the table to csv
    #
    def to_csv( self, output ):
        '''
        This function will save the pandas Dataframe into a csv file.

        Arguments.:

        - output: the path and the name to be use for save the table.

        Ex.:
        m_path = './my_awsome_path
        m_name = 'my_awsome_name.csv'

        output = os.path.join(m_path, m_name)
        '''
        self.table.to_csv(output, index=False)


    #
    # Read the table from csv
    #
    def from_csv( self, input ):
        '''
        This method is used to read a csv file insted to fill the Dataframe from tuned file.

        Arguments:

        - input: the csv file to be opened;
        '''
        self.table = pd.read_csv(input)



    #
    # Get the value using recursive dictionary navigation
    #
    def __get_value(self, history, local):
        '''
        This method will return a value given a history and dictionary with keys.

        Arguments:

        - history: the tuned information file;
        - local: the path caming from config_dict;
        '''
        # Protection to not override the history since this is a 'mutable' object
        var = copy.copy(history)
        for key in local.split('/'):
            var = var[key.split('#')[0]][int(key.split('#')[1])] if '#' in key else var[key]
        return var


    def get_etbin(self, job):
        return int(  re.findall(r'et[a]?[0-9]', job)[0][-1] )


    def get_etabin(self, job):
        return int( re.findall(r'et[a]?[0-9]',job)[1] [-1] )


    def get_etbin_edges(self, et_bin):
        return (self.__etbins[et_bin], self.__etbins[et_bin+1])


    def get_etabin_edges(self, eta_bin):
        return (self.__etabins[eta_bin], self.__etabins[eta_bin+1])




    #
    # Return only best inits
    #
    def filter_inits(self, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best inits for every sort.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = self.table.groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'sort', 'op_name'])[key].idxmin().values
            return self.table.loc[idxmask]
        else:
            idxmask = self.table.groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'sort', 'op_name'])[key].idxmax().values
            return self.table.loc[idxmask]


    #
    # Get the best sorts from best inits table
    #
    def filter_sorts(self, best_inits, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best model for every configuration.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = best_inits.groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'op_name'])[key].idxmin().values
            return best_inits.loc[idxmask]
        else:
            idxmask = best_inits.groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'op_name'])[key].idxmax().values
            return best_inits.loc[idxmask]


    #
    # Calculate the mean/std table from best inits table
    #
    def describe(cls, best_inits ):
        '''
        This method will give the mean and std for construct the beamer presentation for each train tag.

        Arguments:

        - best_inits:
        '''
        # Create a new dataframe to hold this table
        dataframe = { 'train_tag' : [], 'et_bin' : [], 'eta_bin' : [], 'op_name':[],
                      'pd_ref':[], 'fa_ref':[], 'sp_ref':[]}
        # Include all wanted keys into the dataframe
        for key in best_inits.columns.values:
            if key in ['train_tag', 'et_bin', 'eta_bin', 'op_name']:
                continue
            elif 'passed' in key or 'total' in key:
                continue
            elif ('op' in key) or ('val' in key):
                dataframe[key+'_mean'] = []; dataframe[key+'_std'] = []
            else:
                continue

        # Loop over all tuning tags and et/eta bins
        for tag in best_inits.train_tag.unique():
            for et_bin in best_inits.et_bin.unique():
                for eta_bin in best_inits.eta_bin.unique():
                    for op in best_inits.op_name.unique():

                        cv_bin = best_inits.loc[ (best_inits.train_tag == tag) & (best_inits.et_bin == et_bin) & (best_inits.eta_bin == eta_bin) & (best_inits.op_name == op)]
                        dataframe['train_tag'].append( tag )
                        dataframe['et_bin'].append( et_bin )
                        dataframe['eta_bin'].append( eta_bin )
                        dataframe['op_name'].append(op)
                        dataframe['pd_ref'].append(cv_bin.pd_ref.values[0])
                        dataframe['fa_ref'].append(0)#cv_bin.fa_ref.values[0])
                        dataframe['sp_ref'].append(0)#cv_bin.sp_ref.values[0])

                        for key in best_inits.columns.values:
                            if key in ['train_tag', 'et_bin', 'eta_bin', 'op_name']:
                                continue # skip common
                            elif 'passed' in key or 'total' in key:
                                continue # skip counts
                            elif ('op' in key) or ('val' in key):
                                dataframe[key+'_mean'].append( cv_bin[key].mean() ); dataframe[key+'_std'].append( cv_bin[key].std() )
                            else: # skip others
                                continue


        # Return the pandas dataframe
        return pd.DataFrame(dataframe)



    #
    # Get tge cross val integrated table from best inits
    #
    def integrate( self, best_inits, tag ):
        '''
        This method is used to get the integrate information of a given tag.

        Arguments:

        - best_inits: a pandas Dataframe which contains all information for the best inits.
        - tag: the training tag that will be integrate.
        '''
        keys = [ key for key in best_inits.columns.values if 'passed' in key or 'total' in key]
        table = best_inits.loc[best_inits.train_tag==tag].groupby(['sort']).agg(dict(zip( keys, ['sum']*len(keys))))

        for key in keys:
            if 'passed' in key:
                orig_key = key.replace('_passed','')
                values = table[key].div( table[orig_key+'_total'] )
                table[orig_key] = values
                table=table.drop([key],axis=1)
                table=table.drop([orig_key+'_total'],axis=1)
        return table.agg(['mean','std'])



    #
    # Create the beamer table file
    #
    def dump_beamer_table( self, best_inits, output, tags=None, toPDF=True, title='' , test_table=None, et_table_index=None):
        '''
        This method will use a pandas Dataframe in order to create a beamer presentation which summary the tuning cross-validation.

        Arguments:
        - best_inits: a pandas Dataframe which contains all information for the best inits.
        - operation_poins: the operation point that will be used.
        - output: a name for the pdf
        - tags: the training tag that will be used. If None then the tags will be get from the Dataframe.
        - title: the pdf title
        - et_table_index: order of idx bins to shoe up per page, each list represents a page. Ex: [0,1,2] or [[1,2],[3]]
        '''

        if not et_table_index:
            et_table_index = best_inits.et_bin.unique()

        cv_table = self.describe( best_inits )
        # Create Latex Et bins
        etbins_str = []; etabins_str = []

        # check number of pages
        et_table_page_numbers = len(et_table_index) if isinstance(et_table_index[0], list) else 1

        ### loop per page to build et line
        for page_number in range(0,et_table_page_numbers,1):
            et_bins = et_table_index[page_number] if et_table_page_numbers>1 else et_table_index
            etbins_temp_str = []
            for etBinIdx in et_bins:
                etbin = self.get_etbin_edges(etBinIdx)
                if etbin[1] > 1000 :
                    etbins_temp_str.append( r'$E_{T}\text{[GeV]} > %d$' % etbin[0])
                else:
                    etbins_temp_str.append(  r'$%d \leq E_{T} \text{[Gev]}<%d$'%etbin )
            if et_table_page_numbers>1:
                etbins_str.append(etbins_temp_str)
            else:
                etbins_str = etbins_temp_str

        # Create Latex eta bins
        for etaBinIdx in best_inits.eta_bin.unique():
            #etabin = (self.__etabins[etaBinIdx], self.__etabins[etaBinIdx+1])
            etabin = self.get_etabin_edges(etaBinIdx)
            etabins_str.append( r'$%.2f\leq\eta<%.2f$'%etabin )

        # Default colors
        colorPD = '\\cellcolor[HTML]{9AFF99}'; colorPF = ''; colorSP = ''

        train_tags = cv_table.train_tag.unique() if not tags else tags

        # fix tags to list
        if type(tags) is str: tags=[tags]

        # Apply beamer
        with BeamerTexReportTemplate1( theme = 'Berlin'
                                 , _toPDF = toPDF
                                 , title = title
                                 , outputFile = output
                                 , font = 'structurebold' ):
                
            tex_lines=[]

            # loop over pages
            for page_number in range(0,et_table_page_numbers,1):
                latex_et_bin = et_table_index[page_number] if et_table_page_numbers>1 else et_table_index
                latex_etbins_str = etbins_str[page_number] if et_table_page_numbers>1 else etbins_str
                ### Prepare tables
                tuning_names = ['']; tuning_names.extend( train_tags )
                tex_line = []
                tex_line += [ HLine(_contextManaged = False) ]
                tex_line += [ HLine(_contextManaged = False) ]
                tex_line += [ TableLine( columns = ['','','kinematic region'] + reduce(lambda x,y: x+y,[['',s,''] for s in latex_etbins_str]), _contextManaged = False ) ]
                tex_line += [ HLine(_contextManaged = False) ]
                tex_line += [ TableLine( columns = ['Det. Region','Method','Type'] + reduce(lambda x,y: x+y,[[colorPD+r'$P_{D}[\%]$',colorSP+r'$SP[\%]$',colorPF+r'$P_{F}[\%]$'] \
                                      for _ in latex_etbins_str]), _contextManaged = False ) ]
                tex_line += [ HLine(_contextManaged = False) ]

                for etaBinIdx in cv_table.eta_bin.unique():
                    for idx, tag in enumerate( train_tags ):
                        cv_values=[]; ref_values=[]; test_values=[]
                        for etBinIdx in latex_et_bin:
                            current_table = cv_table.loc[ (cv_table.train_tag==tag) & (cv_table.et_bin==etBinIdx) & (cv_table.eta_bin==etaBinIdx) ]
                            sp = current_table['sp_val_mean'].values[0]*100
                            pd = current_table['pd_val_mean'].values[0]*100
                            fa = current_table['fa_val_mean'].values[0]*100
                            sp_std = current_table['sp_val_std'].values[0]*100
                            pd_std = current_table['pd_val_std'].values[0]*100
                            fa_std = current_table['fa_val_std'].values[0]*100
                            sp_ref = 0#current_table['sp_ref'].values[0]*100
                            pd_ref = current_table['pd_ref'].values[0]*100
                            fa_ref = 0#current_table['fa_ref'].values[0]*100

                            cv_values   += [ colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorSP+('%1.2f$\pm$%1.2f')%(sp,sp_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std),    ]
                            ref_values  += [ colorPD+('%1.2f')%(pd_ref), colorSP+('%1.2f')%(sp_ref), colorPF+('%1.2f')%(fa_ref)]
                            
                            if test_table is not None:
                                current_test_table = test_table.loc[ (test_table.train_tag==tag) & (test_table.et_bin==etBinIdx) & (test_table.eta_bin==etaBinIdx) ]
                                pd = current_test_table['pd_test'].values[0]*100
                                sp = current_test_table['sp_test'].values[0]*100
                                fa = current_test_table['fa_test'].values[0]*100
                                test_values   += [ colorPD+('%1.2f')%(pd),colorSP+('%1.2f')%(sp),colorPF+('%1.2f')%(fa)    ]

                        if idx > 0:
                            tex_line += [ TableLine( columns = ['', tuning_names[idx+1], 'Cross Val.'] + cv_values   , _contextManaged = False ) ]
                        
                        
                            if test_table is not None:
                                tex_line += [ TableLine( columns = ['', tuning_names[idx+1], 'Test'] + test_values   , _contextManaged = False ) ]
                    
                        
                        else:
                            tex_line += [ TableLine( columns = ['\multirow{%d}{*}{'%(len(tuning_names))+etabins_str[etaBinIdx]+'}',tuning_names[idx], 'Reference'] + ref_values   ,
                                                _contextManaged = False ) ]
                            tex_line += [ TableLine( columns = ['', tuning_names[idx+1], 'Cross Val.'] + cv_values    , _contextManaged = False ) ]
                            if test_table is not None:
                                tex_line += [ TableLine( columns = ['', tuning_names[idx+1], 'Test'] + test_values   , _contextManaged = False ) ]
                    
                    tex_line += [ HLine(_contextManaged = False) ]
                tex_line += [ HLine(_contextManaged = False) ]
                tex_lines.append(tex_line)

            page_eff_index = len(tex_lines)
            tex_line = []           

            ### Calculate the final efficiencies
            tex_line += [ HLine(_contextManaged = False) ]
            tex_line += [ HLine(_contextManaged = False) ]
            tex_line += [ TableLine( columns = ['',colorPD+r'$P_{D}[\%]$',colorPF+r'$F_{a}[\%]$'], _contextManaged = False ) ]
            tex_line += [ HLine(_contextManaged = False) ]
            for idx, tag in enumerate( train_tags ):
                current_table = self.integrate( best_inits, tag )
                pd = current_table['pd_val'].values[0]*100
                pd_std = current_table['pd_val'].values[1]*100
                fa = current_table['fa_val'].values[0]*100
                fa_std = current_table['fa_val'].values[1]*100

                pd_ref = current_table['pd_ref'].values[0]*100
                fa_ref = 0#current_table['fa_ref'].values[0]*100

                if test_table is not None:
                    keys = [ key for key in test_table.columns.values if 'passed' in key or 'total' in key]
                    current_test_table = test_table.loc[test_table.train_tag=='v8'].agg(dict(zip( keys, ['sum']*len(keys))))
                    test_pd = (current_test_table['pd_test_passed']/current_test_table['pd_test_total'])*100
                    test_fa = (current_test_table['fa_test_passed']/current_test_table['fa_test_total'])*100

                if idx > 0:
                    tex_line += [ TableLine( columns = [tag.replace('_','\_') + ' (Cross Val.)' ,
                        colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std) ], _contextManaged = False ) ]
                
                    if test_table is not None:
                        tex_line += [ TableLine( columns = [tag.replace('_','\_') + ' (Test)',
                        colorPD+('%1.2f')%(test_pd),colorPF+('%1.2f')%(test_fa) ], _contextManaged = False ) ]
                
                else:
                    tex_line += [ TableLine( columns = ['Ref.' ,colorPD+('%1.2f')%(pd_ref),colorPF+('%1.2f')%(fa_ref) ], _contextManaged = False ) ]
                    tex_line += [ TableLine( columns = [tag.replace('_','\_') + ' (Cross Val.)',
                                colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std) ], _contextManaged = False ) ]
                    if test_table is not None:
                        tex_line += [ TableLine( columns = [tag.replace('_','\_')+ ' (Test)' ,
                        colorPD+('%1.2f')%(test_pd),colorPF+('%1.2f')%(test_fa) ], _contextManaged = False ) ]
                    
            tex_lines.append(tex_line) # also append to page list, maybe it's not necessary

            # Create all tables into the PDF Latex
            for page_number,tex_line in enumerate(tex_lines): # skip last page
                if page_number == page_eff_index:
                    # The General Efficiency
                    with BeamerSlide( title = "The General Efficiency"  ):
                        with Table( caption = 'The general efficiency for the cross validation method for each method.') as table:
                            with ResizeBox( size = 0.7 ) as rb:
                                with Tabular( columns = 'lc' + 'c' * 2 ) as tabular:
                                    tabular = tabular
                                    for line in tex_line:
                                        if isinstance(line, TableLine):
                                            tabular += line
                                        else:
                                            TableLine(line, rounding = None)
                
                else:
                    # Cross Validation Efficiency
                    latex_etbins_str = etbins_str[page_number] if et_table_page_numbers>1 else etbins_str
                    with BeamerSlide( title = "The Cross Validation Efficiency Values For All Tunings"  ):
                        with Table( caption = 'The $P_{d}$, $F_{a}$ and $SP $ values for each phase space for each method.') as table:
                            with ResizeBox( size = 1. ) as rb:
                                with Tabular( columns = '|lcc|' + 'ccc|' * len(latex_etbins_str) ) as tabular:
                                    tabular = tabular
                                    for line in tex_line:
                                        if isinstance(line, TableLine):
                                            tabular += line
                                        else:
                                            TableLine(line, rounding = None)

    #
    # Load all keras models given the best sort table
    #
    def get_best_models( self, best_sorts , remove_last=True, with_history=True):
        '''
        This method will load the best models.

        Arguments:

        - best_sorts: the table that contains the best_sorts;
        - remove_last: a bolean variable to remove or not the tanh in tha output layer;
        - with_history: unused variable.
        '''
        from tensorflow.keras.models import Model, model_from_json
        import json

        models = [[ None for _ in range(len(self.__etabins)-1)] for __ in range(len(self.__etbins)-1)]
        for et_bin in range(len(self.__etbins)-1):
            for eta_bin in range(len(self.__etabins)-1):
                d_tuned = {}
                best = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin)]
                tuned = load_file(best.file_name.values[0])[best.tuned_idx.values[0]]
                model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) ) #custom_objects={'RpLayer':RpLayer} )
                model.set_weights( tuned['weights'] )
                new_model = Model(model.inputs, model.layers[-2].output) if remove_last else model
                #new_model.summary()
                d_tuned['model']    = new_model
                d_tuned['etBin']    = [self.__etbins[et_bin], self.__etbins[et_bin+1]]
                d_tuned['etaBin']   = [self.__etabins[eta_bin], self.__etabins[eta_bin+1]]
                d_tuned['etBinIdx'] = et_bin
                d_tuned['etaBinIdx']= eta_bin
                d_tuned['sort']     = best.sort.values[0]
                d_tuned['init']     = best.init.values[0]
                d_tuned['model_idx']= best.model_idx.values[0]
                d_tuned['file_name']= best.file_name.values[0]
                if with_history:
                    d_tuned['history']  = tuned['history']
                models[et_bin][eta_bin] = d_tuned
        return models





    #
    # Load all keras models given the best sort table
    #
    def get_best_init_models( self, best_inits , remove_last=True, with_history=True):
        '''
        This method will load the best init models.

        Arguments:

        - best_inits: the table that contains the best_inits;
        - remove_last: a bolean variable to remove or not the tanh in tha output layer;
        - with_history: unused variable.
        '''
        from tensorflow.keras.models import Model, model_from_json
        import json

        models = [[ [] for _ in range(len(self.__etabins)-1)] for __ in range(len(self.__etbins)-1)]

        for et_bin in range(len(self.__etabins)-1):
            for eta_bin in range(len(self.__etabins)-1):
                for sort in best_inits.sort.unique():
                    d_tuned = {}
                    best = best_inits.loc[(best_inits.et_bin==et_bin) & (best_inits.eta_bin==eta_bin) & (best_inits.sort==sort)]
                    tuned = load_file(best.file_name.values[0])[best.tuned_idx.values[0]]
                    model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) ) #custom_objects={'RpLayer':RpLayer} )
                    model.set_weights( tuned['weights'] )
                    new_model = Model(model.inputs, model.layers[-2].output) if remove_last else model
                    #new_model.summary()
                    d_tuned['model']    = new_model
                    d_tuned['etBin']    = [self.__etbins[et_bin], self.__etbins[et_bin+1]]
                    d_tuned['etaBin']   = [self.__etabins[eta_bin], self.__etabins[eta_bin+1]]
                    d_tuned['etBinIdx'] = et_bin
                    d_tuned['etaBinIdx']= eta_bin
                    d_tuned['sort']     = best.sort.values[0]
                    d_tuned['init']     = best.init.values[0]
                    d_tuned['model_idx']= best.model_idx.values[0]
                    d_tuned['file_name']= best.file_name.values[0]
                    if with_history:
                        d_tuned['history']  = tuned['history']
                    models[et_bin][eta_bin].append(d_tuned)
        return models
