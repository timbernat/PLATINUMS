import math, re
from pathlib import Path
import matplotlib.pyplot as plt
 
# utilities specifically written to avoid having to import entire modules for a single object's functionality
def average(iterable, precision=4): 
    '''Calculate and return average of an iterable'''
    isum, n = 0, 0
    for i in iterable: # iterates, rather than using sum/len, so that generators work as inputs
        isum += i
        n += 1
    avg = isum/n
    return (precision and round(avg, precision) or avg)

def standard_dev(iterable): # eventually make this use a more computationally efficient stdev formula
    '''Calculate the standard deviation of an iterable'''
    avg = average(iterable)  # done to avoid repeated calculation of average for every term
    return (sum((i - avg)**2 for i in iterable)/len(iterable))**0.5

def format_time(sec):
    '''Converts a duration in seconds into an h:mm:ss string; written explicitly to avoid importing datetime.timedelta'''
    minutes, seconds = divmod(round(sec), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'

def counter(iterable):
    '''Takes an iterable and returns a dict of the number of occurrences of each item; written explicitly to avoid importing collections.Counter'''
    counts = {}
    for item in iterable:
        counts[item] = counts.get(item, 0) + 1
    return counts
        
# some general-purpose utilities
def ordered_and_counted(iterable):
    '''Takes an iterable of items and returns a sorted set of the items, and a dict of the counts of each item
    Specifically useful for getting the listing and counts of both species and families when jsonizing or transforming'''
    data = [i for i in iterable] # temporarily store data, in the case that the iterable is a generator
    return sorted(set(data)), counter(data)

def normalized(iterable):
    '''Normalize an iterable using min-max feature scaling (casts all values between 0 and 1)'''
    try:
        return tuple( (i - min(iterable))/(max(iterable) - min(iterable)) for i in iterable)
    except ZeroDivisionError: # if all data have the same value, max=min and min/max normalization will fail
        return tuple(i for i in iterable) # in that case, just return the original data
    
def dictmerge(dictlist):
    '''Takes a list of dictionaries with identical keys and combines them into a single dictionary with the values combined into lists under each entry'''
    return {key : [subdict[key] for subdict in dictlist] for key in dictlist[0]}

def one_hot_mapping(iterable):
    '''Takes and iterable and returns a dictionary of the values in the iterable, assigned sequentially to one-hot vectors
    each of which is the length of the iterable (akin to an identity matrix)'''
    items = [i for i in iterable] # temporarily store data, in the case that the iterable is a generator
    return {value : tuple(int(val == value) for val in items) for value in items}

def get_RIP(mode1_spectrum):
    '''Naive but surprisingly effective method for identifying the RIP value for Mode 1 spectra'''
    return max(mode1_spectrum[:len(mode1_spectrum)//2]) # takes the RIP to be the maximum value in the first half of the spectrum

#file and path utilities
def get_by_filetype(extension, path=Path.cwd()):  
    '''Get all files of a particular file type present in a given directory, (the current directory by default)'''
    if type(path) == str:
        path = Path(path) # convert any string input (i.e. just the name) into Path object
    
    filetypes_present = tuple(file.stem for file in path.iterdir() if file.suffix == extension)
    if filetypes_present == ():
        filetypes_present = (None,)
    return filetypes_present

def clear_folder(path):
    '''Clear out the contents of a folder. A more tactful approach than without deleting the folder and remaking it'''
    if not path.is_dir():
        raise ValueError(f'{path} does not point to a folder')
    
    for file in path.iterdir():
        if file.is_dir():
            clear_folder(file) # recursively clear any subfolders, as path can only delete empty files
            try:
                file.rmdir()
            except OSError: # meant for the case where the file won't be deleted because the user is still inside it
                raise PermissionError # convert to permission error (which my file checkers are built to handle)
        else:
            file.unlink()

# utilities for handling instance naming and information packaging
def sort_instance_names(name_list, data_key=lambda x:x):
    '''Sorts a a list of instance names in ascending order based on the tailing digits. Optional "key" arg for when some operation is needed to return the name (e.g. Instance.name)'''
    return sorted( name_list, key=lambda y : int(re.findall('[0-9]+\Z', data_key(y))[0]) )
        
def isolate_species(instance_name): # NOTE: consider expanding range of allowable strings in the future
    '''Strips extra numbers off the end of the name of an instance and just tells you its species'''
    return re.sub('(\s|-)\d+\s*\Z', '', instance_name)  # regex to crop terminal digits off of an instance in a variety of possible formats

def get_family(species): # while called species, this method works with instance names as well
    '''Takes the name of a species OR of an instance and returns the chemical family that that species belongs to;
    determination is based on IUPAC naming conventions by suffix'''
    iupac_suffices = {  'ate':'Acetates', # Esters might be preferable outside the context of the current datasets
                        'ol':'Alcohols',
                        'al':'Aldehydes',
                        'ane':'Alkanes',
                        'ene':'Alkenes',
                        'yne':'Alkynes',
                        'ine':'Amines',
                        'oic acid': 'Carboxylic Acids',
                        'ether':'Ethers',
                        'one':'Ketones'  }                    
    for suffix, family in iupac_suffices.items():
        if re.search(f'(?i){suffix}\Z', isolate_species(species)): # ignore capitalization (particular to ethers), only check end of name (particular to pinac<ol>one)
            return family
    else:
        return 'Unknown'
    
def get_carbon_ordering(species):
    '''Naive method to help with ordering compound names based on carbon number and a handful of prefices, used to ensure cinsistent sorting by species name.
    NOTE that the number this method assigns is not precisely the carbon number, but an analog that allows for numerical ordering in the desired manner'''
    iupac_numbering = {'meth' : 1,
                       'eth(?!er)' : 2, # prevents all ethers from being assigned "2"
                       'prop' : 3,
                       'but'  : 4,
                       'pent' : 5,
                       'hex'  : 6,
                       'hept' : 7,
                       'oct'  : 8,
                       'non(?!e)'  : 9, # prevents all ketones from being assigned "9"
                       'dec'  : 10}
    for affix, number in iupac_numbering.items():
        if re.search(f'(?i){affix}', species): # ignore capitalization (finds affix anywhere in word)
            return number + 0.5*bool(re.search(f'(?i)(iso|sec-){affix}', species)) # places "iso" and "sec-" compounds slightly lower on the list (+0.5, between compounds)
    else:
        return 100 # arbitrary, needs to return a number much greater than the rest to be placed at end
    
class Instance:
    '''A single instance of a particular chemical, useful for storing and accessing chemical and spectral information during training and transformations '''
    def __init__(self, name, species, family, spectrum, vector):
        self.name     = name
        self.species  = species
        self.family   = family
        self.spectrum = spectrum
        self.vector   = vector
        
# utilities for plotting and for processing summary data post-training
class BundledPlot:
    '''Helper class for adagraph, makes it easier to package data which is to be plotted'''
    valid_plot_types = ('s', 'm', 'f', 'p', 'v')
    
    def __init__(self, data, title, plot_type='s', x_data=None): # s (for standard or spectrum) is the default plot type
        if plot_type not in self.valid_plot_types:
            raise ValueError('Invalid plot type specified')

        self.plot_type = plot_type
        if self.plot_type == 'v': # special cases needed for calculating pointwise aggregate variation plots
            self.data = [tuple(map(funct, zip(*data))) for funct in (min, average, max)]
            self.x_data = (x_data and x_data or range(len(data[0]))) # if no x_data is specified, default to just range over number of points
        else:
            self.data  = data # otherwise, leave data as is
            self.x_data = (x_data and x_data or range(len(data))) # if no x_data is specified, default to just range over data 
            
        self.title = title 
        # eventually, add marker defaults + customization
        
def adagraph(plot_list, ncols=6, save_dir=None, display_size=20):  # consider adding axis labels
        '''a general tidy internal graphing utility of my own devising, used to produce all manner of plots during training with one function'''
        nrows = math.ceil(len(plot_list)/ncols)  #  determine the necessary number of rows needed to accomodate the data
        fig, axs = plt.subplots(nrows, ncols, figsize=(display_size, display_size * nrows/ncols)) 
        
        for idx, plot in enumerate(plot_list):                         
            if nrows > 1:                        # locate the current plot, unpack linear index into coordinate
                row, col = divmod(idx, ncols)      
                curr_plot = axs[row][col]  
            else:                                # special case for indexing plots with only one row; my workaround of implementation in matplotlib
                curr_plot = axs[idx]    
            curr_plot.set_title(plot.title)
            
            # specifying plot conditions by plot type
            if plot.plot_type == 's':                 # for plotting spectra or just generic plots
                curr_plot.plot(plot.x_data, plot.data, 'c-') 
            elif plot.plot_type == 'm':               # for plotting neural netwrok training metrics 
                curr_plot.plot(plot.x_data, plot.data, ('Loss' in plot.title and 'r-' or 'g-')) 
            elif plot.plot_type == 'f':               # for plotting fermi-dirac plots
                curr_plot.plot(plot.x_data, plot.data, 'm-')  
                curr_plot.set_ylim(0, 1.05)
            elif plot.plot_type == 'p':               # for plotting predictions
                curr_plot.bar(plot.x_data, plot.data, color=('Summation' in plot.title and 'r' or 'b'))  # unpacking accounts for column labels by family for predictions
                curr_plot.set_ylim(0,1)
                curr_plot.tick_params(axis='x', labelrotation=45)
            elif plot.plot_type == 'v':               # for plotting variational summary/noise plots
                (minima, averages, maxima) = plot.data
                curr_plot.set_xlabel('Point Number')
                curr_plot.set_ylabel('Intensity')
                curr_plot.plot(plot.x_data, maxima, 'b-', plot.x_data, averages, 'r-', plot.x_data, minima, 'g-') 
                curr_plot.legend(['Maximum', 'Average', 'Minimum'], loc='upper left')
        plt.tight_layout()
        if not save_dir: # consider adding a show AND plot option, not at all necessary now, however
            plt.show()
        else:
            plt.savefig(save_dir)
        plt.close('all')
              
class SpeciesSummary:
    '''Helper class for processing raw name-prediction data into species-based plots, scores, fermi data, and other useful information'''
    family_mapping = None # class variable, set for all SpeciesSummary objects prior to use
    
    def __init__(self, species):
        if not self.family_mapping: # requires a class-wide mapping for ease of functionality
            raise ValueError('SpeciesSummary.family_mapping has not been instantiated')
        
        self.species = species
        self.family = get_family(species)
        self.hotbit = self.family_mapping[self.family].index(1)
        self.inst_mapping = {}
        
        self.score = None
        self.fermi_plot = None
        self.summation_plot = None
        
    def add_inst(self, name, pred):
        if isolate_species(name) == self.species:
            self.inst_mapping[name] = pred
        else:
            return # don't add instances that are of a different species to the prediction registry
        
    def add_all_insts(self, name_list, predictions):
        '''The self.add() operation, but over two whole lists instead, saves time and code'''
        for name, pred in zip(name_list, predictions):
            self.add_inst(name, pred)
            
    def process_insts(self):
        '''Compute the species score and assemble the fermi and summation plots''' 
        predictions = self.inst_mapping.values()
        corr_array = [max(pred) == pred[self.hotbit] for pred in predictions]
        self.score = average(corr_array)          
        self.summation_plot = BundledPlot([average(column) for column in zip(*predictions)], 'Standardized Summation', plot_type='p', x_data=self.family_mapping.keys())

        fermi_data = normalized( sorted((pred[self.hotbit] for pred in predictions), reverse=True) )
        fermi_title = f'{self.species}, {sum(corr_array)}/{len(corr_array)} correct'
        self.fermi_plot = BundledPlot(fermi_data, fermi_title, plot_type='f')

    def graph(self, save_dir=None, ncols=6, prepended_plots=[]):
        '''Produce the result plot collection for the species'''
        if not self.fermi_plot:
            self.process_insts() # process instances if it has not already been done
        instance_plots = [BundledPlot(data, name, plot_type='p', x_data=self.family_mapping.keys()) # bundle the name-prediction pairs into my custom plot objects
                          for name, data in sort_instance_names(self.inst_mapping.items(), data_key=lambda x:x[0])] # ensure data is in ascending order by name
        
        plot_list = [*prepended_plots, self.fermi_plot, self.summation_plot, *instance_plots]
        adagraph(plot_list, ncols=ncols, save_dir=save_dir) 

class FamilySummary:
    '''Helper class for lumping together the SpeciesSummary objects from a given dataset into score sheets and fermi summaries. Used as well for autocollation'''
    families = None # class variable, set for all FamilyMapping objects prior to use
    
    def __init__(self, source_file, fam_status):
        if not self.families:
            raise ValueError('FamilySummary.families has not been instantiated')
            
        self.source_file = source_file
        self.fam_status  = fam_status
    
        self.scores = {family : [] for family in self.families}        
        self.fermi_summary = []
        self.is_processed = False      

    def add_specsum(self, specsum):
        self.scores[specsum.family].append((specsum.species, specsum.score))
        self.fermi_summary.append(specsum.fermi_plot)
        
    def add_all_specsums(self, specsum_list):
        '''The self.add() operation, but over a whole list instead, saves time and code'''
        for specsum in specsum_list:
            self.add_specsum(specsum)
        
    def process_specsums(self):
        if self.is_processed: # do not reprocess data if processing has already been done (is time consuming and leads to data corruption)
            return
        else:
            self.scores = {family : score_pairs for family, score_pairs in self.scores.items() if score_pairs} # remove any families with no members
            for family, score_pairs in self.scores.items():
                ordered_scores = sorted(score_pairs, key=lambda x : get_carbon_ordering(x[0])) # arrange scores entries by carbon ordering
                self.scores[family] = {species : score for (species, score) in ordered_scores} # map to second-level dict
                self.scores[family]['AVERAGE'] = average(self.scores[family].values()) # append the average value after sorting
            self.is_processed = True # raise processed flag if all goes well
        
    def unpack_summaries(self, save_dir, indicator=''):
        '''Takes a list of species summaries, unpacks them into score sheets and fermi plots summaries, and writes them to the relevant directory'''
        self.process_specsums() # ensure results are properly processed before attempting unpacking
        adagraph(self.fermi_summary, ncols=5, save_dir=save_dir/'Fermi Summary.png') # plot the fermi summary
        
        score_file_path = save_dir/f'{indicator}{bool(indicator)*" "}Scores.csv' # prepend an optional indicator, to differentiate files and allow them to be viewed at the same time
        with score_file_path.open(mode='w', newline='') as score_file: # open the score file and unpack scores by family                         
            for family, species_scores in self.scores.items(): # iterate through species score mapping (skips over omitted families)     
                score_file.write(family) 
                for species, score in species_scores.items():
                    score_file.write(f'\n{species}, {score}')
                score_file.write('\n\n')   # leave a gap between each family
                
class SummaryCollator:
    '''Highest level of summary organization and unpacking, creates data-wide collated summaries for multiple data sets, slices, and familiar cyclings'''
    def __init__(self):
        self.collated_scores = {}
        self.datafiles = [] # the word "data" will always be the first entry regardless (mainly as informative padding)
        self.is_processed = False
       
    def add_famsum(self, famsum):
        if famsum.source_file not in self.datafiles: # can do this naively because of the predictable cycling structure of the main PLATINUMS app
            self.datafiles.append(famsum.source_file)
            
        if famsum.fam_status not in self.collated_scores: # for first entry, expand score values into lists which can be filled in
            self.collated_scores[famsum.fam_status] = []
        self.collated_scores[famsum.fam_status].append(famsum.scores)
  
    def add_all_famsums(self, famsum_list):
        '''The self.add() operation, but over a whole list instead, saves time and code'''
        for famsum in famsum_list:
            self.add_famsum(famsum)
            
    def process_famsums(self):
        if self.is_processed: # do not reprocess data if processing has already been done (is time consuming and leads to data corruption)
            return
        else:    
            self.datafiles = ','.join(self.datafiles) + '\n' # merge the datafiles into a single string
            for fam_status, fam_data in self.collated_scores.items(): # collapses species entries into lists, still placed under the appropraite families
                self.collated_scores[fam_status] = {f'{family}\n' : [f'{species},{",".join(str(i) for i in scores)}\n' # turn score entry lists into easily writable strings
                                                      for species, scores in dictmerge(spec_data).items()]    # riffle together species scores
                                                        for family, spec_data in dictmerge(fam_data).items()} # riffle together family data
        self.is_processed = True # raise processed flag if all goes well
    
    def unpack_summaries(self, save_dir, point_range): # use point range as indicator
        self.process_famsums() # ensure score are already processed
        file_path = save_dir/f'Compiled Results - {point_range}.csv'  # write separate collated results file for each point range
        file_path.touch()
        
        with file_path.open(mode='w') as coll_file:
            for fam_status, fam_data in self.collated_scores.items(): # write two separate blocks, for each set of familiars/unfamiliars
                coll_file.write(f'{fam_status},{self.datafiles}')
                for family, spec_scores in fam_data.items():
                    coll_file.write(family)
                    for scoreset in spec_scores:
                        coll_file.write(scoreset)
                    coll_file.write('\n') # leave gap between each family (will also separate familirs and unfamiliars)
