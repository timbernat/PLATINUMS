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
def get_by_filetype(extension, path=Path.cwd()):  
    '''Get all files of a particular file type present in a given directory, (the current directory by default)'''
    filetypes_present = tuple(file.name for file in path.iterdir() if file.suffix == extension)
    if filetypes_present == ():
        filetypes_present = (None,)
    return filetypes_present
   
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

def one_hot_mapping(iterable):
    '''Takes and iterable and returns a dictionary of the values in the iterable, assigned sequentially to one-hot vectors
    each of which is the length of the iterable (akin to an identity matrix)'''
    items = [i for i in iterable] # temporarily store data, in the case that the iterable is a generator
    return {value : tuple(int(val == value) for val in items) for value in items}

def get_RIP(mode1_spectrum):
    '''Naive but surprisingly effective method for identifying the RIP value for Mode 1 spectra'''
    return max(mode1_spectrum[:len(mode1_spectrum)//2]) # takes the RIP to be the maximum value in the first half of the spectrum

# utilities for handling instance naming and information packaging
def sort_instance_names(name_list, key=lambda x:x):
    '''Sorts a a list of instance names in ascending order based on the tailing digits. Optional "key" arg for when some operation is needed to return the name (e.g. Instance.name)'''
    return sorted( name_list, key=lambda y : int(re.findall('[0-9]+\Z', key(y))[0]) )
        
def isolate_species(instance_name): # NOTE: consider expanding range of allowable strings in the future
    '''Strips extra numbers off the end of the name of an instance and just tells you its species'''
    return re.sub('(\s|-)\d+\s*\Z', '', instance_name)  # regex to crop terminal digits off of an instance in a variety of possible formats

def get_family(species):
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
    
class Instance:
    '''A single instance of a particular chemical, useful for storing and accessing chemical and spectral information during training and transformations '''
    def __init__(self, name, species, family, spectrum, vector):
        self.name = name
        self.species = species
        self.family = family
        self.spectrum = spectrum
        self.vector = vector
        
# utilities for plotting and for processing summary data post-training
class BundledPlot:
    '''Helper class for adagraph, makes it easier to package data which is to be plotted'''
    valid_plot_types = ('s', 'm', 'f', 'p', 'v')
    
    def __init__(self, data, title, plot_type='s', x_data=None): # s (for standard or spectrum) is the default plot type
        self.has_x_data = bool(x_data)
        self.data = (self.has_x_data and (x_data, data) or data) # if x_data is given, will package that data together
        self.title = title 
        # eventually, add marker defaults + customization
        
        if plot_type not in self.valid_plot_types:
            raise ValueError('Invalid plot type specified')
        else:
            self.plot_type = plot_type
        
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
                if plot.has_x_data: # unpack data, if necessary (meant specifically for plots with RIP slicing)
                    curr_plot.plot(*plot.data, 'c-') 
                else: 
                    curr_plot.plot(plot.data, 'c-') 
            elif plot.plot_type == 'm':               # for plotting neural netwrok training metrics 
                curr_plot.plot(plot.data, ('Loss' in plot.title and 'r-' or 'g-')) 
            elif plot.plot_type == 'f':               # for plotting fermi-dirac plots
                curr_plot.plot(plot.data, 'm-')  
                curr_plot.set_ylim(0, 1.05)
            elif plot.plot_type == 'p':               # for plotting predictions
                curr_plot.bar(*plot.data, color=('Summation' in plot.title and 'r' or 'b'))  # unpacking accounts for column labels by family for predictions
                curr_plot.set_ylim(0,1)
                curr_plot.tick_params(axis='x', labelrotation=45)
            elif plot.plot_type == 'v':               # for plotting variational summary/noise plots
                (minima, averages, maxima) = plot.data
                curr_plot.set_xlabel('Point Number')
                curr_plot.set_ylabel('Intensity')
                curr_plot.plot(maxima, 'b-', averages, 'r-', minima, 'g-') 
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
        self.process_insts() # since we know all instances have been input, we can process immediately
            
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
        if not self.fermi_plot:
            self.process_insts() # process instances if it has not already been done
        instance_plots = [BundledPlot(data, name, plot_type='p', x_data=self.family_mapping.keys()) # bundle the name-prediction pairs into my custom plot objects
                          for name, data in sort_instance_names(self.inst_mapping.items(), key=lambda x:x[0])] # ensure data is in ascending order by name
        
        plot_list = [*prepended_plots, self.fermi_plot, self.summation_plot, *instance_plots]
        adagraph(plot_list, ncols=ncols, save_dir=save_dir) 
        
def unpack_summaries(species_summaries, save_dir, indicator=None):
    '''Takes a list of species summaries (presume)'''
    try:
        families = SpeciesSummary.family_mapping.keys() # ensure that the summary class has a mapping present and use it to enumerate the families
    except ValueError:
        raise ValueError('SpeciesSummary.family_mapping has not been instantiated')
    
    fermi_summary = [spec_sum.fermi_plot for spec_sum in species_summaries] # collect together all the fermi plots
    adagraph(fermi_summary, ncols=5, save_dir=save_dir/'Fermi Summary.png') # plot the fermi summary

    score_folder_name = f'{indicator and (indicator + " ") or ""}Scores.csv' # prepend an optional indicator, so that multiple score files can be opened at once
    with open(save_dir/score_folder_name, 'w', newline='') as score_file: # open the score file and unpack scores by family                         
        for family in families:
            processed_scores = [(spec_sum.species, spec_sum.score) for spec_sum in species_summaries if spec_sum.family == family] 
            if processed_scores: # only write scores if the family is actually present
                processed_scores.sort(key=lambda x : x[1], reverse=True) # zip scores together and sort in ascending order by score
                processed_scores.append( ('AVERAGE', average(pair[1] for pair in processed_scores)) ) # score in second position (still bundled to preserve pairing)

                score_file.write(family) 
                for name, score in processed_scores:
                    score_file.write(f'\n{name}, {score}')
                score_file.write('\n\n')   # leave a gap between each family
