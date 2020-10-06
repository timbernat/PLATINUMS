import math, re
from pathlib import Path
import matplotlib.pyplot as plt
   
def average(iterable, precision=None): 
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
        
def get_by_filetype(extension):  
    '''Get all files of a particular file type present in the current directory'''
    filetypes_present = tuple(file.name for file in Path.cwd().iterdir() if file.suffix == extension)
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
    return tuple( (i - min(iterable))/(max(iterable) - min(iterable)) for i in iterable)

def one_hot_mapping(iterable):
    '''Takes and iterable and returns a dictionary of the values in the iterable, assigned sequentially to one-hot vectors
    each of which is the length of the iterable (akin to an identity matrix)'''
    items = [i for i in iterable] # temporarily store data, in the case that the iterable is a generator
    return {value : tuple(int(val == value) for val in items) for value in items}


class Instance:
    '''A single instance of a particular chemical, useful for storing and accessing chemical and spectral information during training and transformations '''
    def __init__(self, name, species, family, spectrum, vector):
        self.name = name
        self.species = species
        self.family = family
        self.spectrum = spectrum
        self.vector = vector

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
        if re.search(f'(?i){suffix}\Z', isolate_species(species)):   #ignore capitalization (particular to ethers), only check end of name (particular to pinac<ol>one)
            return family
    else:
        return 'Unknown'
        
def get_RIP(mode1_spectrum):
    '''Naive but surprisingly effective method for identifying the RIP value for Mode 1 spectra'''
    return max(mode1_spectrum[:len(mode1_spectrum)//2]) # takes the RIP to be the maximum value in the first half of the spectrum
              

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
