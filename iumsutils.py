import csv, math, re, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def average(iterable):
    '''Calculcate and return average of an iterable'''
    return sum(iterable)/len(iterable)

def standard_dev(iterable):
    '''Calculate the standard deviation of an iterable'''
    avg = average(iterable)  # done to avoid repeated calculation
    return ( sum((i - avg)**2 for i in iterable)/len(iterable) )**0.5

def get_csvs():  # NOTE: make this check for correct formatting as well
    '''Get all csv files present in the current directory'''
    csvs_present = tuple(file for file in os.listdir() if re.search('.csv\Z', file))
    if csvs_present == ():
        csvs_present = (None,)
    return csvs_present

def isolate_species(instance_name): # NOTE: consider expanding range of allowable strings in the future
    '''Strips extra numbers off the end of the name of an instance in a csv and just tells you its species'''
    return re.sub('(\s|-)\d+\s*\Z', '', instance_name)  # regex to crop terminal digits off of an instance in a variety of possible formats

def get_family(species):
    '''Takes the name of a species OR of an instance and returns the chemical family that that species belongs to;
    determination is based on IUPAC naming conventions by suffix'''
    iupac_suffices = {  'ate':'Acetates',
                        'ol':'Alcohols',
                        'al':'Aldehydes',
                        'ane':'Alkanes',
                        'ene':'Alkenes',
                        'yne':'Alkynes',
                        'ine':'Amines',
                        'oic acid': 'Carboxylic Acids',
                        #'oate':'Esters',
                        'ether':'Ethers',
                        'one':'Ketones'  }                    
    for suffix, family in iupac_suffices.items():
        if re.search(f'(?i){suffix}\Z', isolate_species(species)):   #ignore capitalization (particular to ethers), only check end of name (particular to pinac<ol>one)
            return family
        
def fourierize(source_file_name):  
    '''Creates a copy of a PLATIN-UMS-compatible data file with all spectra data replaced by their Discrete Fourier Transforms'''
    dest_file_name = re.sub('.csv', '(FT).csv', source_file_name) # explicit naming for the new file
    with open(source_file_name, 'r') as source_file, open(dest_file_name, 'w', newline='') as dest_file:
        for row in csv.reader(source_file):
            name, data = row[0], [float(i) for i in row[1:]]    # isolate the name and data
            data_fft = np.abs(np.fft.fft(data))                 # perform a fast Fourier transform over the data
            csv.writer(dest_file).writerow( [name, *data_fft] ) # write the resulting row to the named target file
            
def normalize(file_name, cutoff_value=0.5):  # this method is very much WIP, quality of normalization cannot be spoken for at the time of writing
    '''Duplicate a dataset, omitting all spectra whose maximum falls below the specified cutoff value and applying Savitzky-Golay Filtering'''
    with open(f'{file_name}.csv', 'r') as source_file, open(f'{file_name}(S).csv', 'w', newline='') as dest_file:
        for row in csv.reader(source_file):
            instance, spectrum = row[0], [float(i) for i in row[1:]]
            if max(spectrum) > cutoff_value:
                new_row = (instance, *savgol_filter(spectrum, 5, 1))  # create a new row with the filtered data after culling
                csv.writer(dest_file).writerow(new_row)

def analyze_noise(file_name, ncols=4, display_size=20):
    '''Generate a set of plots for all species in a dataset which shows how the baseline noise varies across spectra at each sample point'''
    spectrum_len = 0
    with open(f'{file_name}.csv', 'r') as source_file:
        data_by_species = {}
        for row in csv.reader(source_file):
            instance, species, spectrum = row[0], isolate_species(row[0]), [float(i) for i in row[1:]]
            
            if not spectrum_len:
                spectrum_len = len(spectrum)
            
            if species not in data_by_species:  # ensure entries exists to avoid KeyError
                data_by_species[species] = []
            data_by_species[species].append(spectrum)
     
    nrows = math.ceil(len(data_by_species)/ncols)  #  determine the necessary number of rows needed to accomodate the data
    fig, axs = plt.subplots(nrows, ncols, figsize=(display_size, display_size * nrows/ncols))

    for idx, (species, spectra) in enumerate(data_by_species.items()):
        # zip(*spectra) is akin to matrix transposition, gives a list of the values of a single point for every spectra of a species
        (minima, averages, maxima) = [ tuple( map(funct, zip(*spectra)) ) for funct in (min, average, max)]
        point_nums = range(spectrum_len)
        
        if nrows > 1:                        # locate the current plot, unpack linear index into coordinate
            row, col = divmod(idx, ncols)      
            curr_plot = axs[row][col]  
        else:                                # special case for indexing plots with only one row; my workaround of implementation in matplotlib
            curr_plot = axs[idx]   

        curr_plot.set_title('Spectral Variation of ' + species)
        curr_plot.set_xlabel('Point Number')
        curr_plot.set_ylabel('Intensity')
        curr_plot.plot(point_nums, maxima, 'b-', label='Maximum')
        curr_plot.plot(point_nums, averages, 'r-', label='Average')
        curr_plot.plot(point_nums, minima, 'g-', label='Minimum') 
        curr_plot.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./Dataset Noise Plots/Spectral Variation by Species, {file_name}')
    plt.close()
