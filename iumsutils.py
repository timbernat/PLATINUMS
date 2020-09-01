import csv, math, re, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# general use methods to ease may common tasks
def average(iterable, precision=None):
    '''Calculcate and return average of an iterable'''
    avg = sum(iterable)/len(iterable)
    if precision:
        avg = round(avg, precision)
    return avg

def standard_dev(iterable):
    '''Calculate the standard deviation of an iterable'''
    avg = average(iterable)  # done to avoid repeated calculation
    return ( sum((i - avg)**2 for i in iterable)/len(iterable) )**0.5

def get_by_filetype(extension):  # NOTE: make this check for correct formatting as well
    '''Get all files of a particular file type present in the current directory'''
    csvs_present = tuple(file for file in os.listdir() if re.search(f'.{extension}\Z', file))
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

def adagraph(plot_list, ncols, save_dir, display_size=20):  # ADD AXIS LABELS, SUBCLASS BUNDLED PLOT OBJECTS
        '''a general tidy internal graphing utility of my own devising, used to produce all manner of plots during training with one function'''
        nrows = math.ceil(len(plot_list)/ncols)  #  determine the necessary number of rows needed to accomodate the data
        fig, axs = plt.subplots(nrows, ncols, figsize=(display_size, display_size * nrows/ncols)) 
        
        for idx, (plot_data, plot_title, plot_type) in enumerate(plot_list):                         
            if nrows > 1:                        # locate the current plot, unpack linear index into coordinate
                row, col = divmod(idx, ncols)      
                curr_plot = axs[row][col]  
            else:                                # special case for indexing plots with only one row; my workaround of implementation in matplotlib
                curr_plot = axs[idx]    
            curr_plot.set_title(plot_title)
            
            # enumerate plot conditions by plot type, plan to generalize this later with a custom class BundledPlot()
            if plot_type == 's':                 # for plotting spectra
                curr_plot.plot(*plot_data, 'c-') # unpacking accounts for shift in x-boundaries if RIP trimming is occuring
            elif plot_type == 'm':               # for plotting metrics from training
                curr_plot.plot(plot_data, ('Loss' in plot_title and 'r-' or 'g-')) 
            elif plot_type == 'f':               # for plotting fermi-dirac plots
                curr_plot.plot(plot_data, 'm-')  
                curr_plot.set_ylim(0, 1.05)
            elif plot_type == 'p':               # for plotting predictions
                curr_plot.bar(*plot_data, color=('Summation' in plot_title and 'r' or 'b'))  # unpacking accounts for column labels by family for predictions
                curr_plot.set_ylim(0,1)
                curr_plot.tick_params(axis='x', labelrotation=45)
            elif plot_type == 'v':               # for plotting variation in noise by family
                (minima, averages, maxima) = plot_data
                curr_plot.set_xlabel('Point Number')
                curr_plot.set_ylabel('Intensity')
                curr_plot.plot(maxima, 'b-', averages, 'r-', minima, 'g-') 
                curr_plot.legend(['Maximum', 'Average', 'Minimum'], loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close('all')
        
        
# data cleaning and transformation methods-------------------------------------------------------------------------------------------------
def fourierize(source_file_name, cutoff=None, smooth_only=False):  
    '''Creates a copy of a PLATIN-UMS-compatible data file, with all spectra being replaced by their Discrete Fourier Transforms'''
    dest_file_name = f'{source_file_name}(FT{smooth_only and "S" or ""}).csv'
    with open(f'{source_file_name}.csv', 'r') as source_file, open(dest_file_name, 'w', newline='') as dest_file:
        for row in csv.reader(source_file):
            name, data = row[0], [float(i) for i in row[1:]]    # isolate the name and data
            fft_data = np.fft.hfft(data)                 # perform a Hermitian (real-valued) fast Fourier transform over the data
            if cutoff:
                blanks = np.zeros(np.size(fft_data)-cutoff) 
                fft_data = np.concatenate( (fft_data[:cutoff], blanks) )
            if smooth_only:
                fft_data = np.fft.ihfft(fft_data).real      # if smoothing, perform the inverse transform and return the real component
            csv.writer(dest_file).writerow( [name, *fft_data] ) # write the resulting row to the named target file
            
def filter_and_smooth(file_name, cutoff=0.5):  # this method is very much WIP, quality of normalization cannot be spoken for at the time of writing
    '''Duplicate a dataset, omitting all spectra whose maximum falls below the specified cutoff value and applying Savitzky-Golay Filtering'''
    with open(f'{file_name}.csv', 'r') as source_file, open(f'{file_name}(S).csv', 'w', newline='') as dest_file:
        for row in csv.reader(source_file):
            instance, spectrum = row[0], [float(i) for i in row[1:]]
            if max(spectrum) > cutoff:
                new_row = (instance, *savgol_filter(spectrum, 5, 1))  # create a new row with the filtered data after culling
                csv.writer(dest_file).writerow(new_row)

def analyze_noise(file_name, ncols=4):  # consider making min/avg/max calculations in-place, rather than after reading
    '''Generate a set of plots for all species in a dataset which shows how the baseline noise varies across spectra at each sample point'''
    with open(f'{file_name}.csv', 'r') as source_file:
        data_by_species = {}
        for row in csv.reader(source_file):
            instance, species, spectrum = row[0], isolate_species(row[0]), [float(i) for i in row[1:]]

            if species not in data_by_species:  # ensure entries exists to avoid KeyError
                data_by_species[species] = []
            data_by_species[species].append(spectrum)
    
    noise_plots = []
    for species, spectra in data_by_species.items():
        noise_stats = [tuple(map(funct, zip(*spectra))) for funct in (min, average, max)] # tuple containing the min, avg, and max values across all points in all current species' spectra
        plot_title = f'S.V. of {species}, ({len(spectra)} instances)'
        noise_plots.append((noise_stats, plot_title, 'v'))  # bundled plot for the current species

    adagraph(noise_plots, ncols, f'./Dataset Noise Plots/Spectral Variation by Species, {file_name}')
