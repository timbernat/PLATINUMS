import csv, json, math, re
from pathlib import Path
from collections import Counter

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
    filetypes_present = tuple(file.name for file in Path.cwd().iterdir() if file.suffix == extension)
    if filetypes_present == ():
        filetypes_present = (None,)
    return filetypes_present

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
        
        
# file conversion methods - csv to json and vice versa-------------------------------------------------------------------------------------------------
def jsonize(source_file_name): 
    '''Process spectral data csvs, generating labels, vector mappings, species counts, and other information,
    then cast the data to a json for ease of data reading in other applications and methods'''
    chem_data, species, families, family_mapping, spectrum_size, species_count = ({}, set(), set(), {}, 0, Counter())
    with open(f'{source_file_name}.csv', 'r') as csv_file:
        for row in csv.reader(csv_file):
            instance, curr_species, spectrum = row[0], isolate_species(row[0]), [float(i) for i in row[1:]]

            chem_data[instance] = spectrum
            species.add(curr_species)
            species_count[curr_species] += 1
            families.add(get_family(instance))
             
            if not spectrum_size: # record size of first spectrum, throw an error if any subsequent spectra are not the same size
                spectrum_size = len(spectrum)
            elif len(spectrum) != spectrum_size:
                raise Exception(ValueError)
                return
                    
    species, families = sorted(species), sorted(families)  # sort and convert to lists

    for family in families: # build a dict of onehot mapping vectors by family
        family_mapping[family] = tuple(int(i == family) for i in families)

    for instance, data in chem_data.items():  # add mapping vector to all data entries
        vector = family_mapping[get_family(instance)]
        chem_data[instance] = (data, vector)
        
    packaged_data = {   # package all the data into a single dict for json dumping
        'chem_data' : chem_data,
        'species' : species,
        'families' : families,
        'family_mapping' : family_mapping,
        'spectrum_size' : spectrum_size,
        'species_count' : species_count
    }
    with open(f'{source_file_name}.json', 'w') as json_file:
        json.dump(packaged_data, json_file) # dump our data into a json file with the same name as the original datacsv

def csvize(source_file_name):
    '''Inverse of jsonize, takes a processed chemical data json file and reduces it to a csv with just the listed spectra'''
    with open(f'{source_file_name}.json', 'r') as source_file, open(f'{source_file_name}.csv', 'w', newline='') as dest_file:
        json_data = json.load(source_file)
        for instance, data in json_data['chem_data'].items():
            row = [instance, *data[0]] # include only the spectrum (not the vector) after the name in the row
            csv.writer(dest_file).writerow(row)

# data transformation methods - note that these only work with jsons for simplicity, if you want a csv, then use csvize after the transformation
def base_transform(file_name, operation, discriminator=None, indicator='', prevent_overwrites=False, **opargs):
    '''The base method for transforming data, takes target file (always a .json) and a function to operate on each spectrum in the file.
    Optionally, a boolean-valued function over spectra can be passed as a discriminator to set criteria for removal of spectra in the transform'''
    source_file_name, dest_file_name = f'{file_name}.json', f'{file_name}{indicator}.json'
    if prevent_overwrites and Path(dest_file_name).exists():  # if overwrite prevention is enabled, throw an error instead of transforming
        raise Exception(FileExistsError)
    
    with open(source_file_name, 'r') as source_file, open(dest_file_name, 'w', newline='') as dest_file:
        json_data = json.load(source_file)
           
        temp_dict = {} # temporary dictionary is used to allow for deletion of chemdata entries (can't delete while iterating, can't get length in dict comprehension)
        for instance, (spectrum, vector) in json_data['chem_data'].items():
            if not discriminator or not discriminator(spectrum): # only perform the operation if no discriminator exists or if the discrimination criterion is unmet
                temp_dict[instance] = (operation(spectrum, **opargs), vector)
        json_data['chem_data'] = temp_dict
        json_data['spectrum_size'] = len(operation(spectrum, **opargs)) # takes the length to be that of the last spectrum in the set; all spectra are
        # guaranteed to be the same size by the jsonize method, so under a uniform transform, any change in spectrum size should be uniform throughout
        
        if discriminator:  # only necessary to recount families, species, and instances is spectra are being removed
            all_instances = json_data['chem_data'].keys()
            json_data['families'] = sorted( set(get_family(instance) for instance in all_instances) )
            json_data['species'] = sorted( set(isolate_species(instance) for instance in all_instances) )
            json_data['species_count'] = Counter(isolate_species(instance) for instance in all_instances)

        json.dump(json_data, dest_file) # dump the result in the new file


def fft_with_smoothing(spectrum, harmonic_cutoff=None, smooth_only=False):                                                      
    '''Performs a fast Fourier Transform over a spectrum; can optionally cut off higher frequencies, as well as
    converting the truncated frequency space into a (now noise-reduced) spectrum using the inverse transform'''
    fft_spectrum = np.fft.hfft(spectrum)  # perform a Hermitian (real-valued) fast Fourier transform over the data
    if harmonic_cutoff:
        blanks = np.zeros(np.size(fft_spectrum)-harmonic_cutoff) # make everything beyond the harmonic cut-off point zeros
        fft_spectrum = np.concatenate( (fft_spectrum[:harmonic_cutoff], blanks) )
    if smooth_only:
        fft_spectrum = np.fft.ihfft(fft_spectrum).real # keep only real part of inverse transform (imag part is 0 everywhere, but is kept complex type for some reason)
    return list(fft_spectrum)
    
def fourierize(file_name, harmonic_cutoff=None, smooth_only=False):  
    '''Creates a copy of a PLATIN-UMS-compatible data file, with all spectra being replaced by their Discrete Fourier Transforms'''
    base_transform(file_name, operation=fft_with_smoothing, indicator=f'(FT{smooth_only and "S" or ""})', harmonic_cutoff=harmonic_cutoff, smooth_only=smooth_only)

    
def sav_golay_smoothing(spectrum, window_length=5, polyorder=1):
    '''Wrapper to convert SG-smoothed ndarray into lists for packaging into jsons'''
    return list(savgol_filter(spectrum, window_length=window_length, polyorder=polyorder))
    
def filterize(file_name, cutoff=0.5):  # this method is very much WIP, quality of normalization cannot be spoken for at the time of writing
    '''Duplicate a dataset, omitting all spectra whose maximum falls below the specified cutoff value and applying Savitzky-Golay Filtering'''
    base_transform(file_name, operation=sav_golay_smoothing, discriminator=lambda spectrum : max(spectrum) < cutoff, indicator='(S)')

       
def analyze_noise(file_name, ncols=4):  # consider making min/avg/max calculations in-place, rather than after reading
    '''Generate a set of plots for all species in a dataset which shows how the baseline noise varies across spectra at each sample point'''
    with open(f'{file_name}.json', 'r') as source_file:
        json_data, data_by_species = json.load(source_file), {}
        for instance, (spectrum, vector) in json_data['chem_data'].items():
            species = isolate_species(instance)
            if species not in data_by_species:  # ensure entries exists to avoid KeyError
                data_by_species[species] = []
            data_by_species[species].append(spectrum)
    
    noise_plots = []
    for species, spectra in data_by_species.items():
        noise_stats = [tuple(map(funct, zip(*spectra))) for funct in (min, average, max)] # tuple containing the min, avg, and max values across all points in all current species' spectra
        plot_title = f'S.V. of {species}, ({len(spectra)} instances)'
        noise_plots.append((noise_stats, plot_title, 'v'))  # bundled plot for the current species
    adagraph(noise_plots, ncols, f'./Dataset Noise Plots/Spectral Variation by Species, {file_name}')
    
def analyze_fourier_smoothing(file_name, instance, cutoff, nrows=1, ncols=4, display_size=20, save_plot=False):
    '''Investigate the original spectrum, Fourier Spectrum, truncated Fourier Spectrum, and reconstructed truncated spectrum of a
    single instance in the specified dataset. Optionally, can save the figure to the current directory, if it is of interest'''
    with open(f'{file_name}.json', 'r') as json_file:
        json_data = json.load(json_file)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(display_size, display_size * nrows/ncols))
    
    orig_data = json_data['chem_data'][instance][0]
    axs[0].plot(orig_data, 'r-')
    axs[0].set_title('Original Spectrum')
    
    fft_data = np.fft.hfft(orig_data)
    axs[1].plot(fft_data)
    axs[1].set_title('FFT Spectrum')
    
    cut_fft_data = np.concatenate( (fft_data[:cutoff], np.zeros(np.size(fft_data)-cutoff)) )
    axs[2].plot(cut_fft_data)
    axs[2].set_title(f'FFT Spectrum (to point {cutoff})')
    
    rec_data = np.fft.ihfft(cut_fft_data).real
    axs[3].plot(rec_data, 'r-')
    axs[3].set_title(f'Reconstructed Spectrum (to point {cutoff})')
    
    plt.suptitle(instance)
    if save_plot:
        plt.savefig(f'Fourier Smoothing of {instance} (to point {cutoff})')
    else:
        plt.show()
    plt.close('all')
