import csv, json, math, re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# general-use methods to ease may common tasks
def format_time(sec):
    '''Converts a duration in seconds into an h:mm:ss string.
    written explicitly to avoid importing datetime.timedelta'''
    minutes, seconds = divmod(round(sec), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'

def average(iterable, precision=None):
    '''Calculate and return average of an iterable'''
    avg = sum(iterable)/len(iterable)
    if precision:
        avg = round(avg, precision)
    return avg

def standard_dev(iterable):
    '''Calculate the standard deviation of an iterable'''
    avg = average(iterable)  # done to avoid repeated calculation of average for every term
    return (sum((i - avg)**2 for i in iterable)/len(iterable))**0.5

def normalized(iterable):
    '''Normalize an iterable using min-max feature scaling (casts all values between 0 and 1)'''
    return tuple( (i - min(iterable))/(max(iterable) - min(iterable)) for i in iterable)

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

def adagraph(plot_list, ncols=6, save_dir=None, display_size=20):  # ADD AXIS LABELS, SUBCLASS BUNDLED PLOT OBJECTS
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
        if not save_dir: # consider adding a show AND plot option, not at all necessary now, however
            plt.show()
        else:
            plt.savefig(save_dir)
        plt.close('all')
        
        
# file conversion methods - csv to json and vice versa-------------------------------------------------------------------------------------------------
def jsonize(source_file_name): 
    '''Process spectral data csvs, generating labels, vector mappings, species counts, and other information,
    then cast the data to a json for ease of data reading in other applications and methods'''
    with open(f'{source_file_name}.csv', 'r') as csv_file:
        data_dict = {row[0] : [float(i) for i in row[1:]] for row in csv.reader(csv_file)} # pull out a dict of instance : spectrum pairs to circumvent using up the reader generator
        
    for spectrum in data_dict.values():
        try: 
            if len(spectrum) != spectrum_size: # throw an error if any of the spectra are a different size than the others
                raise ValueError('Spectra must all be of the same length')
        except NameError: # take spectrum_size to be the length of the first spectrum encountered (in that case, spectrum_size is as yet unassigned)
            spectrum_size = len(spectrum)
    
    species = sorted(set(isolate_species(instance) for instance in data_dict.keys()))   # sorted list of one of each species
    species_count = Counter(isolate_species(instance) for instance in data_dict.keys()) # dict of the number of each species
    
    families = sorted(set(get_family(instance) for instance in data_dict.keys()))       # sorted list of one of each family
    family_count = Counter(get_family(instance) for instance in data_dict.keys())       # dict of the number of each family
    family_mapping = {family : tuple(int(i == family) for i in families) for family in families}  # dict of onehot mapping vectors by family
    
    chem_data = {}
    for instance, spectrum in data_dict.items():
        curr_species = isolate_species(instance)
        if curr_species not in chem_data:
            chem_data[curr_species] = {}
        chem_data[curr_species][instance] = (spectrum, family_mapping[get_family(instance)])
        
    packaged_data = {   # package all the data into a single dict for json dumping
        'chem_data' : chem_data,
        'species' : species,
        'families' : families,
        'family_mapping' : family_mapping,
        'spectrum_size' : spectrum_size,
        'species_count' : species_count,
        'family_count' : family_count
    }
    with open(f'{source_file_name}.json', 'w') as json_file:
        json.dump(packaged_data, json_file) # dump our data into a json file with the same name as the original datacsv
        
def csvize(source_file_name):
    '''Inverse of jsonize, takes a processed chemical data json file and reduces it to a csv with just the listed spectra'''
    with open(f'{source_file_name}.json', 'r') as source_file, open(f'{source_file_name}.csv', 'w', newline='') as dest_file:
        json_data = json.load(source_file)
        for species, instances in json_data['chem_data'].items():
            for instance, data in jinstances.items():
                row = [instance, *data[0]] # include only the spectrum (not the vector) after the name in the row
                csv.writer(dest_file).writerow(row)

            
# data transformation methods - note that these only work with jsons for simplicity, if you want a csv, then use csvize after the transformation
def base_transform(file_name, operation=lambda x : x, discriminator=None, indicator='', prevent_overwrites=False, **opargs):
    '''The base method for transforming data, takes target file (always a .json) and a function to operate on each spectrum in the file.
    Optionally, a boolean-valued function over spectra can be passed as a discriminator to set criteria for removal of spectra in the transform'''
    source_file_name, dest_file_name = f'{file_name}.json', f'{file_name}{indicator}.json'
    if prevent_overwrites and Path(dest_file_name).exists():  # if overwrite prevention is enabled, throw an error instead of transforming
        raise FileExistsError('Overwrite permission denied in function call')
    
    with open(source_file_name, 'r') as source_file, open(dest_file_name, 'w', newline='') as dest_file:
        json_data = json.load(source_file) # this comment is a watermark - 2020, timotej bernat
           
        temp_dict = {} # temporary dictionary is used to allow for deletion of chemdata entries (can't delete while iterating, can't get length in dict comprehension)
        for species, instances in json_data['chem_data'].items():
            if species not in temp_dict:
                temp_dict[species] = {} # ensure entries for each species exists in the new dict
            for instance, (spectrum, vector) in instances.items():
                if not discriminator or not discriminator(instance, spectrum):    # only perform the operation if no discriminator exists or if the discrimination criterion is unmet
                    temp_dict[species][instance] = (operation(spectrum, **opargs), vector) # if no operation is passed, spectra will remain unchanged
        json_data['chem_data'] = temp_dict                              
        json_data['spectrum_size'] = len(operation(spectrum, **opargs)) # takes the length to be that of the last spectrum in the set; all spectra are
        # guaranteed to be the same size by the jsonize method, so under a uniform transform, any change in spectrum size should also be uniform throughout
        
        if discriminator:  # only necessary to recount families, species, and instances if spectra are being removed
            all_instances = [instance for instances in json_data['chem_data'].values() for instance in instances.keys()]
            json_data['families'] = sorted( set(map(get_family, all_instances)) )
            json_data['family_count'] = Counter( map(get_family, all_instances) )
            
            json_data['species'] = sorted( set(map(isolate_species, all_instances)) )
            json_data['species_count'] = Counter( map(isolate_species, all_instances) )

        json.dump(json_data, dest_file) # dump the result in the new file

        
def filterize(file_name, cutoff=0.5): 
    '''Most naive possible transform, removes all spectra whose maximum falls below the specified cutoff value (applied indiscriminately to all instances)'''
    base_transform(file_name, discriminator=lambda instance, spectrum : max(spectrum) < cutoff, indicator='(S)')      
        

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

    
def get_RIP(mode1_spectrum):
    '''Naive but surprisingly effective method for identifying the RIP value for Mode 1 spectra'''
    return max(mode1_spectrum[:len(mode1_spectrum)//2]) # takes the RIP to be the maximum value in the first half of the spectrum
        
def get_RIP_cutoffs(file_name, lower_limit=0.15, upper_limit=0.95): 
    '''Helper method for filtering Mode 1 data specifically. Takes a json file and normalized lower and upper limits (from 0 to 1)
    and returns a dict (by species) of the lower and upper RIP cutoff values corresponding to these limits'''
    if 'Mode 1' not in file_name: # ensure this is not applied to data for which it is not compatible
        raise TypeError('File is not a Mode 1 dataset')
        
    if lower_limit > 1 or upper_limit > 1 or lower_limit > upper_limit: # ensure limits passed actually make sense
        raise ValueError('Limit(s) exceed 1 or are mismatched')
    
    with open(f'{file_name}.json', 'r') as file:
        json_data = json.load(file)

    # largest value in the first half of the spectrum is taken to be the RIP
    RIP_ranges = {species : sorted(get_RIP(spectrum) for (spectrum, vector) in instances.values()) # order and normalize all RIP values of a given species species
                                                     for species, instances in json_data['chem_data'].items()}   # do this for all species in the data
    
    for species, RIP_list in RIP_ranges.items():
        lower_cutoff, *middle, upper_cutoff = [val for i, val in enumerate(RIP_list) if lower_limit < normalized(RIP_list)[i] < upper_limit] # return ORIGINAL values if normalized values are within the cutoff range
        RIP_ranges[species] = (lower_cutoff, upper_cutoff) # throw out the midpoints within the range and only return the endpoints as cutoffs
    
    return RIP_ranges  

def filterize_mode1(file_name, lower_limit=0.15, upper_limit=0.95, species_cap=80):
    '''Filtering regime specific to Mode 1, will not work with other Modes, and Mode 1 sets should not be used with other filtering regimes.
    Culls all spectra whose RIP lies outside of some prescribed normalized bounds for the RIP for that particular species'''
    RIP_cutoffs = get_RIP_cutoffs(file_name, lower_limit=lower_limit, upper_limit=upper_limit)
    
    def discriminator(instance, spectrum):
        '''Discriminator function for this regime, necessary to define this as a function-in-a-function, as the parameters
        will change depending on the limits passed, opted for def rather than lambda for readability'''
        lower_cutoff, upper_cutoff = RIP_cutoffs[isolate_species(instance)] # find the correct cutoffs by species for the current instance
        return not (lower_cutoff < get_RIP(spectrum) < upper_cutoff) # flag if RIP is not within these cutoffs
    
    base_transform(file_name, discriminator=discriminator, indicator='(S1)')
    

# analysis and data characterization methods
def inspect_spectra(file_name, species, ncols=6, save_dir=None):
    '''Plot the spectra of all instances of one species in the chosen dataset'''
    with open(f'{file_name}.json') as json_file:
        json_data = json.load(json_file)

    if species not in json_data['species']:
        raise ValueError(f'Species "{species}" not in dataset')
        
    plot_list = [ ((range(len(spectrum)), spectrum), instance, 's') for instance, (spectrum, vector) in json_data['chem_data'][species].items()]
    adagraph(plot_list, ncols=ncols, save_dir=save_dir)

def analyze_noise(file_name, ncols=4):  # consider making min/avg/max calculations in-place, rather than after reading
    '''Generate a set of plots for all species in a dataset which shows how the baseline noise varies across spectra at each sample point'''
    with open(f'{file_name}.json', 'r') as source_file:
        chem_data = json.load(source_file)['chem_data']
        data_by_species = {species : [spectrum for (spectrum, vector) in instances.values()] 
                                for species, instances in chem_data.items()}
    
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
    
    orig_data = json_data['chem_data'][isolate_species(instance)][instance][0]
    axs[0].plot(orig_data, 'c-')
    axs[0].set_title('Original Spectrum')
    
    fft_data = np.fft.hfft(orig_data)
    axs[1].plot(fft_data, 'm-')
    axs[1].set_title('FFT Spectrum')
    
    cut_fft_data = np.concatenate( (fft_data[:cutoff], np.zeros(np.size(fft_data)-cutoff)) )
    axs[2].plot(cut_fft_data, 'm-')
    axs[2].set_title(f'FFT Spectrum (to point {cutoff})')
    
    rec_data = np.fft.ihfft(cut_fft_data).real
    axs[3].plot(rec_data, 'c-')
    axs[3].set_title(f'Reconstructed Spectrum (to point {cutoff})')
    
    plt.suptitle(instance)
    if save_plot:
        plt.savefig(f'Fourier Smoothing of {instance} (to point {cutoff})')
    else:
        plt.show()
    plt.close('all')
    
def analyze_fsmoothing_range(file_name, instance, step_size, ncols=7):
    '''Investigate the Fourier-smoothed spectra for a given instance over a range of harmonic cutoff points'''
    if 'FT' in file_name:
        raise TypeError('Method only applies to non-Fourier Transformed data')
    
    plot_list = []
    with open(f'{file_name}.json', 'r') as json_file:
        json_data = json.load(json_file)
    
    x_range = range(json_data['spectrum_size'])
    orig_data = json_data['chem_data'][isolate_species(instance)][instance][0]
    plot_list.append( ((x_range, orig_data), f'Original Spectrum ({instance})', 's') )
    
    fft_data = np.fft.hfft(orig_data)
    for cutoff in range(step_size, fft_data.size, step_size):
        cut_fft_data = np.concatenate( (fft_data[:cutoff], np.zeros(np.size(fft_data)-cutoff)) ) 
        rec_data = tuple(np.fft.ihfft(cut_fft_data).real)
        plot_list.append( ((x_range, rec_data), f'Reconstructed Spectrum  (to freq. {cutoff})', 's') )
    plot_list.append( ((x_range, np.fft.ihfft(fft_data).real), 'Fully Reconstructed Spectrum', 's') )
    
    adagraph(plot_list, ncols, f'Incremented FTSmoothing of {instance}')
