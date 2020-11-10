import csv, json, math, random, re
import numpy as np
from pathlib import Path

from iumsutils import *

def check_filetype(path, ext):
    if path.suffix != ext:
        raise TypeError(f'Input must be a(n) {ext} file')

# base method for reading in jsons and parsing instance data
def load_chem_json(target_path):
    '''Read a chemical data json, de-serializes the Instance objects from "chem_data", and return the contents of the file'''
    check_filetype(target_path, '.json')
    with target_path.open() as source_file:
        json_data = json.load(source_file) # this comment is a watermark - 2020, timotej bernat
        json_data['chem_data'] = [Instance(*properties) for properties in json_data['chem_data']] # unpack the properties into Instance objects
    return json_data

# methods for converting back and forth between csvs and jsons
def jsonize(target_path, correct_names=False): 
    '''Process spectral data csvs, generating labels, vector mappings, species counts, and other information,
    then cast the data to a json for ease of data reading in other applications and methods'''
    check_filetype(target_path, '.csv')
    rep_flags = {'MIBK' : 'Methyl-iBu-Ketone', # dictionary of names to flag and replace to ensure total consistency of naming between files
                'Propanol' : '1-Propanol',     # add flags as they come up, these are the ones for Modes 1-3 I've come across so far
                'Butanol'  : '1-Butanol',
                'Pentanol' : '1-Pentanol',
                'Hexanol'  : '1-Hexanol',
                'Heptanol' : '1-Heptanol',
                'Octanol'  : '1-Octanol',
                'IsoButanol'  : 'Isobutanol',
                'Iso-Butanol' : 'Isobutanol',
                'Sec Butyl Acetate' : 'Sec-Butyl Acetate',
                'Secbutyl Acetate'  : 'Sec-Butyl Acetate'} 
    temp_dict = {}
    with target_path.open() as csv_file:
        for row in csv.reader(csv_file):
            name, spectrum = row[0], [float(i) for i in row[1:]] # isolate the instance name and spectrum for ease of reference
            if correct_names:
                species = isolate_species(name) # regex only replaces if string occurs at beginning (to avoid the "2-1-Propanol" bug)
                name = re.sub(f'\A{species}', rep_flags.get(species, species), name) # replaced string will only be different if it appears in the flags dict     

            try: # error checking to ensure all spectra are of the same size - based entirely on the first spectrum's length
                if len(spectrum) != spectrum_size: 
                    raise ValueError(f'Spectrum of {name} is of different length to the others')
            except NameError: # take spectrum_size to be the length of the first spectrum encountered (in that case, spectrum_size is as yet unassigned)
                spectrum_size = len(spectrum)
                
            temp_dict[name] = spectrum # if all checks and corrections are passed, map the name to the spectrum
    
    species, species_count = ordered_and_counted(isolate_species(instance) for instance in temp_dict.keys())
    families, family_count = ordered_and_counted(get_family(instance) for instance in temp_dict.keys())      
    family_mapping = one_hot_mapping(families)  # dict of onehot mapping vectors by family   
    chem_data = [(name, isolate_species(name), get_family(name), spectrum, family_mapping[get_family(name)]) for name, spectrum in temp_dict.items()] 
    
    packaged_data = {   # package all the data into a single dict for json dumping
        'chem_data' : chem_data,
        'species'   : species,
        'families'  : families,
        'family_mapping' : family_mapping,
        'spectrum_size'  : spectrum_size,
        'species_count'  : species_count,
        'family_count'   : family_count
    }
    
    dest_path = target_path.parent/f'{target_path.stem}{correct_names and "(@)" or ""}.json' # add indicator to target name if correcting names
    dest_path.touch() # create the new file
    with dest_path.open(mode='w') as json_file:
        json.dump(packaged_data, json_file) # dump our data into a json file with the same name as the original datacsv
        
def csvize(target_path):
    '''Inverse of jsonize, takes a processed chemical data json file and reduces it to a csv with just the listed spectra'''
    check_filetype(target_path, '.json')
    json_data = load_chem_json(target_path) 
    dest_path = target_path.parent/f'{target_path.stem}(C).csv' # add "C" indicator to denote that this file has been csvized
    dest_path.touch()
    with dest_path.open(mode='w', newline='') as dest_file:
        for instance in json_data['chem_data']:
            csv.writer(dest_file).writerow([instance.name, *instance.spectrum]) # merge name and spectral data into a single row and write it to the csv
            

# data transformation methods (for jsons only) - if you want a csv, then use csvize after the transformation
# NOTE: actual transforms will always end in the suffiz "-ize", while any helper methods will not
def base_transform(target_path, operator=None, discriminator=None, indicator='', **opargs):
    '''The base method for transforming data, takes a .json data file name, an optional operator to modify spectra (takes spectra and optional arguments),
    an optional discriminator to omit spectra if some condition is met (takes an Instance object and the full chem_data list as arguments), 
    and an optional indicator to denote that a tranform has occurred. NOTE: tranformed data is written to a new file, ORIGINAL DATA IS READ ONLY'''
    json_data = load_chem_json(target_path)
    temp_data = []
    for instance in json_data['chem_data']:
        if not (discriminator and discriminator(instance)): # omit instance only when there is a discriminator present AND its condition is met
            if operator:
                instance.spectrum = operator(instance.spectrum, **opargs) # operate on the spectrum if an operator is given
            temp_data.append(instance)
    
    json_data['spectrum_size'] = len(instance.spectrum) # takes size to be that of the last spectrum (jsonize guarantees uniform length, so a uniform transformation shouldn't change that)
    if discriminator:  # only when spectra are being omitted might it be necessary to recount species and families
        json_data['species'], json_data['species_count'] = ordered_and_counted(instance.species for instance in temp_data)
        json_data['families'], json_data['family_count'] = ordered_and_counted(instance.family for instance in temp_data)
        
        if json_data['family_mapping'].keys() != json_data['families']: # if the families present have changed, must redo the family mapping as well
            json_data['family_mapping'] = one_hot_mapping(json_data['families']) # rebuild the family mapping
            for instance in temp_data:
                instance.vector = json_data['family_mapping'][instance.family] # all mapping vectors based on the new mapping         
    json_data['chem_data'] = [list(instance.__dict__.values()) for instance in temp_data] # flatten and listify the Instance objects to prepare for JSON serialization
    
    dest_path = target_path.parent/f'{target_path.stem}{indicator}.json'
    dest_path.touch()
    with dest_path.open(mode='w') as dest_file: # this comment is a watermark - 2020, timotej bernat
        json.dump(json_data, dest_file) # dump the result in the new file

# some basic tranform operations
def duplicate(target_path):
    '''Makes a duplicate of a json dataset in the same directory'''
    base_transform(target_path, indicator='(D)') # with no operator or discriminator, all items are copied verbatim
    
def truncate(target_path, cutoff):
    '''Truncates all spectra below some cutoff'''
    base_transform(target_path, operator=lambda spectrum : spectrum[:cutoff], indicator='(T)')
          
def filterize(target_path, cutoff=0.5): 
    '''Removes all spectra whose maximum falls below a specified cutoff value'''
    base_transform(target_path, discriminator=lambda instance : max(instance.spectrum) < cutoff, indicator='(S)')      

def baseline_standardize(target_path, lower=0, upper=20, base_value=0): # if a non-zero baseline is chosen, this will be reflected in the indicator
    '''Baseline standardizes a dataset. Takes a spectrum, two bounds, and a desired baseline value and uses the average noise in the region specified between the
    two bounds to center the overall baseline around the desired value. NOTE: it is CRITICAL that the bounds denote a region containing ONLY NOISE (NO PEAKS!)''' 
    base_transform(target_path, operator=lambda spectrum : [point - average(spectrum[lower:upper]) + base_value for point in spectrum], indicator=f'(B{base_value and base_value or ""})') 
    

# transforms that require helper methods, usually to gain extra info from the whole dataset    
def get_RIP_cutoffs(target_path, lb=0.15, ub=0.95):
    '''Takes a dataset and normalized cutoff bounds and returns the RIP values, for each species, which correspond to those bounds'''  
    json_data = load_chem_json(target_path)
    
    RIP_cutoffs = {}
    for species in json_data['species']:
        RIPs = sorted(get_RIP(instance.spectrum) for instance in json_data['chem_data'] if instance.species == species)
        lower_cutoff, *middle, upper_cutoff = [val for val, norm_val in zip(RIPs, normalized(RIPs)) if lb < norm_val < ub]
        RIP_cutoffs[species] = (lower_cutoff, upper_cutoff)        
    return RIP_cutoffs

def mode1_filterize(target_path, lower_bound=0.15, upper_bound=0.95):
    '''Filtering regime specific to Mode 1, will not work with other Modes, and Mode 1 sets should not be used with other filtering regimes.
    Culls all spectra whose RIP lies outside of some prescribed normalized bounds for the RIP for that particular species'''
    if 'Mode 1' not in target_path.stem: # ensure this transform is not applied to data for which it is not compatible
        raise TypeError('File is not a Mode 1 dataset')
        
    if not (0 < lower_bound < 1) or not (0 < upper_bound < 1): # some error checking to ensure that the imposed limits make sense
        raise ValueError('Limit(s) should be between 0 and 1')
    elif lower_bound > upper_bound:
        raise ValueError('Limits are mismatched')
        
    limits = get_RIP_cutoffs(target_path, lb=lower_bound, ub=upper_bound) # the dictionary of the RIP cutoffs by species
    base_transform(target_path, discriminator=lambda instance : not (limits[instance.species][0] < get_RIP(instance.spectrum) < limits[instance.species][1]), indicator='(SM1)')
    

def get_reduction_listing(target_path, lower_cap=60, upper_cap=80):
    '''Within a chemical data file, if the number of instances of a given species exceed the upper_cap, the number of instances to be kept will be randomly selected
    within a range from the upper to the lower cap, and that number of instances will be randomly selected to be kept. Returns a list of all the instances (by name) to keep'''
    json_data = load_chem_json(target_path)
    
    kept_count = {species : (count < upper_cap and count or random.randint(lower_cap, upper_cap)) for species, count in json_data['species_count'].items()}
    random.shuffle(json_data['chem_data'])
    
    kept_listing = []
    for instance in json_data['chem_data']:
        if kept_count[instance.species] > 0:
            kept_listing.append(instance.name)
            kept_count[instance.species] -= 1   
    return kept_listing

def reductize(target_path, lower_cap=60, upper_cap=80):
    '''Reduces a dataset such that no species has more than the "upper_cap" amount of instances, in a doubly-random and bias-free way'''
    instances_to_keep = get_reduction_listing(target_path, lower_cap=lower_cap, upper_cap=upper_cap)
    base_transform(target_path, discriminator=lambda instance : instance.name not in instances_to_keep, indicator='(R)')

    
fold = lambda chem_data, funct, **kwargs : funct((funct(instance.spectrum, **kwargs) for instance in chem_data), **kwargs) # useful for finding single smallest point in a dataset, for example
    
def logarithmize(target_path):
    '''Finds the absolute minimum point of a baseline-standardized dataset, makes this the new baseline (to ensure all points are positive and avoid a log domain error)
    and takes the natural log over all spectra (exaggerates relative differences even further)'''
    if '(B' not in target_path.stem: # consider using regex for this check (numerical value in baseline indicator is variable)
        raise TypeError('Transform must be performed over baseline-standardized data')       
    chem_data = load_chem_json(target_path)['chem_data']    
    eps_baseline = -fold(chem_data, min) + np.finfo(float).eps # the minimum non-biased/equitable baseline that guarantees all data are positive    
    base_transform(target_path, operator=lambda spectrum : [math.log(point + eps_baseline) for point in spectrum], indicator='(L)')
    
    
# Fourier-Transform transformation methods
def fourier(spectrum, cutoff=None):                                                      
    '''Returns the real-valued FFT of a spectrum. Optionally, can clear all frequencies in the transform above some cutoff'''
    fft_spectrum = np.fft.hfft(spectrum)  # perform a Hermitian (real-valued) fast Fourier transform over the data
    if cutoff:
        fft_spectrum[cutoff:] = 0   # set everything above the cutoff to 0 (if cutoff is specified) to preserve spectrum size upon inverse tranform
    return list(fft_spectrum) # must return as a list in order to be json serializable

inv_fourier = lambda spectrum : list(np.fft.ihfft(spectrum).real) # returns real-valued inverse thransform as a serializable list

def fourierize(target_path, cutoff=None): # if no cutoff is given, will simply yield the full spectra
    '''Replaces spectra in a set with their Fourier Tranforms (Hermitian and real-valued)'''
    if '(FT)' in target_path.stem:
        raise TypeError('Input cannot already be Fourierized')   
    base_transform(target_path, operator=fourier, indicator=f'(FT{cutoff and cutoff or ""})', cutoff=cutoff) 

def inv_fourierize(target_path): # cutoff is list index of highest point to keep
    '''Replaces spectra in a set with their Fourier Tranforms (Hermitian and real-valued)'''
    if '(FT)' not in target_path.stem:
        raise TypeError('Input must first be Fourierized')   
    base_transform(target_path, operator=inv_fourier, indicator='(IFT)') # transform is converted to list (np arrays are not JSON serializable)  
    
def fourier_filterize(target_path, cutoff):  # combines functionality of fourierize (with cutoff) and the inverse transform
    '''Reduces high-frequency noise in a dataset'''
    base_transform(target_path, operator=lambda spectrum : inv_fourier(fourier(spectrum, cutoff=cutoff)), indicator=f'(SFT{cutoff})')
    
    
# analysis and data characterization methods---------------------------------------------------------------------------------------------------------------------------
def inspect_spectra(target_path, species, ncols=6, save_path=None):
    '''Plot the spectra of all instances of one species in the chosen dataset'''
    json_data = load_chem_json(target_path)
    if species not in json_data['species']:
        raise ValueError(f'Species "{species}" not in dataset')
        
    plots = [BundledPlot(data=instance.spectrum, title=instance.name, plot_type='s')
             for instance in json_data['chem_data'] if instance.species == species]
    adagraph(plots, ncols=ncols, save_dir=save_path)

def inspect_variation(target_path, ncols=4, save_folder=Path('Dataset Noise Plots')):  # consider making min/avg/max calculations in-place, rather than after reading
    '''Generate a set of plots for all species in a dataset which shows how the baseline noise varies across spectra at each sample point'''
    if not save_folder.is_dir():
        raise ValueError('save_folder does not point to a folder')
    
    json_data = load_chem_json(target_path)
    plots = [BundledPlot(data=[instance.spectrum for instance in json_data['chem_data'] if instance.species == species],
                         title=f'SV of {species}, ({json_data["species_count"][species]} instances)', # NOTE: leave the double quotes (they're deliberate!)
                         plot_type='v')
            for species in json_data['species']] # one plot per species       
    adagraph(plots, ncols, save_folder/f'{target_path.stem} Species-wise PWAs')
    
def inspect_fsmoothing(target_path, inst, initial_cutoff=0, n_steps=1, step_size=1, save_figure=False):
    '''Investigate the original spectrum, Fourier Spectrum, truncated Fourier Spectrum, and reconstructed truncated spectrum of a
    single instance in the specified dataset. Optionally, can save the figure to the current directory, if it is of interest'''
    if 'FT' in target_path.stem:
        raise TypeError('Method only applies to non-Fourier Transformed data') 
   
    json_data = load_chem_json(target_path)
    for instance in json_data['chem_data']:
        if instance.name == inst:
            orig_data = instance.spectrum
            fft_data = np.fft.hfft(orig_data)
            break
    else:
        raise ValueError(f'Instance "{inst}" not in dataset')
    
    max_cutoff = initial_cutoff + nsteps*step_size 
    if max_cutoff > fft_data.size:
        raise ValueError('The resulting amount of cutoffs exceeds the FFT data size')
    
    plots = [BundledPlot(data=orig_data, title='Original Spectrum')]
    for cutoff in range(initial_cutoff, max_cutoff, step_size):
        cut_fft_data = np.fft.hfft(orig_data)
        cut_fft_data[cutoff:] = 0 
        plots.append(BundledPlot(data=np.fft.ihfft(cut_fft_data).real, title=f'Reconstructed Spectrum  (to freq. {cutoff})'))

    if nsteps == 1:
        plots.insert(1, BundledPlot(data=fft_data, title='FFT Spectrum'))
        plots.insert(2, BundledPlot(data=cut_fft_data, title=f'FFT Spectrum (to freq. {cutoff})'))
    else:
        plots.append(BundledPlot(data=np.fft.ihfft(fft_data).real, title='Fully Reconstructed Spectrum'))
 
    save_dir = (save_figure and f'{cutoff and "Singular" or "Ranged"} Fourier Smoothing of {inst}' or None)
    adagraph(plots, ncols=4, save_dir=save_dir)

def inspect_fourier_maxima(target_path, save_folder=None):
    '''Plot all Fourier maxima (e.g. the baseline magnitudes) by family, in the order the families appear in the data'''
    if '(FT)' not in target_path.stem:
        raise ValueError('Method only applies to Fourier-Transformed datasets')
    
    json_data = load_chem_json(target_path)
    plots = [BundledPlot(data=[max(instance.spectrum) for instance in json_data['chem_data'] if instance.family == family], title=family)
                                                       for family in json_data['families']]     
    if save_folder:
        save_folder = save_folder/f'Fourier Maxima by Family - {target_path.stem}'
    adagraph(plots, ncols=5, save_dir=save_folder)
