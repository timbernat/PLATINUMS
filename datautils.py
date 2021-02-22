import json, math, random
import numpy as np # at some point, consider replacing all data-wide spectral operations with numpy for spped and cleanliness
from pathlib import Path

from iumsutils import *
from plotutils import *


indicators = { # registry of the indicators being used for various transforms for reference, to be used to prevent collisions
    'roundize' : 'R', # method names as keys   
    'filterize' : 'S', 
    'reductize' : 'R--',
    'duplicatize': 'D', 
    'truncatize' : 'T',
    'fourierize' : 'FT<cutoff>',
    'positivize' : '+',
    'logarithmize' : 'L',
    'inv_fourierize' : 'IFT<cutoff>',
    'name_filterize' : 'N',
    'inv_fourierize' : 'IFT<cutoff>',
    'mode1_filterize' : 'SM1<norm_range>',
    'fourier_filterize' : 'SFT<cutoff>',
    'intensity_filterize' : 'I<cutoff>',
    'baseline_standardize' : 'B<baseval>'    
}

# data transformation methods (for jsons only). NOTE: actual transforms will always end in the suffiz "-ize", while any helper methods will not
def base_transform(source_path, operator=None, discriminator=lambda x : False, indicator='', **opargs):
    '''The base method for transforming data, takes a .json data file name, an optional operator to modify spectra (takes spectra and optional arguments),
    an optional discriminator to omit spectra if some condition is met (takes an Instance object and the full chem_data list as arguments), 
    and an optional indicator to denote that a tranform has occurred. NOTE: tranformed data is written to a new file, ORIGINAL DATA IS READ ONLY'''
    json_data = load_chem_json(source_path)
    json_data['chem_data'] = [
        (operator and instance._replace(spectrum = operator(instance.spectrum, **opargs)) or instance) # operate on the spectrum if an operator is given   
            for instance in json_data['chem_data']
                if not discriminator(instance)] # omit instance when discriminator condition is met
    
    # takes size to be that of the last spectrum (jsonize guarantees uniform length, unchanged by uniform transformation )
    json_data['spectrum_size'] = len(json_data['chem_data'][-1].spectrum)
    
    if discriminator:  # only when spectra are being omitted might it be necessary to recount species and families
        json_data['species'], json_data['species_count'] = ordered_and_counted(instance.species for instance in json_data['chem_data'])
        json_data['families'], json_data['family_count'] = ordered_and_counted(instance.family for instance in json_data['chem_data'])
        
        if json_data['family_mapping'].keys() != json_data['families']: # if the families present have changed, must redo the family mapping as well
            json_data['family_mapping'] = one_hot_mapping(json_data['families']) # rebuild the family mapping
            for i, instance in enumerate(json_data['chem_data']):
                json_data['chem_data'][i] = instance._replace(vector = json_data['family_mapping'][instance.family]) # reassign mapping vectors based on the new mapping     
    
    source_path = sanitized_path(source_path) # ensure Pathlike object pointing to json
    dest_path = source_path.parent/f'{source_path.stem}{indicator}.json'
    dest_path.touch()
    with dest_path.open(mode='w') as dest_file: # this comment is a watermark - 2020, timotej bernat
        json.dump(json_data, dest_file) # dump the result in the new file

# some basic tranform operations
def duplicate(source_path):
    '''Makes a duplicate of a json dataset in the same directory'''
    base_transform(source_path, indicator='(D)') # with no operator or discriminator, all items are copied verbatim
    
def truncatize(source_path, cutoff):
    '''Truncates all spectra below some cutoff'''
    base_transform(source_path, operator=lambda spectrum : spectrum[:cutoff], indicator='(T)')
          
def filterize(source_path, cutoff=0.5): 
    '''Removes all spectra whose maximum falls below a specified cutoff value'''
    base_transform(source_path, discriminator=lambda instance : max(instance.spectrum) < cutoff, indicator='(S)')    
    
def roundize(source_path, precision=6):
    '''Rounds all spectral datapoints to the passed number of decimal places (default 6)'''
    base_transform(source_path, operator=lambda spectrum : [round(i, precision) for i in spectrum], indicator='(R)')
    
def name_filterize(source_path, species_list):
    '''Takes a list of species and removes all instances of each species from the dataset'''
    base_transform(source_path, discriminator=lambda instance : instance.species in species_list, indicator='(N)')

def baseline_standardize(source_path, lower=0, upper=20, base_value=0): # if a non-zero baseline is chosen, this will be reflected in the indicator
    '''Baseline standardizes a dataset. Takes a spectrum, two bounds, and a desired baseline value and uses the average noise in the region specified between the
    two bounds to center the overall baseline around the desired value. NOTE: it is CRITICAL that the bounds denote a region containing ONLY NOISE (NO PEAKS!)''' 
    base_transform(source_path, operator=lambda spectrum : [point - average(spectrum[lower:upper]) + base_value for point in spectrum], indicator=f'(B{base_value and base_value or ""})') 
    

# transforms that require helper methods, usually to gain extra info from the whole dataset    
def norm_index(source_path, operator, lower_bound=0.15, upper_bound=0.95):
    '''Takes a dataset, an operation to apply over spectra, and normalized cutoff bounds and
    returns a dict (by species) of the ranges of data falling within those normalized bounds'''  
    if not (0 <= lower_bound < 1) or not (0 < upper_bound <= 1): # some error checking to ensure that the imposed limits make sense
        raise ValueError('Limit(s) should be between 0 and 1')
    elif lower_bound > upper_bound:
        raise ValueError('Limits are mismatched')
    
    json_data = load_chem_json(source_path)  
    
    bounds = {}
    for species in json_data['species']:
        op_spectra = sorted(operator(instance.spectrum)
                                for instance in json_data['chem_data']
                                    if instance.species == species) # operate over all instances of a species
        lower_cutoff, *middle, upper_cutoff = [val for val, norm_val in zip(op_spectra, normalized(op_spectra)) if lower_bound < norm_val < upper_bound] # discard middle values
        bounds[species] = (lower_cutoff, upper_cutoff)        
    return bounds

def mode1_filterize(source_path, lower_bound=0.15, upper_bound=0.95):
    '''Filtering regime specific to Mode 1, will not work with other Modes, and Mode 1 sets should not be used with other filtering regimes.
    Culls all spectra whose RIP lies outside of some prescribed normalized bounds for the RIP for that particular species'''
    if 'Mode 1' not in str(source_path): # ensure this transform is not applied to data for which it is not compatible
        raise TypeError('File is not a Mode 1 dataset')
        
    RIP_cutoffs = norm_index(source_path, get_RIP, lower_bound=lower_bound, upper_bound=upper_bound) # the dictionary of the RIP cutoffs by species
    base_transform(source_path, discriminator=lambda instance : not (RIP_cutoffs[instance.species][0] < get_RIP(instance.spectrum) < RIP_cutoffs[instance.species][1]), indicator=f'(SM1 {int(lower_bound*100)}-{int(upper_bound*100)})')
    
def intensity_filterize(source_path, cutoff=0.3):
    '''More sophisticated version of filterize, removes all spectra below some intensity on the basis of a normalized cutoff'''
    max_cutoffs = norm_index(source_path, max, lower_bound=cutoff, upper_bound=1) # only care about removing those below the cutoff (upper bound will always be 1)
    base_transform(source_path, discriminator=lambda instance : not (max_cutoffs[instance.species][0] < max(instance.spectrum) < max_cutoffs[instance.species][1]), indicator=f'(I-{int(100*cutoff)})')
    

def get_reduction_listing(source_path, lower_cap=60, upper_cap=80):
    '''Within a chemical data file, if the number of instances of a given species exceed the upper_cap, the number of instances to be kept will be randomly selected
    within a range from the upper to the lower cap, and that number of instances will be randomly selected to be kept. Returns a list of all the instances (by name) to keep'''
    json_data = load_chem_json(source_path)
    
    kept_count = {species : (count < upper_cap and count or random.randint(lower_cap, upper_cap)) for species, count in json_data['species_count'].items()}
    random.shuffle(json_data['chem_data'])
    
    kept_listing = []
    for instance in json_data['chem_data']:
        if kept_count[instance.species] > 0:
            kept_listing.append(instance.name)
            kept_count[instance.species] -= 1   
    return kept_listing

def reductize(source_path, lower_cap=60, upper_cap=80):
    '''Reduces a dataset such that no species has more than the "upper_cap" amount of instances, in a doubly-random and bias-free way'''
    instances_to_keep = get_reduction_listing(source_path, lower_cap=lower_cap, upper_cap=upper_cap)
    base_transform(source_path, discriminator=lambda instance : instance.name not in instances_to_keep, indicator='(R--)')

    
fold = lambda chem_data, funct, **kwargs : funct((funct(instance.spectrum, **kwargs) for instance in chem_data), **kwargs) # useful for finding single smallest point in a dataset, for example
   
def positivize(source_path): # consider omitting entirely, leads to awkward floating point errors and is questionably useful
    '''Finds the absolute minimum of a dataset and if it is negative, raises all data points by that value to ensure no values are below 0'''
    abs_min = fold(load_chem_json(source_path)['chem_data'], min)
    if abs_min < 0: # only perform transform if absolute minimum is actually negative
        base_transform(source_path, operator=lambda spectrum : [i - abs_min for i in spectrum], indicator='(+)')
    else:
        return 'Data already positive'
    
def logarithmize(source_path):
    '''Finds the absolute minimum point of a baseline-standardized dataset, makes this the new baseline (to ensure all points are positive and avoid a log domain error)
    and takes the natural log over all spectra (exaggerates relative differences even further)'''
    if '(B' not in str(source_path): # consider using regex for this check (numerical value in baseline indicator is variable)
        raise TypeError('Transform must be performed over baseline-standardized data')       
    chem_data = load_chem_json(source_path)['chem_data']    
    eps_baseline = -fold(chem_data, min) + np.finfo(float).eps # the minimum non-biased/equitable baseline that guarantees all data are positive    
    base_transform(source_path, operator=lambda spectrum : [math.log(point + eps_baseline) for point in spectrum], indicator='(L)') 
    
    
# Fourier-Transform transformation methods
def fourier(spectrum, cutoff=None):                                                      
    '''Returns the real-valued FFT of a spectrum. Optionally, can clear all frequencies in the transform above some cutoff'''
    fft_spectrum = np.fft.hfft(spectrum)  # perform a Hermitian (real-valued) fast Fourier transform over the data
    if cutoff:
        fft_spectrum[cutoff:] = 0   # set everything above the cutoff to 0 (if cutoff is specified) to preserve spectrum size upon inverse tranform
    return fft_spectrum.tolist() # must return as a list in order to be json serializable

inv_fourier = lambda spectrum : np.fft.ihfft(spectrum).real.tolist() # returns real-valued inverse thransform as a serializable list

def fourierize(source_path, cutoff=None): # if no cutoff is given, will simply yield the full spectra
    '''Replaces spectra in a set with their Fourier Tranforms (Hermitian and real-valued)'''
    if '(FT)' in str(source_path):
        raise TypeError('Input cannot already be Fourierized')   
    base_transform(source_path, operator=fourier, indicator=f'(FT{cutoff and cutoff or ""})', cutoff=cutoff) 

def inv_fourierize(source_path): # cutoff is list index of highest point to keep
    '''Replaces spectra in a set with their Fourier Tranforms (Hermitian and real-valued)'''
    if '(FT)' not in str(source_path):
        raise TypeError('Input must first be Fourierized')   
    base_transform(source_path, operator=inv_fourier, indicator='(IFT)') # transform is converted to list (np arrays are not JSON serializable)  
    
def fourier_filterize(source_path, cutoff):  # combines functionality of fourierize (with cutoff) and the inverse transform
    '''Reduces high-frequency noise in a dataset'''
    base_transform(source_path, operator=lambda spectrum : inv_fourier(fourier(spectrum, cutoff=cutoff)), indicator=f'(SFT{cutoff})')
    
    
# analysis and data characterization methods---------------------------------------------------------------------------------------------------------------------------
def inspect_spectra(source_path, species, ncols=6, save_path=None, marker='c-'):
    '''Plot the spectra of all instances of one species in the chosen dataset'''
    json_data = load_chem_json(source_path)
    if species not in json_data['species']:
        raise ValueError(f'Species "{species}" not in dataset')

    plots = [Single_Line_Plot(instance.spectrum, title=instance.name, marker=marker) 
                 for instance in json_data['chem_data']
                     if instance.species == species] 
    
    panel = Multiplot(ncols=ncols, span=len(plots))
    panel.draw_series(plots)
    
    if save_path:
        panel.save(save_path)

def inspect_variation(source_path, ncols=6, save_path=None):
    '''Generate a set of plots for all species in a dataset which shows how the baseline noise varies across spectra at each sample point'''
    json_data = load_chem_json(source_path)
    plots = [PWA_Plot([instance.spectrum
                           for instance in json_data['chem_data']
                               if instance.species == species], species) 
             for species in json_data['species']]  
    
    panel = Multiplot(ncols=ncols, span=len(plots))
    panel.draw_series(plots)
    
    if save_path:
        source_path = sanitized_path(source_path) # ensure Pathlike object pointing to json
        named_save_path = Path(save_path, f'{sanitized_path(source_path).stem} Species-wise PWAs')
        panel.save(named_save_path)     
    
def inspect_fsmoothing(source_path, inst, initial_cutoff=0, nsteps=1, step_size=1, ncols=6, save_figure=False):
    '''Investigate the original spectrum, Fourier Spectrum, truncated Fourier Spectrum, and reconstructed truncated spectrum of a
    single instance in the specified dataset. Optionally, can save the figure to the current directory, if it is of interest'''
    if 'FT' in str(source_path):
        raise TypeError('Method only applies to non-Fourier Transformed data') 
   
    json_data = load_chem_json(source_path)
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
    
    plots = [Single_Line_Plot(orig_data, title='Original Spectrum')]
    for cutoff in range(initial_cutoff, max_cutoff, step_size):
        cut_fft_data = np.fft.hfft(orig_data)
        cut_fft_data[cutoff:] = 0 
        plots.append(Single_Line_Plot(np.fft.ihfft(cut_fft_data).real, title=f'Reconstructed Spectrum  (to freq. {cutoff})'))

    if nsteps == 1:
        plots.insert(1, Single_Line_Plot(fft_data, title='FFT Spectrum'))
        plots.insert(2, Single_Line_Plot(cut_fft_data, title=f'FFT Spectrum (to freq. {cutoff})'))
    else:
        plots.append(Single_Line_Plot(np.fft.ihfft(fft_data).real, title='Fully Reconstructed Spectrum'))

    panel = Multiplot(ncols=ncols, span=len(plots))
    panel.draw_series(plots)

    if save_figure:      
        save_path = f'{cutoff and "Singular" or "Ranged"} Fourier Smoothing of {inst}'
        panel.save(save_path)

def inspect_fourier_maxima(source_path, save_path=None):
    '''Plot all Fourier maxima (e.g. the baseline magnitudes) by family, in the order the families appear in the data'''
    if '(FT)' not in str(source_path):
        raise ValueError('Method only applies to Fourier-Transformed datasets')
    
    json_data = load_chem_json(source_path)
    
    plots = [Single_Line_Plot([max(instance.spectrum)
                            for instance in json_data['chem_data']
                                if instance.family == family], title=family)
             for family in json_data['families']]  
    
    panel = Multiplot(nrows=1, span=len(json_data['families']))
    panel.draw_series(plots)
    
    if save_path:
        source_path = sanitized_path(source_path) # ensure Pathlike object pointing to json
        save_path = save_path/f'Fourier Maxima by Family - {source_path.stem}'
        panel.save(save_path)
