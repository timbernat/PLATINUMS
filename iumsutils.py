import csv, re, os
import numpy as np

def average(iterable):
    '''Calculcate and return average of an iterable'''
    return sum(iterable)/len(iterable)

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
        # rationale for regex: ignore capitalization (particular to ethers), only check end of name (particular to pinac<ol>one)
        if re.search('(?i){}\Z'.format(suffix), isolate_species(species)):  
            return family
        
def fourierize(source_file_name):  
    '''Creates a copy of a PLATIN-UMS-compatible data file with all spectra data replaced by their Discrete Fourier Transforms'''
    dest_file_name = re.sub('.csv', '- Fourier Transformed.csv', source_file_name) # explicit naming for the new file
    with open(source_file_name, 'r') as source_file, open(dest_file_name, 'w', newline='') as dest_file:
        for row in csv.reader(source_file):
            name, data = row[0], [float(i) for i in row[1:]]    # isolate the name and data
            data_fft = np.abs(np.fft.fft(data))                 # perform a fast Fourier transform over the data
            csv.writer(dest_file).writerow( [name, *data_fft] ) # write the resulting row to the named target file


            
            
            
