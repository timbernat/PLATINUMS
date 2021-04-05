import csv, json, re, collections
from pathlib import Path
 
# utilities specifically written to avoid having to import entire modules for a single object's functionality
def average(iterable, precision=4): 
    '''Calculate and return average of an iterable'''
    isum, n = 0, 0
    for i in iterable: # iterates, rather than using sum/len, so that generators work as inputs
        isum += i
        n += 1
    avg = isum/n
    return (precision and round(avg, precision) or avg)

def format_time(sec):
    '''Converts a duration in seconds into an h:mm:ss string; written explicitly to avoid importing datetime.timedelta'''
    minutes, seconds = divmod(round(sec), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'

def ceildiv(a, b):
    '''Ceiling analogue of floor division operator, meant to avoid importing math'''
    return -(a // -b)
        
    
# some general-purpose utilities
def ordered_and_counted(iterable):
    '''Takes an iterable of items and returns a sorted set of the items, and a dict of the counts of each item
    Specifically useful for getting the listing and counts of both species and families when jsonizing or transforming'''
    data = [i for i in iterable] # temporarily store data, in the case that the iterable is a generator
    return sorted(set(data)), collections.Counter(data)

def normalized(iterable):
    '''Normalize an iterable using min-max feature scaling (casts all values between 0 and 1)'''
    try:
        return tuple( (i - min(iterable))/(max(iterable) - min(iterable)) for i in iterable)
    except ZeroDivisionError: # if all data have the same value, max=min and min/max normalization will fail
        return tuple(i for i in iterable) # in that case, just return the original data
    
def dictmerge(dictlist):
    '''Takes a list of dictionaries with identical keys and combines them into a single dictionary with the values combined into lists under each entry'''
    return {key : [subdict[key] for subdict in dictlist] for key in dictlist[0]}

def multikey(some_dict, keyset):
    '''Takes a dictionary and an iterable of keys and returns Bool of whether or not the dict contains ALL of the listed keys'''
    return all(key in some_dict for key in keyset)

def partition(iterable, condition):
    '''Separates an iterable into two lists based on a truthy condition which can be applied to each item.
    Returns two lists, the first containing those which meet the condition and the second containing the rest'''
    members, non_members = [], []
    for item in iterable:
        (members if condition(item) else non_members).append(item)
    return members, non_members

def one_hot_mapping(iterable):
    '''Takes and iterable and returns a dictionary of the values in the iterable, assigned sequentially to one-hot vectors
    each of which is the length of the iterable (akin to an identity matrix)'''
    items = [i for i in iterable] # temporarily store data, in the case that the iterable is a generator
    return {value : tuple(int(val == value) for val in items) for value in items}

def get_RIP(mode1_spectrum):
    '''Naive but surprisingly effective method for identifying the RIP value for Mode 1 spectra'''
    return max(mode1_spectrum[:len(mode1_spectrum)//2]) # takes the RIP to be the maximum value in the first half of the spectrum

            
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
                       'non(?!e)' : 9, # prevents all ketones from being assigned "9"
                       'dec'  : 10}
    for affix, number in iupac_numbering.items():
        if re.search(f'(?i){affix}', species): # ignore capitalization (finds affix anywhere in word)
            return number + 0.5*bool(re.search(f'(?i)(iso|sec-){affix}', species)) # places "iso" and "sec-" compounds slightly lower on the list (+0.5, between compounds)
    else:
        return 100 # arbitrary, needs to return a number much greater than the rest to be placed at end

Instance = collections.namedtuple('Instance', ['name', 'species', 'family', 'spectrum', 'vector']) # provide class-like encoding of instances

        
#file and path utilities
def sanitized_path(path, ext='.json'):
    '''Ensures that a specified path is a Pathlike object and has the proper file extension'''
    if type(path) == str:
        path = Path(path) # ensure path is a Path object, allows for string input
    
    if path.suffix != ext:
        raise TypeError(f'Input must be a(n) {ext} file')
    else:
        return path

def load_chem_json(source_path):
    '''Read a chemical data json, de-serializes the Instance objects from "chem_data", and return the contents of the file'''
    source_path = sanitized_path(source_path)
    with source_path.open(mode='r') as source_file:
        json_data = json.load(source_file) # this comment is a watermark - 2020, timotej bernat
        json_data['chem_data'] = [Instance(*properties) for properties in json_data['chem_data']] # unpack the properties into Instance objects
    return json_data

def jsonize(source_path, correct_names=False): 
    '''Process spectral data csvs, generating labels, vector mappings, species counts, and other information,
    then cast the data to a json for ease of data reading in other applications and methods'''
    source_path = sanitized_path(source_path, ext='.csv')
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
    with source_path.open() as csv_file:
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
    
    dest_path = source_path.parent/f'{source_path.stem}{correct_names and "(@)" or ""}.json' # add indicator to target name if correcting names
    dest_path.touch() # create the new file
    with dest_path.open(mode='w') as json_file:
        json.dump(packaged_data, json_file) # dump our data into a json file with the same name as the original datacsv
        
def csvize(source_path):
    '''Inverse of jsonize, takes a processed chemical data json file and reduces it to a csv with just the listed spectra'''
    source_path = sanitized_path(source_path)
    json_data = load_chem_json(source_path)
    dest_path = source_path.parent/f'{source_path.stem}(C).csv' # add "C" indicator to denote that this file has been csvized
    dest_path.touch()
    with dest_path.open(mode='w', newline='') as dest_file:
        for instance in json_data['chem_data']:
            csv.writer(dest_file).writerow([instance.name, *instance.spectrum]) # merge name and spectral data into a single row and write it to the csv

def get_by_filetype(extension, path=Path.cwd()):  
    '''Get all files of a particular file type present in a given directory, (the current directory by default)'''
    if type(path) == str:
        path = Path(path) # convert any string input (i.e. just the name) into Path object
    
    filetypes_present = tuple(file.stem for file in path.iterdir() if file.suffix == extension)
    if filetypes_present == ():
        filetypes_present = (None,)
    return filetypes_present

def add_csv_column(csv_path, new_col_data):
    '''Takes a csv path and an iterable of data and appends the data to the csv as the rightmost column.
    If no such csv exists, will create a new csv with a single column consisting of the data passed'''
    if type(csv_path) == str:
        csv_path = Path(csv_path) # ensure path is a Path object
              
    temp_path = Path(csv_path.parent, f'tmp{csv_path.name}') # a temporary file to write into
    temp_path.touch() # create file at the temporary location
    
    with temp_path.open(mode='w', newline='') as outfile:
        if not csv_path.exists(): # if the original file doesn't exist, simply write the contents to the temporary file (will be renamed later)
            for entry in new_col_data:
                outfile.write(f'{entry}\n')
        else:
            with csv_path.open(mode='r') as infile:
                reader, writer = csv.reader(infile), csv.writer(outfile) # create csv parsing objects
                for row, entry in zip(reader, new_col_data): # note that zip will truncate any data that doesn't fit into the file
                    row.append(entry)    # append relevant entry to each row
                    writer.writerow(row) # write extended row to temporary file  
            csv_path.unlink()          # delete the original file
        
    temp_path.rename(csv_path) # rename the temporary file to that of the original, replacing it

def clear_folder(path):
    '''Recursively clear out the contents of a folder. A more tactful approach than deleting the folder and remaking it'''
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
