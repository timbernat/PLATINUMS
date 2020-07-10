import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar

import csv, gc, math, os, re                 # general imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg
#%matplotlib inline

from time import time, sleep                       # single-function imports
from datetime import timedelta
from shutil import rmtree
from random import shuffle
from collections import Counter

import tensorflow as tf
from tensorflow.keras import metrics                   # neural net libraries
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split


# SECTION 2 : some custom widget classes to make building and managing the GUI easier on myself ---------------------------------------------------
class Dyn_OptionMenu():
    '''My addon to the TKinter OptionMenu, adds methods to conveniently update menu contents'''
    def __init__(self, frame, var, options, width=10, row=0, col=0, colspan=1):
        self.options = options
        self.menu = tk.OptionMenu(frame, var, *self.options)
        self.menu.configure(width=width)
        self.menu.grid(row=row, column=col, columnspan=colspan)
        
        self.var = var
        self.contents = self.menu.children['menu']
    
    def update_menu(self):
        self.contents.delete(0, 'end')
        for option in self.options:
            self.contents.add_command(label=option, command=lambda x=option: self.var.set(x))

            
class ToggleFrame(tk.LabelFrame):
    '''A frame whose contents can be easily disabled or enabled, If starting disabled, must put "self.disable()"
    AFTER all widgets have been added to the frame'''
    def __init__(self, window, text, default_state='normal', padx=5, pady=5, row=0, col=0):
        tk.LabelFrame.__init__(self, window, text=text, padx=padx, pady=pady, bd=2, relief='groove')
        self.grid(row=row, column=col)
        self.state = default_state
        self.apply_state()
    
    def apply_state(self):
        for widget in self.winfo_children():
            widget.configure(state = self.state)
            
    def enable(self):
        self.state = 'normal'
        self.apply_state()
     
    def disable(self):
        self.state ='disabled'
        self.apply_state()
    
    def toggle(self):
        if self.state == 'normal':
            self.disable()
        else:
            self.enable()


class ConfirmButton(): 
    '''A confirmation button, will execute whatever function is passed to it when pressed. 
    Be sure to exclude parenthesis when passing the bound functions'''
    def __init__(self, frame, funct, padx=5, row=0, col=0, cs=1, sticky=None):
        self.button =tk.Button(frame, text='Confirm Selection', command=funct, padx=padx)
        self.button.grid(row=row, column=col, columnspan=cs, sticky=sticky)
      
    
class LabelledEntry():
    '''An entry with an adjacent label to the right. Use "self.get_value()" method to retrieve state of
    variable. Be sure to leave two columns worth of space for this widget'''
    def __init__(self, frame, text, var, state='normal', default=None, width=10, row=0, col=0):
        self.default = default
        self.var = var
        self.reset_default()
        self.label = tk.Label(frame, text=text, state=state)
        self.label.grid(row=row, column=col)
        self.entry = tk.Entry(frame, width=width, textvariable=self.var, state=state)
        self.entry.grid(row=row, column=col+1)
        
    def get_value(self):
        return self.var.get()
    
    def set_value(self, value):
        self.var.set(value)
    
    def reset_default(self):
        self.var.set(self.default)
    
    def configure(self, **kwargs):   # allows for disabling in ToggleFrames
        self.label.configure(**kwargs)
        self.entry.configure(**kwargs)
        
    
class Switch(): 
    '''A switch button, clicking inverts the boolean state and button display. State can be accessed via
    the <self>.state() method or with the <self>.var.get() attribute to use dynamically with tkinter'''
    def __init__(self, frame, text, value=False, dep_state='normal', dependents=None, width=10, row=0, col=0):
        self.label = tk.Label(frame, text=text)
        self.label.grid(row=row, column=col)
        self.switch =tk.Button(frame, width=width, command=self.toggle)
        self.switch.grid(row=row, column=col+1)
    
        self.dependents = dependents
        self.dep_state = dep_state
        self.value = value
        self.apply_state()
    
    def get_text(self):
        return self.value and 'Enabled' or 'Disabled'
        
    def get_color(self):
        return self.value and 'green2' or 'red' 
    
    def apply_state(self):
        self.dep_state = (self.value and 'normal' or 'disabled')
        self.switch.configure(text=self.get_text(), bg=self.get_color())
        if self.dependents:
            for widget in self.dependents:
                widget.configure(state=self.dep_state)
                
    def enable(self):
        self.value = True
        self.apply_state()
     
    def disable(self):
        self.value = False
        self.apply_state()
    
    def toggle(self):
        if self.value:
            self.disable()
        else:
            self.enable()  
           
        
class GroupableCheck():
    '''A checkbutton which will add or remove its value to an output list
    (passed as an argument when creating an instance) based on its check status'''
    def __init__(self, frame, value, output, state='normal', row=0, col=0):
        self.var = tk.StringVar()
        self.value = value
        self.output = output
        self.state = state
        self.cb = tk.Checkbutton(frame, text=value, variable=self.var, onvalue=self.value, offvalue=None,
                              state=self.state, command=self.edit_output)
        self.cb.grid(row=row, column=col, sticky='w')
        self.cb.deselect()
        
    def edit_output(self):
        if self.var.get() == self.value:
            self.output.append(self.value)
        else:
            self.output.remove(self.value)
            
    def configure(self, **kwargs):
        self.cb.configure(**kwargs)
            
class CheckPanel():
    '''A panel of GroupableChecks, allows for simple selectivity of the contents of some list'''
    def __init__(self, frame, data, output, state='normal', ncols=4, row_start=0, col_start=0):
        self.output = output
        self.state = state
        self.row_span = math.ceil(len(data)/ncols)
        self.panel = [ GroupableCheck(frame, val, output, state=self.state, row=row_start + i//ncols,
                                      col=col_start + i%ncols) for i, val in enumerate(data) ]
        
    def wipe_output(self):
        self.output.clear()
        
    def apply_state(self):
        for gc in self.panel:
            gc.configure(state=self.state)
    
    def enable(self):
        self.state = 'normal'
        self.apply_state()
     
    def disable(self):
        self.state = 'disabled'
        self.apply_state()
    
    def toggle(self):
        if self.state == 'normal':
            self.disable()
        else:
            self.enable()        
        
class SelectionWindow():
    '''The window used to select unfamiliars'''
    def __init__(self, main, parent_frame, size, selections, output, ncols=1):
        self.window = tk.Toplevel(main)
        self.window.title('Select Members to Include')
        self.window.geometry(size)
        self.parent = parent_frame
        self.parent.disable()
        
        self.panel = CheckPanel(self.window, selections, output, ncols=ncols)
        self.confirm = ConfirmButton(self.window, self.confirm, row=self.panel.row_span + 1, col=ncols-1)

    def confirm(self):
        self.parent.enable()
        self.window.destroy()
        
class TkEpochs(Callback):   
    '''A custom keras Callback to display the current epoch in the training progress window'''
    def __init__(self, train_window):
        super(TkEpochs, self).__init__()
        self.tw = train_window
    
    def on_epoch_begin(self, epoch, logs=None):
        self.tw.set_epoch_progress(epoch + 1)
        
    def on_epoch_end(self, epoch, logs=None):  
        if self.tw.end_training:
            self.model.stop_training = True
        
class TrainingWindow():
    '''The window which displays training progress, was easier to subclass outside of the main GUI class'''
    def __init__(self, main, total_rounds, num_epochs, train_funct, reset_funct):
        self.total_rounds = total_rounds
        self.num_epochs = num_epochs
        self.main = main
        self.training_window = tk.Toplevel(main)
        self.training_window.title('Training Progress')
        self.training_window.geometry('390x142')
        self.end_training = False
        
        # Status Printouts
        self.status_frame = tk.Frame(self.training_window, bd=2, padx=11, relief='groove')
        self.status_frame.grid(row=0, column=0)
        
        self.member_label = tk.Label(self.status_frame, text='Current Unfamiliar: ')
        self.curr_member = tk.Label(self.status_frame)
        self.slice_label = tk.Label(self.status_frame, text='Current Data Slice: ')
        self.curr_slice = tk.Label(self.status_frame)
        self.round_label = tk.Label(self.status_frame)
        self.round_progress = Progressbar(self.status_frame, orient='horizontal', length=250, maximum=total_rounds)
        self.epoch_label = tk.Label(self.status_frame)
        self.epoch_progress = Progressbar(self.status_frame, orient='horizontal', length=250, maximum=num_epochs) 
        self.status_label = tk.Label(self.status_frame, text='Current Status: ')
        self.curr_status = tk.Label(self.status_frame)
        
        self.member_label.grid(  row=0, column=0)
        self.curr_member.grid(   row=0, column=1, sticky='w')
        self.slice_label.grid(   row=1, column=0)
        self.curr_slice.grid(    row=1, column=1, sticky='w')
        self.round_label.grid(   row=2, column=0)
        self.round_progress.grid(row=2, column=1, sticky='w')
        self.epoch_label.grid(   row=3, column=0)
        self.epoch_progress.grid(row=3, column=1, sticky='w')
        self.status_label.grid(  row=4, column=0)
        self.curr_status.grid(   row=4, column=1, sticky='w')
        
        self.reset()
    
        #Training Buttons
        self.button_frame = ToggleFrame(self.training_window, '', padx=0, pady=0, row=1)
        self.retrain_button = tk.Button(self.button_frame, text='Retrain', width=17, bg='dodger blue', command=train_funct)
        self.reinput_button = tk.Button(self.button_frame, text='Reset', width=17, bg='orange', command=reset_funct)
        self.abort_button = tk.Button(self.button_frame, text='Abort Training', width=17, bg='red', command=self.abort)
        
        self.retrain_button.grid(row=0, column=0)
        self.reinput_button.grid(row=0, column=1)
        self.abort_button.grid(  row=0, column=2) 
        
        self.button_frame.disable()
        self.abort_button.configure(state='normal')
        
    def abort(self):
        self.end_training = True
        self.reset()
        
    def reset(self):
        self.set_member('---')
        self.set_slice('---')
        self.set_round_progress(0)
        self.set_epoch_progress(0)
        self.set_status('Standby')
    
    def set_member(self, member):
        self.curr_member.configure(text=member)
        self.main.update()
    
    def set_slice(self, data_slice):
        self.curr_slice.configure(text=data_slice)
        self.main.update()
    
    def set_status(self, status):
        self.curr_status.configure(text=status)
        self.main.update()
    
    def set_round_progress(self, curr_round):
        self.round_label.configure(text='Training Round: {}/{}'.format(curr_round, self.total_rounds) )
        self.round_progress.configure(value=curr_round)
        self.main.update()
        
    def set_epoch_progress(self, curr_epoch):
        self.epoch_label.configure(text='Training Epoch: {}/{}'.format(curr_epoch, self.num_epochs) )
        self.epoch_progress.configure(value=curr_epoch)
        self.main.update()
        
    def destroy(self):
        self.training_window.destroy()
        
# Start of actual GUI app class code ------------------------------------------------------------------------------------------------------------
class PLATINUMS_App():
    def __init__(self, main):
        #Main Window
        self.main = main
        self.main.title('PLATIN-UMS 4.1.9-alpha')
        self.main.geometry('445x395')

        #Frame 1
        self.data_frame = ToggleFrame(self.main, 'Select CSV to Read: ', padx=21, pady=5, row=0)
        self.chosen_file = tk.StringVar()
        self.chosen_file.set('--Choose a CSV--')
        self.data_from_file = {}
        self.all_species = set()
        self.families = set()
        self.family_mapping = {}
        self.spectrum_size = None
        
        self.csv_menu = Dyn_OptionMenu(self.data_frame, self.chosen_file, (None,) , width=28, colspan=2)
        self.read_label = tk.Label(self.data_frame, text='Read Status:')
        self.read_status = tk.Label(self.data_frame, bg='light gray', padx=30, text='No File Read')
        self.refresh_button = tk.Button(self.data_frame, text='Refresh CSVs', command=self.update_csvs, padx=15)
        self.confirm_data = ConfirmButton(self.data_frame, self.import_data, padx=2, row=1, col=2)
        
        self.read_label.grid(row=1, column=0)
        self.read_status.grid(row=1, column=1)
        self.refresh_button.grid(row=0, column=2)
        
        self.update_csvs() # populate csv menu for the first time
        
        #Frame 2
        self.input_frame = ToggleFrame(self.main, 'Select Input Mode: ', padx=5, pady=5, row=1)
        self.read_mode = tk.StringVar()
        self.read_mode.set(None)
        self.selections = []
        
        self.mode_buttons = [tk.Radiobutton(self.input_frame, text=mode, value=mode, var=self.read_mode, command=self.further_sel) 
                             for mode in ('Select All', 'By Family', 'By Species') ]
        for i in range(3):
            self.mode_buttons[i].grid(row=0, column=i)
        self.confirm_sels = ConfirmButton(self.input_frame, self.confirm_inputs, row=0, col=3, sticky='e')
        self.input_frame.disable()
        
        #Frame 3
        self.num_epochs = None
        self.batchsize = None
        self.learnrate = None
        
        self.hyper_frame = ToggleFrame(self.main, 'Set Hyperparameters: ', padx=34, pady=5, row=2)
        self.epoch_entry = LabelledEntry(self.hyper_frame, 'Epochs:', tk.IntVar(), default=8)
        self.batchsize_entry = LabelledEntry(self.hyper_frame, 'Batchsize:', tk.IntVar(), default=32, row=1)
        self.learnrate_entry = LabelledEntry(self.hyper_frame, 'Learnrate:', tk.DoubleVar(), default=2e-5, col=3)
        self.confirm_hyperparams = ConfirmButton(self.hyper_frame, self.confirm_hp, row=1, col=4, sticky='e')
        self.hyper_frame.disable()
        
        #Frame 4
        self.trimming_min = None
        self.trimming_max = None
        self.slice_decrement = None
        self.num_slices = None
        
        self.param_frame = ToggleFrame(self.main, 'Set Training Parameters: ', padx=15, pady=5, row=3)
        self.stop_switch = Switch(self.param_frame, 'Early Stopping: ', row=0, col=2)
        self.trim_switch = Switch(self.param_frame, 'RIP Trimming: ', row=1, col=2)
        self.n_slice_entry = LabelledEntry(self.param_frame, 'Number of Slices:', tk.IntVar(), default=1, row=2, col=1)
        self.upper_bound_entry = LabelledEntry(self.param_frame, 'Upper Bound:', tk.IntVar(), default=400, row=2, col=3)
        self.slice_decrement_entry = LabelledEntry(self.param_frame, 'Slice Decrement:', tk.IntVar(), default=50, row=3, col=1)
        self.lower_bound_entry = LabelledEntry(self.param_frame, 'Lower Bound:', tk.IntVar(), default=50, row=3, col=3)
        self.confirm_training_params = ConfirmButton(self.param_frame, self.confirm_tparams, row=4, col=3, cs=2, sticky='e')

        self.trim_switch.dependents = (self.n_slice_entry, self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry)
        self.keras_callbacks = []
        
        self.param_frame.disable()
    
        #Training Buttons and values
        self.train_button = tk.Button(self.main, text='TRAIN', padx=20, width=45, bg='dodger blue', state='disabled', command=self.begin_training)
        self.train_button.grid(row=4, column=0)
        self.train_window = None
        
        self.total_rounds = None
        self.summaries = {}

        #General/Misc
        self.frames = (self.data_frame, self.input_frame, self.hyper_frame, self.param_frame)
        self.entries = (self.epoch_entry, self.batchsize_entry, self.learnrate_entry, self.n_slice_entry,
                        self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry)
        
        self.exit_button = tk.Button(self.main, text='Exit', padx=22, pady=22, bg='red', command=self.shutdown)
        self.reset_button = tk.Button(self.main, text='Reset', padx=20, bg='orange', command=self.reset)
        self.exit_button.grid(row=0, column=4)
        self.reset_button.grid(row=4, column=4)
        
    
    #General Methods
    def isolate(self, on_frame):
        '''Enable just one frame.'''
        for frame in self.frames:
            if frame == on_frame:
                frame.enable()
            else:
                frame.disable()   
    
    def shutdown(self):
        '''Close the application, with confirm prompt'''
        if messagebox.askokcancel('Exit', 'Are you sure you want to close?'):
            self.main.destroy()
    
    def reset(self):
        '''reset the GUI to the opening state, along with all variables'''
        self.isolate(self.data_frame)
        self.trim_switch.disable()
        self.stop_switch.disable()
        
        self.read_status.configure(bg='light grey', text='No File Read')
        self.train_button.configure(state='disabled')
        self.chosen_file.set('--Choose a CSV--')
        self.read_mode.set(None)
        
        for datum in (self.data_from_file, self.selections):
            datum.clear()
        self.all_species = set()
        self.families = set()
        self.family_mappings = {}
        
        for entry in self.entries:
            entry.reset_default()
        
        self.reset_training()

                
    #Frame 1 (Reading) Methods 
    def get_species(self, species):
        '''Strips extra numbers off the end of the name of a' species in a csv and just tells you the species name'''
        return re.sub('(\s|-)\d+\s*\Z', '', species)  # regex to crop off terminal digits in a variety of possible 
    
    def get_family(self, species):
        '''Takes the name of a species and returns the chemical family that that species belongs to, based on IUPAC naming conventions'''
        iupac_suffices = {   'ane':'alkane',
                            'ene':'alkene',
                            'yne':'alkyne',
                            'oic acid': 'carboxylic acid',
                            #'oate':'ester',
                            'ol':'alcohol',
                            'ate':'acetate',
                            'ether':'ether',
                            'al':'aldehyde',
                            'one':'ketone'  }                    
        for regex, family in iupac_suffices.items():
            if re.search('(?i){}'.format(regex), species):  # ignore case/capiatlization (particular to case of ethers)
                return family
    
    def read_chem_data(self): 
        '''Used to read and format the data from the csv provided into a form usable by the training program
        Returns the read data (with vector) and sorted lists of the species and families found in the data'''
        csv_name = './{}.csv'.format( self.chosen_file.get() )
        with open(csv_name, 'r') as file:
            for row in csv.reader(file):
                label = row[0]
                spectrum_data = [float(i) for i in row[1:]]  # convert data point from str to floats
                
                self.data_from_file[label] = spectrum_data
                self.all_species.add( self.get_species(label) )
                self.families.add( self.get_family(label) )
                if not self.spectrum_size:
                    self.spectrum_size = len(spectrum_data)

        self.upper_bound_entry.set_value(self.spectrum_size)
        self.all_species, self.families = sorted(self.all_species), sorted(self.families)  # sort and convert to lists
        
        num_families = len(self.families)    # generate family mapping based what families present in the current dataset
        for index, family in enumerate(self.families):
            one_hot_vector = tuple(i == index and 1 or 0 for i in range(len(self.families)) )
            self.family_mapping[family] = one_hot_vector
                                   
        for species, data in self.data_from_file.items():  # add mapping vector to all data entries
            vector = self.family_mapping[self.get_family(species)]
            self.data_from_file[species] = (data, vector)
    
    def update_csvs(self):
        '''Update the CSV dropdown selection to catch any changes in the files present'''
        csvs_present = tuple(i[:-4] for i in os.listdir('.') if re.search('.csv\Z', i))
        if csvs_present == ():
            csvs_present = (None,)
        
        self.csv_menu.options = csvs_present
        self.csv_menu.update_menu()
    
    def import_data(self):
        '''Read in data based on the selected data file'''
        if self.chosen_file.get() == '--Choose a CSV--':
            messagebox.showerror('File Error', 'No CSV selected')
        else:
            self.read_chem_data()
            self.read_status.configure(bg='green2', text='CSV Read!')
            self.isolate(self.input_frame)

    
    #Frame 2 (Input) Methods
    def further_sel(self): 
        '''logic for selection of members to include in training'''
        self.selections.clear()
        if self.read_mode.get() == 'Select All':
            self.selections = self.all_species
        elif self.read_mode.get() == 'By Species':
            SelectionWindow(self.main, self.input_frame, '1000x210', self.all_species, self.selections, ncols=8)
        elif self.read_mode.get() == 'By Family':
            SelectionWindow(self.main, self.input_frame, '250x90', self.families, self.selections, ncols=3)

    def confirm_inputs(self):
        '''Confirm species input selections'''
        if self.selections == []:
            messagebox.showerror('No selections made', 'Please select unfamiliars')
        else:
            if self.read_mode.get() == 'By Family':
                self.selections = [species for family in self.selections for species in self.all_species if self.get_family(species) == family]
            self.isolate(self.hyper_frame)

    
    # Frame 3 (hyperparameter) Methods
    def confirm_hp(self):
        '''Confirm hyperparameter selections'''
        self.num_epochs = self.epoch_entry.get_value()
        self.batchsize = self.batchsize_entry.get_value()
        self.learnrate = self.learnrate_entry.get_value()
        self.isolate(self.param_frame)
        self.trim_switch.disable()
    
    
    #Frame 4 (training parameter) Methods
    def confirm_tparams(self):
        '''Confirm training parameter selections, perform pre-training error checks if necessary'''
        if self.stop_switch.value:
            self.keras_callbacks.append( EarlyStopping(monitor='loss', mode='min', verbose=1, patience=8) )
        
        if not self.trim_switch.value: # if RIP trimming is not selected
            self.param_frame.disable()
            self.train_button.configure(state='normal')
        else:
            self.num_slices = self.n_slice_entry.get_value()
            self.slice_decrement = self.slice_decrement_entry.get_value()
            self.trimming_max = self.upper_bound_entry.get_value()
            self.trimming_min = self.lower_bound_entry.get_value()
                
            if self.trimming_min < 0 or type(self.trimming_min) != int:
                messagebox.showerror('Boundary Value Error', 'Trimming Min must be a positive integer')
            elif self.trimming_max < 0 or type(self.trimming_max) != int:
                messagebox.showerror('Boundary Value Error', 'Trimming Max must be a positive integer')
            elif self.trimming_max > self.spectrum_size:                            
                messagebox.showerror('Limit too high', 'Upper bound greater than data size, please decrease "Upper Bound"')
            elif self.trimming_max - self.num_slices*self.slice_decrement <= self.trimming_min:  
                messagebox.showerror('Boundary Mismatch', 'Upper limit will not always exceed lower;\nDecrease either slice decrement or lower bound')
            else:
                self.param_frame.disable()
                self.train_button.configure(state='normal')
        
    #Training and Neural Net-Specific Methods    
    def begin_training(self):
        self.total_rounds = len(self.selections)
        if self.trim_switch.value:  #if RIP trimming is enabled
            self.total_rounds += len(self.selections)*self.num_slices

        self.reset_training()
        self.train_window = TrainingWindow(self.main, self.total_rounds, self.num_epochs, self.begin_training, self.reset)
        self.keras_callbacks.append( TkEpochs(self.train_window) )

        self.training()
                
    def training(self, verbosity=False):
        '''The neural net training function itself'''
        start_time = time()    # log start of runtime
        num_spectra = 3
        
        for filename in os.listdir('.'):       # deletes results folders from prior trainings to prevent overwriting
            if re.match('\A(Training Results)', filename):
                self.train_window.set_status('Deleting {}'.format(filename) )
                rmtree('./%s'% filename, ignore_errors=True)
        
        current_round = 0       
        RIP_trimming = self.trim_switch.value
        for instance, member in enumerate(self.selections):
            for select_RIP in range(1 + int(RIP_trimming)):     # treats 0, 1 iteration as bool (optional true)
                for segment in range(select_RIP and self.num_slices or 1):   
                    self.train_window.set_status('Training...')
                    
                    current_round += 1
                    self.train_window.set_round_progress(current_round)
                    
                    if select_RIP:
                        lower_bound, upper_bound = self.trimming_min, self.trimming_max - self.slice_decrement*segment
                        point_range = 'points {}-{}'.format(lower_bound, upper_bound)
                    else:
                        lower_bound, upper_bound =  0, self.spectrum_size
                        point_range = 'Full Spectra'

                # DATA SELECTION AND ANALYSIS  
                    unfam_data, unfam_titles, spectra_titles = [], [], []
                    features, labels, families = [], [], []
                    train_set_size, unfam_set_size = 0, 0
                                               
                    for species, (data, vector) in self.data_from_file.items():
                        data = data[lower_bound:upper_bound]                      
                        if re.findall('\A{}.*'.format(member), species):       # if the current species begins with <member>
                            unfam_set_size += 1
                            unfam_data.append(data)
                            unfam_titles.append( (species, 'p') )
                            if unfam_set_size <= num_spectra:
                                spectra_titles.append( (species, 's') )
                        else:                                            # add all other species to the training set (unfamiliar training only)
                            train_set_size += 1
                            features.append(data)
                            labels.append(vector)
                            families.append(self.get_family(species))
                                               
                    unfam_titles.append( ('Standardized Summation', 'p') )
                    occurrences = Counter(families)
                    self.train_window.set_member( '{} ({} instances found)'.format(member, unfam_set_size) )
                    self.train_window.set_slice(point_range)
                                               
                    x_train, x_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.2)  
                    train_feat_and_lab = (x_train.shape[0], y_train.shape[0])
                    test_feat_and_lab = (x_test.shape[0], y_test.shape[0])

                # MODEL CREATION AND TRAINING
                    with tf.device('CPU:0'):                            
                        model = Sequential()                              # model block is created, layers are created/added in this block
                        model.add(Dense(512, input_dim=(upper_bound - lower_bound), activation='relu'))  
                        model.add(Dropout(0.5))                           # dropout layer, to reduce overfit
                        model.add(Dense(512, activation='relu'))          # 512 neuron hidden layer
                        #model.add(Dense(512, activation='relu'))         # 512 neuron hidden layer
                        model.add(Dense(len(self.families), activation='softmax')) # softmax gives prob. dist. of identity over all families
                        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learnrate), metrics=['accuracy']) 
                    
                    if verbosity:   # optional prinout, gives overview of the training data and the model settings
                        for (x, y, group) in ( (x_train, y_train, 'training'), (x_test, y_test, 'test') ):
                            print('{} features/labels in {} set ({} of the data)'.format(
                                  (x.shape[0], y.shape[0]), group, round(100 * x.shape[0]/train_set_size, 2)) )
                        print('\n{} features total. Of the {} species in training dataset:'.format(train_set_size + unfam_set_size, train_set_size) )
                        for family in self.family_mapping.keys():
                            print('    {}% of data are {}s'.format( round(100 * occurrences[family]/train_set_size, 2), family) ) 
                        model.summary()
                                      
                    hist = model.fit(x_train, y_train, epochs=self.num_epochs, batch_size=self.batchsize,   # model training occurs here
                                     callbacks=self.keras_callbacks, verbose=verbosity and 2 or 0)  
                    
                    if self.train_window.end_training:
                        messagebox.showerror('Training has Stopped', 'Training aborted by user;\nProceed from Progress Window')
                        self.train_window.button_frame.enable()
                        return None     # without this, aborting training only pauses one iteration of loop

                # EVALUATION OF PERFORMANCE
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)    # keras' self-evaluation, not the most reliable                    
                    if verbosity:
                        print('\nTest loss: {} \nTest accuracy: {} \n'.format(test_loss, test_acc))
                    hist_metrics = [hist.history['loss'], hist.history['accuracy']]
                    hist_titles = [('Training Loss (Final = %0.2f)' % test_loss, 'm'),
                                   ('Training Accuracy (Final = %0.2f%%)' % (100 * test_acc), 'm')]

                    # More intensive/accurate evaluation using our unfamiliar sample set
                    preds, targets, num_correct = [], [], 0          
                    for prediction in list( model.predict(np.array(unfam_data)) ):
                        target_index = self.family_mapping[self.get_family(member)].index(1)   # the index of the actual identity of current member
                        target = prediction[target_index]
                        
                        preds.append(prediction)                     
                        targets.append(target)                       
                        if max(prediction) == target:
                            num_correct += 1  
                    targets.sort(reverse=True)   # targets are sorted in reverse order for the Fermi plot
                    preds.append( [sum(i)/len(preds) for i in map(list, zip(*preds)) ])   # add summation of the predictions to the end of preds
                    
                    fermi_title = ('{} FDP, {}/{} correct'.format(member, num_correct, unfam_set_size),'f')                 
                    data = hist_metrics + unfam_data[:num_spectra] + [targets] + preds    # wrap the data to be plotted into a single data list
                    titles = hist_titles + spectra_titles + [fermi_title] + unfam_titles  # wrap the names of plots into a single title list

                    if point_range not in self.summaries:    # create a new entry if none exists for the current spectrum slice                                     
                        self.summaries[point_range] = ( [], [], [] )
                    fermi_data, names, scores = self.summaries[point_range]
                    fermi_data.append(targets)
                    names.append(fermi_title)   # add plot data, titles, and prediction scores to relevant sub-dict
                    scores.append( (member, num_correct/unfam_set_size) )

                    self.train_window.set_status('Writing Results to Folders...')    
                    dir_name = './Training Results, {}'.format(point_range)   # ensure that a results folder for the current slice range exists
                    if not os.path.exists(dir_name):                                                           
                        os.makedirs(dir_name)

                    self.adagraph(data, titles, 6, lower_bound, upper_bound)  # produce summary plot 
                    plt.savefig('{}/{}.png'.format(dir_name, member))          # save current plot of results (trimming is labelled)
                    plt.close() 
                    gc.collect()    # collect any junk remaining in RAM

        self.train_window.set_status('Distributing Result Summaries...')
        for point_range, (fermi_data, names, scores) in self.summaries.items():  # distribution of summary data to the appropriate respective folders
                scores.sort(key=lambda pair : pair[1], reverse=True)     
                with open('./Training Results, {}/Scores.txt'.format(point_range), 'a') as score_file:
                    for (species, score) in scores:
                        score_file.write('{} : {}\n'.format(species, score) )
                self.adagraph(fermi_data, names, 5, None, None)
                plt.savefig('./Training Results, {}/Fermi Summary'.format(point_range) )
                plt.close()
        
        self.train_window.button_frame.enable()
        self.train_window.set_status('Finished')
        runtime = timedelta(seconds=round(time() - start_time))   
        messagebox.showinfo('Training Complete', 'Routine completed in {}\nResults can be found in "Training Results" folders'.format(runtime) )
    
    def reset_training(self):
        if self.train_window: # if a window already exists
            self.train_window.destroy()
            self.train_window = None
        self.summaries.clear()
        self.keras_callbacks.clear()
    
    def adagraph(self, data_list, name_list, ncols, lower_bound, upper_bound):  
        '''a general tidy internal graphing utility of my own devising, used to produce all manner of plots during training'''
        nrows = math.ceil(len(data_list)/ncols)  #  determine the necessary number of rows needed to accomodate the data
        display_size = 20                        # 20 seems to be good size for jupyter viewing
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(display_size, display_size * nrows/ncols)) 
        for idx, data in enumerate(data_list):                         
            if nrows > 1:                        # locate the current plot, unpack linear index into coordinate
                row, col = divmod(idx, ncols)      
                curr_plot = axs[row][col]  
            else:                                # special case for indexing plots with only one row; my workaround of implementation in matplotlib
                curr_plot = axs[idx]              
            
            plot_title, plot_type = name_list[idx]
            curr_plot.set_title(plot_title)
            
            if plot_type == 's':                 # for plotting spectra
                curr_plot.plot(range(lower_bound, upper_bound), data, 'k,') 
                curr_plot.axis( [lower_bound , upper_bound+1, min(data), max(data)] )
            elif plot_type == 'p':               # for plotting predictions
                bar_color = (plot_title == 'Standardized Summation' and 'r' or 'b')
                curr_plot.bar( self.family_mapping.keys(), data, color=bar_color)  
                curr_plot.set_ylim(0,1)
                curr_plot.tick_params(axis='x', labelrotation=45)
            elif plot_type == 'm':               # for plotting metrics from training
                curr_plot.plot(range(1, self.num_epochs + 1), data, idx and 'g' or 'r') # rework the alternating color scheme
            elif plot_type == 'f':               # for the fermi-dirac plots
                curr_plot.plot( [i/len(data) for i in range(len(data))], data, linestyle='-', color='m')  # normalized by dividing by length
                curr_plot.axis( [0, 1, 0, 1] )
        plt.tight_layout()

if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
