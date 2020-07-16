import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar

import csv, gc, math, os, re                 # general imports
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.backends.backend_tkagg
#%matplotlib inline

from time import time, strftime                      # single-function imports
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
class Dyn_OptionMenu:
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


class StatusBox:
    '''A simple label which changes color and gives indictation of the status of something'''
    def __init__(self, frame, on_message='On', off_message='Off', status=False, width=17, padx=0, row=0, col=0):
        self.on_message = on_message
        self.off_message = off_message
        self.status_box = tk.Label(frame, width=width, padx=padx)
        self.status_box.grid(row=row, column=col)
        self.set_status(status)
        
    def set_status(self, status):
        if type(status) != bool:
            raise Exception(TypeError)
        
        if status:
            self.status_box.configure(bg='green2', text=self.on_message)
        else:
            self.status_box.configure(bg='light gray', text=self.off_message)
            
            
class ConfirmButton: 
    '''A confirmation button, will execute whatever function is passed to it when pressed. 
    Be sure to exclude parenthesis when passing the bound functions'''
    def __init__(self, frame, funct, padx=5, row=0, col=0, cs=1, sticky=None):
        self.button =tk.Button(frame, text='Confirm Selection', command=funct, padx=padx)
        self.button.grid(row=row, column=col, columnspan=cs, sticky=sticky)
      
    
class LabelledEntry:
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
        
    
class Switch: 
    '''A switch button, clicking inverts the boolean state and button display. State can be accessed via
    the <self>.state() method or with the <self>.var.get() attribute to use dynamically with tkinter'''
    def __init__(self, frame, text, value=False, dep_state='normal', dependents=None, width=10, row=0, col=0):
        self.label = tk.Label(frame, text=text)
        self.label.grid(row=row, column=col)
        self.switch = tk.Button(frame, width=width, command=self.toggle)
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
           
        
class GroupableCheck:
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
            
class CheckPanel:
    '''A panel of GroupableChecks, allows for simple selectivity of the contents of some list'''
    def __init__(self, frame, data, output, state='normal', ncols=4, row_start=0, col_start=0):
        self.output = output
        self.state = state
        self.row_span = math.ceil(len(data)/ncols)
        self.panel = [ GroupableCheck(frame, val, output, state=self.state, row=row_start + i//ncols, col=col_start + i%ncols) for i, val in enumerate(data) ]
        
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
        
class SelectionWindow:
    '''The window used to select species for evaluation'''
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
        
class TrainingWindow:
    '''The window which displays training progress, was easier to subclass outside of the main GUI class'''
    def __init__(self, main, total_rounds, num_epochs, train_funct, reset_funct, train_button):
        self.total_rounds = total_rounds
        self.num_epochs = num_epochs
        self.main = main
        self.train_button = train_button
        self.training_window = tk.Toplevel(main)
        self.training_window.title('Training Progress')
        self.training_window.geometry('390x142')
        self.end_training = False
        
        # Status Printouts
        self.status_frame = tk.Frame(self.training_window, bd=2, padx=11, relief='groove')
        self.status_frame.grid(row=0, column=0)
        
        self.member_label = tk.Label(self.status_frame, text='Current Species: ')
        self.curr_member = tk.Label(self.status_frame)
        self.slice_label = tk.Label(self.status_frame, text='Current Data Slice: ')
        self.curr_slice = tk.Label(self.status_frame)
        self.round_label = tk.Label(self.status_frame)
        self.round_progress = Progressbar(self.status_frame, orient='horizontal', length=240, maximum=total_rounds)
        self.epoch_label = tk.Label(self.status_frame)
        self.epoch_progress = Progressbar(self.status_frame, orient='horizontal', length=240, maximum=num_epochs) 
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
        self.train_button.configure(state='normal')
        self.training_window.destroy()
        
# Start of actual GUI app class code ------------------------------------------------------------------------------------------------------------
class PLATINUMS_App:
    def __init__(self, main):
        #Main Window
        self.main = main
        self.main.title('PLATIN-UMS 4.2.1-alpha')
        self.main.geometry('445x420')
        self.preds = None

        #Frame 1
        self.data_frame = ToggleFrame(self.main, 'Select CSV to Read: ', padx=21, pady=5, row=0)
        self.chosen_file = tk.StringVar()
        self.chosen_file.set('--Choose a CSV--')
        self.chem_data = {}
        self.all_species = set()
        self.families = set()
        self.family_mapping = {}
        self.spectrum_size = None
        
        self.csv_menu = Dyn_OptionMenu(self.data_frame, self.chosen_file, (None,) , width=28, colspan=2)
        self.read_label = tk.Label(self.data_frame, text='Read Status:')
        self.read_status = StatusBox(self.data_frame, on_message='CSV Read!', off_message='No File Read', row=1, col=1)
        self.refresh_button = tk.Button(self.data_frame, text='Refresh CSVs', command=self.update_csvs, padx=15)
        self.confirm_data = ConfirmButton(self.data_frame, self.import_data, padx=2, row=1, col=2)
        
        self.refresh_button.grid(row=0, column=2)
        self.read_label.grid(row=1, column=0)
        
        self.update_csvs() # populate csv menu for the first time
        
        #Frame 2
        self.input_frame = ToggleFrame(self.main, 'Select Input Mode: ', padx=5, pady=5, row=1)
        self.read_mode = tk.StringVar()
        self.read_mode.set(None)
        self.selections = []
        self.choices = None
        
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
        
        self.hyper_frame = ToggleFrame(self.main, 'Set Hyperparameters: ', padx=8, pady=5, row=2)
        self.epoch_entry = LabelledEntry(self.hyper_frame, 'Epochs:', tk.IntVar(), width=19, default=8)
        self.batchsize_entry = LabelledEntry(self.hyper_frame, 'Batchsize:', tk.IntVar(), width=19, default=32, row=1)
        self.learnrate_entry = LabelledEntry(self.hyper_frame, 'Learnrate:', tk.DoubleVar(), width=18, default=2e-5, col=3)
        self.confirm_hyperparams = ConfirmButton(self.hyper_frame, self.confirm_hp, row=1, col=3, cs=2, sticky='e')
        self.hyper_frame.disable()
        
        #Frame 4
        self.trimming_min = None
        self.trimming_max = None
        self.slice_decrement = None
        self.num_slices = None
        
        self.param_frame = ToggleFrame(self.main, 'Set Training Parameters: ', padx=9, pady=5, row=3)
        self.fam_switch = Switch(self.param_frame, 'Familiar Training :', row=0, col=1)
        self.stop_switch = Switch(self.param_frame, 'Early Stopping: ', row=1, col=1)
        self.trim_switch = Switch(self.param_frame, 'RIP Trimming: ', row=2, col=1)
        
        self.upper_bound_entry = LabelledEntry(self.param_frame, 'Upper Bound:', tk.IntVar(), default=400, row=3, col=0)
        self.slice_decrement_entry = LabelledEntry(self.param_frame, 'Slice Decrement:', tk.IntVar(), default=20, row=3, col=2)
        self.lower_bound_entry = LabelledEntry(self.param_frame, 'Lower Bound:', tk.IntVar(), default=50, row=4, col=0)
        self.n_slice_entry = LabelledEntry(self.param_frame, 'Number of Slices:', tk.IntVar(), default=1, row=4, col=2)
        self.confirm_training_params = ConfirmButton(self.param_frame, self.confirm_tparams, row=5, col=2, cs=2, sticky='e')

        self.trim_switch.dependents = (self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry, self.n_slice_entry)
        self.switches = (self.fam_switch, self.stop_switch, self.trim_switch)
        self.keras_callbacks = []
        
        self.param_frame.disable()
    
        #Training Buttons and values
        self.train_button = tk.Button(self.main, text='TRAIN', padx=20, width=45, bg='dodger blue', state='disabled', command=self.begin_training)
        self.train_button.grid(row=4, column=0)
        self.train_window = None
        
        self.total_rounds = None
        self.results_folder = None
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
        for switch in self.switches:
            switch.disable()
        
        self.read_status.set_status(False)
        self.train_button.configure(state='disabled')
        self.chosen_file.set('--Choose a CSV--')
        self.read_mode.set(None)
        
        for datum in (self.chem_data, self.selections, self.family_mapping):
            datum.clear()
        self.all_species = set()
        self.families = set()
        
        for entry in self.entries:
            entry.reset_default()
        
        self.reset_training()
        
    def average(self, iterable):
        '''Caculcate and return average of an iterable'''
        return sum(iterable)/len(iterable)

                
    #Frame 1 (Reading) Methods 
    def isolate_species(self, species):
        '''Strips extra numbers off the end of the name of a species in a csv and just tells you the species name'''
        return re.sub('(\s|-)\d+\s*\Z', '', species)  # regex to crop off terminal digits ofin a variety of possible 
    
    def get_family(self, species):
        '''Takes the name of a species and returns the chemical family that that species belongs to, based on IUPAC naming conventions'''
        iupac_suffices = {  'ate':'Acetates',
                            'ol':'Alcohols',
                            'al':'Aldehydes',
                            'ane':'Alkanes',
                            'ene':'Alkenes',
                            'yne':'Alkynes',
                            'ine':'Amines',
                            'oic acid': 'Carboxylic Acids',
                            #'ate':'Esters',
                            'ether':'Ethers',
                            'one':'Ketones'  }                    
        for regex, family in iupac_suffices.items():
            # ratioanle for regex: ignore capitalization (particular to ethers), only check end of name (particular to pinac<ol>one)
            if re.search('(?i){}\Z'.format(regex), self.isolate_species(species) ):  
                return family
    
    def read_chem_data(self): 
        '''Used to read and format the data from the csv provided into a form usable by the training program
        Returns the read data (with vector) and sorted lists of the species and families found in the data'''
        csv_name = './{}.csv'.format( self.chosen_file.get() )
        with open(csv_name, 'r') as file:
            for row in csv.reader(file):
                name = row[0]
                spectrum_data = [float(i) for i in row[1:]]  # convert data point from str to floats
                
                self.chem_data[name] = spectrum_data
                self.all_species.add( self.isolate_species(name) )
                self.families.add( self.get_family(name) )
                if not self.spectrum_size:
                    self.spectrum_size = len(spectrum_data)

        self.upper_bound_entry.set_value(self.spectrum_size)
        self.all_species, self.families = sorted(self.all_species), sorted(self.families)  # sort and convert to lists
        
        for index, family in enumerate(self.families):
            one_hot_vector = tuple(i == index and 1 or 0 for i in range(len(self.families)) )
            self.family_mapping[family] = one_hot_vector
                                   
        for species, data in self.chem_data.items():  # add mapping vector to all data entries
            vector = self.family_mapping[ self.get_family(species) ]
            self.chem_data[species] = (data, vector)
    
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
            self.read_status.set_status(True)
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
            SelectionWindow(self.main, self.input_frame, '270x85', self.families, self.selections, ncols=3)

    def confirm_inputs(self):
        '''Confirm species input selections'''
        if self.selections == []:
            messagebox.showerror('No selections made', 'Please select species to evaluate')
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
        self.trim_switch.disable()   #ensures that the trimming menu stays greyed out, not necessary for the other switches 
    
    
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
        self.train_button.configure(state='disabled')
        self.train_window = TrainingWindow(self.main, self.total_rounds, self.num_epochs, self.begin_training, self.reset, self.train_button)
        self.keras_callbacks.append( TkEpochs(self.train_window) )
        self.training()
                
    def training(self, verbosity=False):
        '''The neural net training function itself'''
        start_time = time()    # log start of runtime
        num_spectra = 2
        current_round = 0       
        RIP_trimming, fam_training = self.trim_switch.value, self.fam_switch.value 
        
        for filename in os.listdir('.'):       # deletes results folders from prior trainings to prevent overwriting
            if re.search('\A(Training Results)', filename):
                rmtree('./%s'% filename, ignore_errors=True)
        #self.results_folder = './Training Results'
        self.results_folder = './Training Results, {}'.format(strftime('%m-%d-%y at %H;%M;%S'))   # maine results folder is named with time and date of training start
        os.makedirs(self.results_folder)

        with open('./{}/Training Settings.txt'.format(self.results_folder), 'a') as settings_file:  # make these variables more compact at some point
            settings_file.write('Familiar Training? : {}\n'.format(fam_training))
            settings_file.write('Number of Epochs : {}\n'.format(self.num_epochs))
            settings_file.write('Batchsize : {}\n'.format(self.batchsize))
            settings_file.write('Learn Rate : {}\n'.format(self.learnrate))
                  
        for instance, member in enumerate(self.selections):
            for select_RIP in range(1 + int(RIP_trimming)):     # treats 0, 1 iteration as bool (optional true)
                for segment in range(select_RIP and self.num_slices or 1): 
                # INITIALIZE SOME INFORMATION REGARDING THE CURRENT ROUND
                    self.train_window.set_status('Training...')
                    
                    current_round += 1
                    self.train_window.set_round_progress(current_round)
                                
                    if select_RIP:
                        lower_bound, upper_bound = self.trimming_min, self.trimming_max - self.slice_decrement*segment
                        point_range = 'Points {}-{}'.format(lower_bound, upper_bound)
                    else:
                        lower_bound, upper_bound =  0, self.spectrum_size
                        point_range = 'Full Spectra'
                    self.train_window.set_slice(point_range) 
                    
                    curr_family = self.get_family(member)

                # EVALUATION AND TRAINING SET SELECTION AND SPLITTING 
                    plot_list = []
                    eval_data, eval_titles = [], []
                    features, labels, occurrences = [], [], Counter()
                    train_set_size, eval_set_size = 0, 0
                         
                    for species, (data, vector) in self.chem_data.items():
                        data = data[lower_bound:upper_bound]                      
                        #if re.search('\A{}.*'.format(member), species):       # if the current species begins with <member>
                        if self.isolate_species(species) == member:
                            eval_set_size += 1
                            eval_data.append(data)
                            eval_titles.append(species)
                                                        
                            if eval_set_size <= num_spectra:
                                plot_list.append( (data, species, 's') )
                                
                            if fam_training:
                                train_set_size += 1
                                features.append(data)
                                labels.append(vector)
                                occurrences[self.get_family(species)] += 1                         
                        else:                                         
                            train_set_size += 1
                            features.append(data)
                            labels.append(vector)
                            occurrences[self.get_family(species)] += 1
                    self.train_window.set_member( '{} ({} instances found)'.format(member, eval_set_size) )                      
                    x_train, x_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.2)                     

                    if verbosity:   # optional prinout, gives overview of the training data and the model settings
                        for (x, y, group) in ( (x_train, y_train, 'training'), (x_test, y_test, 'test') ):
                            print('{} features/labels in {} set ({} of the data)'.format(
                                  (x.shape[0], y.shape[0]), group, round(100 * x.shape[0]/train_set_size, 2)) )
                        print('\n{} features total. Of the {} species in training dataset:'.format(len(self.chem_data), train_set_size) )
                        for family in self.family_mapping.keys():
                            print('    {}% of data are {}'.format( round(100 * occurrences[family]/train_set_size, 2), family) )
                  
                # MODEL CREATION AND TRAINING
                    with tf.device('CPU:0'):                            
                        model = Sequential()                              # model block is created, layers are created/added in this block
                        model.add(Dense(512, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer
                        model.add(Dropout(0.5))                                    # dropout layer, to reduce overfit
                        #model.add(Dense(512, activation='relu'))                  # 512 neuron hidden layer
                        model.add(Dense(512, activation='relu'))                   # 512 neuron hidden layer
                        model.add(Dense(len(self.families), activation='softmax')) # softmax gives prob. dist. of identity over all families
                        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learnrate), metrics=['accuracy']) 
                    #model.summary()
                      
                    hist = model.fit(x_train, y_train, epochs=self.num_epochs, batch_size=self.batchsize, callbacks=self.keras_callbacks, verbose=verbosity and 2 or 0)  
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=(verbosity and 2 or 0))  # keras' self evaluation of loss and accuracy metrics
                    
                    if self.train_window.end_training:  # condition to escape training loop of training is aborted
                        messagebox.showerror('Training has Stopped', 'Training aborted by user;\nProceed from Progress Window')
                        self.train_window.button_frame.enable()
                        return     # without this, aborting training only pauses one iteration of loop

                # PREDICTION OVER EVALUATION SET, EVALUATION OF PERFORMANCE                 
                    targets, num_correct = [], 0    # produce prediction values using the model and determine the accuracy of these predictions
                    preds = [list(prediction) for prediction in model.predict(np.array(eval_data))]
                    
                    for prediction in preds:
                        target_index = self.family_mapping[curr_family].index(1)
                        target = prediction[target_index]
                        targets.append(target)
                        
                        if max(prediction) == target:
                            num_correct += 1
                    targets.sort(reverse=True)
                    fermi_data = [AAV/max(targets) for AAV in targets]
                    
                # PACKAGING OF ALL PLOTS, APART FROM THE SAMPLE SPECTRA
                    loss_plot = (hist.history['loss'], 'Training Loss (Final = %0.2f)' % test_loss, 'm') 
                    accuracy_plot = (hist.history['accuracy'], 'Training Accuracy (Final = %0.2f%%)' % (100 * test_acc), 'm') 
                    fermi_plot = (fermi_data, '{}, {}/{} correct'.format(member, num_correct, eval_set_size), 'f')  
                    summation_plot = ([self.average(column) for column in zip(*preds)], 'Standardized Summation', 'p')
                    prediction_plots =  zip(preds, eval_titles, tuple('p' for i in preds))   # all the prediction plots                    
                    
                    for plot in (loss_plot, accuracy_plot, fermi_plot, summation_plot, *prediction_plots): 
                        plot_list.append( plot )    

                # ORGANIZATION AND ADDITION OF RELEVANT DATA TO THE SUMMARY DICT
                    if point_range not in self.summaries:    # adding relevant data to the summary dict                                 
                        self.summaries[point_range] = ( [], {} )
                    fermi_data, score_data = self.summaries[point_range]
                    
                    fermi_data.append(fermi_plot)

                    if curr_family not in score_data:
                        score_data[curr_family] = ( [], [] )
                    names, scores = score_data[curr_family]
                    names.append(member)
                    scores.append(num_correct/eval_set_size)

                # CREATION OF FOLDERS, IF NECESSARY, AND PLOTS FOR THE CURRENT ROUND
                    self.train_window.set_status('Writing Results to Folders...')    # creating folders as necessary, writing results to folders
                    dir_name = './{}/{}'.format(self.results_folder, point_range)   
                    if not os.path.exists(dir_name):                                                           
                        os.makedirs(dir_name)
                    self.adagraph(plot_list, 6, lower_bound, upper_bound, '{}/{}.png'.format(dir_name, member))
                    gc.collect()    # collect any junk remaining in RAM
                    
        self.train_window.set_status('Distributing Result Summaries...')   # distribution of summary data to the appropriate respective folders
        for point_range, (fermi_data, score_data) in self.summaries.items(): 
            curr_dir = './{}/{}/'.format(self.results_folder, point_range)
            self.adagraph(fermi_data, 5, None, None, curr_dir + 'Fermi Summary.png')
                
            with open(curr_dir + 'Scores.txt'.format(point_range), 'a') as score_file:
                for family, (names, scores) in score_data.items():
                    family_header = '{}\n{}\n{}\n'.format('-'*20, family, '-'*20)
                    score_file.write(family_header)    # an underlined heading for each family

                    names.append('AVERAGE')
                    scores.append(self.average(scores))
                    processed_scores = sorted(zip(names, scores), key=lambda x : x[1], reverse=True)  # zip the scores together, then sort them in ascending order by score

                    for name, score in processed_scores:
                        score_file.write('{} : {}\n'.format(name, score))
                    score_file.write('\n')   # leave a gap between each family
        
        self.train_window.button_frame.enable()
        self.train_window.set_status('Finished')
        
        runtime = timedelta(seconds=round(time() - start_time))   
        with open('./{}/Training Settings.txt'.format(self.results_folder), 'a') as settings_file:
            settings_file.write('\nTraining Time : {}'.format(runtime))
        messagebox.showinfo('Training Complete', 'Routine completed in {}\nResults can be found in "Training Results" folder'.format(runtime) )
    
    
    def reset_training(self):
        if self.train_window: # if a window already exists
            self.train_window.destroy()
            self.train_window = None
        self.summaries.clear()
        self.keras_callbacks.clear()
    
    def adagraph(self, plot_list, ncols, lower_bound, upper_bound, save_dir):  # ADD AXIS LABELS!
        '''a general tidy internal graphing utility of my own devising, used to produce all manner of plots during training with one function'''
        nrows = math.ceil(len(plot_list)/ncols)  #  determine the necessary number of rows needed to accomodate the data
        display_size = 20                        # 20 seems to be good size for jupyter viewing
        fig, axs = plt.subplots(nrows, ncols, figsize=(display_size, display_size * nrows/ncols)) 
        
        for idx, (plot_data, plot_title, plot_type) in enumerate(plot_list):                         
            if nrows > 1:                        # locate the current plot, unpack linear index into coordinate
                row, col = divmod(idx, ncols)      
                curr_plot = axs[row][col]  
            else:                                # special case for indexing plots with only one row; my workaround of implementation in matplotlib
                curr_plot = axs[idx]    
            curr_plot.set_title(plot_title)
            
            if plot_type == 's':                 # for plotting spectra
                curr_plot.plot(range(lower_bound, upper_bound), plot_data, 'k,') 
                curr_plot.axis( [lower_bound , upper_bound+1, min(plot_data), max(plot_data)] )
            elif plot_type == 'm':               # for plotting metrics from training
                line_color = ('Loss' in plot_title and 'r' or 'g')
                curr_plot.plot(range(1, self.num_epochs + 1), plot_data, line_color) 
            elif plot_type == 'f':               # for plotting fermi-dirac plots
                num_AAVs = len(plot_data)
                curr_plot.plot( range(num_AAVs), plot_data, linestyle='-', color='m')  # normalized by dividing by length
                curr_plot.axis( [0, num_AAVs, 0, 1.05] )
            elif plot_type == 'p':               # for plotting predictions
                bar_color = ('Summation' in plot_title and 'r' or 'b')
                curr_plot.bar( self.family_mapping.keys(), plot_data, color=bar_color)  
                curr_plot.set_ylim(0,1)
                curr_plot.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close('all')

if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
