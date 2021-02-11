# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of MS Neural Network applications
import plotutils           # library of plotting utilities useful for summarizing and visualizing training results
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 

# Built-in Imports
import json, os
from time import time                      
from pathlib import Path

# Built-In GUI imports
import tkinter as tk   
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog

# PIP-installed Imports                
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf     # Neural Net Libraries
from tensorflow.keras import metrics                  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
tf.get_logger().setLevel('ERROR') # suppress deprecation warnings which seem to riddle tensorflow


# SECTION 1 : custom classes needed to operate some features of the main GUI  ---------------------------------------------------                   
class TkEpochs(Callback):   
    '''A custom keras Callback which interfaces between the training window and the Keras model training session'''
    def __init__(self, training_window):
        super(TkEpochs, self).__init__()
        self.tw = training_window
    
    def on_epoch_begin(self, epoch, logs=None): # update the epoch progress bar at the start of each epoch
        self.tw.epoch_progress.set_progress(epoch + 1)
        self.tw.app.main.update()
        
    def on_epoch_end(self, epoch, logs=None):  # abort training if the abort flag has been raised by the training window
        if self.tw.end_training:
            self.model.stop_training = True
            self.tw.set_status('Training Aborted')

class TrainingWindow(): 
    '''The window which displays training progress and information, subclassed TopLevel allows it to be separate from the main GUI'''
    def __init__(self, main_app):
        self.app = main_app # the main app GUI itself (as an object)
        self.training_window = tk.Toplevel(self.app.main)
        self.training_window.title('Training Progress')
        #self.training_window.geometry('390x210')
        self.training_window.attributes('-topmost', True)
        self.end_training = False # flag for aborting training
        
        # Status Printouts
        self.status_frame = tk.Frame(self.training_window, bd=2, padx=9, relief='groove')
        self.status_frame.grid(row=0, column=0)
        
        self.file_label     = tk.Label(self.status_frame, text='Current Data File: ')
        self.curr_file      = tk.Label(self.status_frame)
        self.evaluand_label = tk.Label(self.status_frame, text='Current Evaluand: ')
        self.curr_evaluand  = tk.Label(self.status_frame)
        self.slice_label    = tk.Label(self.status_frame, text='Current Data Slice: ')
        self.curr_slice     = tk.Label(self.status_frame)
        self.fam_label      = tk.Label(self.status_frame, text='Evaluation Type: ')
        self.curr_fam       = tk.Label(self.status_frame)
        
        self.epoch_label       = tk.Label(self.status_frame, text='Training Epoch: ')
        self.epoch_progress    = ttl.NumberedProgBar(self.status_frame, row=4, col=1)
        self.round_label       = tk.Label(self.status_frame, text='Evaluation Round: ')
        self.round_progress    = ttl.NumberedProgBar(self.status_frame, style_num=2, row=5, col=1) 
        self.file_num_label    = tk.Label(self.status_frame, text='Datafile Number: ')
        self.file_num_progress = ttl.NumberedProgBar(self.status_frame, style_num=3, row=6, col=1) 
        
        self.status_label   = tk.Label(self.status_frame, text='Current Status: ')
        self.curr_status    = tk.Label(self.status_frame)
        
        self.file_label.grid(    row=0, column=0, sticky='w')
        self.curr_file.grid(     row=0, column=1, sticky='w')
        self.evaluand_label.grid(row=1, column=0, sticky='w')
        self.curr_evaluand.grid( row=1, column=1, sticky='w')
        self.slice_label.grid(   row=2, column=0, sticky='w')
        self.curr_slice.grid(    row=2, column=1, sticky='w')
        self.fam_label.grid(     row=3, column=0, sticky='w')
        self.curr_fam.grid(      row=3, column=1, sticky='w')
        
        self.epoch_label.grid(   row=4, column=0, sticky='w')
        #self.epoch_progress, like all ttl widgets, has gridding built-in
        self.round_label.grid(   row=5, column=0, sticky='w')      
        #self.round_progress, like all ttl widgets, has gridding built-in
        self.file_num_label.grid(row=6, column=0, sticky='w')
        #self.file_num_progress, like all ttl widgets, has gridding built-in
        
        self.status_label.grid(  row=7, column=0, sticky='w')
        self.curr_status.grid(   row=7, column=1, sticky='w')
        
        self.readouts  = (self.curr_file, self.curr_evaluand, self.curr_slice, self.curr_fam) # omitted self.curr_status, as it is not reset in the same way as the other readouts
        self.prog_bars = (self.round_progress, self.epoch_progress)
        
        #Training Buttons
        self.button_frame = ttl.ToggleFrame(self.training_window, text='', padx=0, pady=0, row=1) # leave commands as they are (internal methods destroy train window also) 
        
        self.retrain_button    = tk.Button(self.button_frame, text='Retrain', width=17, underline=2, bg='deepskyblue2', command=self.retrain) 
        self.reset_main_button = tk.Button(self.button_frame, text='Reset',   width=17, underline=0, bg='orange', command=self.reset_main)
        self.quit_button       = tk.Button(self.button_frame, text='Quit',    width=17, underline=0, bg='red', command=self.app.quit)
        
        self.retrain_button.grid(   row=0, column=0)
        self.reset_main_button.grid(row=0, column=1)
        self.quit_button.grid(      row=0, column=2) 
        
        # Abort Button, standalone and frameless
        self.abort_button = tk.Button(self.training_window, text='Abort Training', width=54, underline=1, bg='sienna2', command=self.abort)
        self.abort_button.grid(row=2, column=0, pady=(0,2))
        
        self.training_window.bind('<Key>', self.key_bind)
        self.reset() # ensure menu begins at default status when instantiated
     
    def retrain(self):
        self.destroy()      # kill self (to avoid persistence issues)
        self.app.training() # run the main window's training function
        
    def reset_main(self):
        self.destroy()   # kill self (to avoid persistence issues)
        self.app.reset() # reset the main window
    
    def abort(self):
        self.end_training = True
        self.reset() # has to be in this order in order for buttons to stay on after abortion
        self.button_frame.enable()
         
    def key_bind(self, event):
        '''command to bind hotkeys, contingent on menu enabled status'''
        if self.button_frame.state == 'normal':
            if event.char == 't':
                self.retrain()
            elif event.char == 'r':
                self.reset_main()
            elif event.char == 'q':
                self.app.quit()
        elif self.abort_button.cget('state') == 'normal' and event.char == 'b':
            self.abort()
              
    def reset(self):
        for prog_bar in self.prog_bars:
            prog_bar.set_progress(0)
            self.app.main.update()
            
        for readout in self.readouts:
            self.set_readout(readout, '---')
        
        self.set_status('Standby')
        self.button_frame.disable()
    
    def set_readout(self, readout, value):
        '''Base method for updating a readout on the menu'''
        readout.configure(text=value)
        self.app.main.update()
    
    def set_file(self, file):
        self.set_readout(self.curr_file, file)
    
    def set_evaluand(self, evaluand):
        self.set_readout(self.curr_evaluand, evaluand)
    
    def set_slice(self, data_slice):
        self.set_readout(self.curr_slice, data_slice)
    
    def set_familiar_status(self, fam_status):
        self.set_readout(self.curr_fam, fam_status)
    
    def set_status(self, status):
        self.set_readout(self.curr_status, status)
    
    def destroy(self):
        '''Wrapper for builtin Tkinter destroy method'''
        self.training_window.destroy()
        
        
# Section 2: Start of code for the actual GUI and application ---------------------------------------------------------------------------------------
class PLATINUMS_App:
    '''PLATINUMS : Prediction, Training, And Labelling INterface for Unlabelled Mobility Spectra'''
    data_path = Path('TDMS Datasets')          # specify here which folder to look in to find datasets
    save_path = Path('Saved Training Results') # specify here which folders to save results to
    
    def __init__(self, main):       
    #Main Window
        self.main = main
        self.main.title('PLATINUMS v-6.0.0a') # set name with version here
        self.parameters = {}
        
        self.data_path.mkdir(exist_ok=True)  
        self.save_path.mkdir(exist_ok=True)  # will make directories if they don't already exist

    # General Buttons
        self.tpmode = tk.BooleanVar() # option to switch from training to prediction mode, WIP and disabled for now
        
        self.quit_button   = tk.Button(self.main, text='Quit', underline=0, padx=22, pady=11, bg='red', command=self.quit)
        self.tpmode_button = tk.Checkbutton(self.main, text='Predict', var=self.tpmode, command=self.switch_tpmode, state='disabled')
        self.reset_button  = tk.Button(self.main, text='Reset', underline=0, padx=20, bg='orange', command=self.reset)
        
        self.quit_button.grid(  row=0, column=4, sticky='s')
        self.tpmode_button.grid(row=2, column=4)
        self.reset_button.grid( row=6, column=4, padx=2)
        
        self.main.bind('q', lambda event : self.quit()) # universally bind quit and reset actions to keys as well
        self.main.bind('r', lambda event : self.reset())
        
    # Frame 0
        self.spectrum_size  = 0  # initialize empty variables for various data attributes
        self.species        = []
        self.families       = []
        self.family_mapping = {} 
        
        self.input_frame = ttl.ToggleFrame(self.main, text='Select Parameter Input Method: ', padx=4)
        self.input_mode  = tk.StringVar() 
        for i, mode in enumerate(('Manual Input', 'Preset from File')):  # build selection type buttons sequentially w/o class references (not needed)
            tk.Radiobutton(self.input_frame, text=mode, value=mode, underline=0, padx=9, var=self.input_mode, command=self.initial_input).grid(row=0, column=i)
        
        self.input_button = tk.Button(self.input_frame, text='Confirm Selection', command=self.confirm_input_mode, bg='deepskyblue2', underline=0, padx=4)
        self.input_button.grid(row=0, column=2)
        
    #Frame 1
        self.selection_frame = ttl.ToggleFrame(self.main, text='Select Instances to Evaluate: ', padx=5, row=2)
        self.read_mode       = tk.StringVar()
        
        for i, (mode, underline) in enumerate( (('Select All', 7), ('By Family', 3), ('By Species', 3)) ): # build selection type buttons sequentially w/o class references
            tk.Radiobutton(self.selection_frame, text=mode, value=mode, var=self.read_mode, underline=underline, command=self.further_sel).grid(row=0, column=i)
        
        self.confirm_sels = tk.Button(self.selection_frame, text='Confirm Selection', command=self.confirm_selections, bg='deepskyblue2', underline=0, padx=4)
        self.confirm_sels.grid(row=0, column=3)
        
    #Frame 2
        self.hyperparam_frame    = ttl.ToggleFrame(self.main, text='Set Hyperparameters: ', padx=8, row=3)
        
        self.epoch_entry         = ttl.LabelledEntry(self.hyperparam_frame, text='Epochs:',    var=tk.IntVar(),    default=2048, width=18, row=0, col=0)
        self.batchsize_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Batchsize:', var=tk.IntVar(),    default=32,   width=18, row=1, col=0)
        self.learnrate_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Learnrate:', var=tk.DoubleVar(), default=2e-5, width=18, row=0, col=3)
        
        self.confirm_hyperparams = tk.Button(self.hyperparam_frame, text='Confirm Selection', command=self.confirm_hp, padx=5, bg='deepskyblue2')
        self.confirm_hyperparams.grid(row=1, column=3, columnspan=2, sticky='e')
        
    #Frame 3
        self.slicing_frame       = ttl.ToggleFrame(self.main, text='Set Slicing Parameters: ', padx=7, row=4)  
        
        self.n_slice_entry       = ttl.LabelledEntry(self.slicing_frame, text='Slices:',    var=tk.IntVar(), default=0,  width=18, row=0, col=0)
        self.lower_bound_entry   = ttl.LabelledEntry(self.slicing_frame, text='Bottom:',    var=tk.IntVar(), default=0,  width=18, row=1, col=0)
        self.slice_dec_entry     = ttl.LabelledEntry(self.slicing_frame, text='Decrement:', var=tk.IntVar(), default=10, width=18, row=0, col=3) 
        
        self.confirm_sliceparams = tk.Button(self.slicing_frame, text='Confirm Selection', command=self.confirm_sparams, padx=5, bg='deepskyblue2')
        self.confirm_sliceparams.grid(row=1, column=3, columnspan=2, sticky='e')
        
    #Frame 4
        self.param_frame  = ttl.ToggleFrame(self.main, text='Set Training Parameters: ',  padx=6, row=5) 
        self.fam_switch   = ttl.Switch(self.param_frame, text='Familiar Training :', underline=0, row=0, col=0)
        self.cycle_switch = ttl.Switch(self.param_frame, text='Cycle Familiars: ',   underline=1, row=1, col=0)
        self.save_switch  = ttl.Switch(self.param_frame, text='Save Weights:',       underline=5, row=2, col=0)
        self.stop_switch  = ttl.Switch(self.param_frame, text='Early Stopping: ',    underline=0, row=3, col=0)       
        
        self.blank_var1   = tk.BooleanVar()
        self.blank_var2   = tk.BooleanVar()
        self.save_preset  = tk.BooleanVar()     
        
        self.blank_option1_button = tk.Checkbutton(self.param_frame, text='      ', var=self.blank_var1, padx=5) # leaving room for future expnadability
        self.blank_option2_button = tk.Checkbutton(self.param_frame, text='      ', var=self.blank_var2, padx=5)
        self.save_preset_button   = tk.Checkbutton(self.param_frame, text='Save Preset to Main?', var=self.save_preset, underline=0, padx=5)  
        
        self.blank_option1_button.grid(row=0, column=2, sticky='e', padx=(0, 90))
        self.blank_option2_button.grid(row=1, column=2, sticky='e', padx=(0, 90))
        self.save_preset_button.grid(  row=2, column=2, sticky='e', padx=(21, 0))
        
        self.confirm_train_params = tk.Button(self.param_frame, text='Confirm Selection', command=self.confirm_tparams, bg='deepskyblue2', underline=0, padx=6)
        self.confirm_train_params.grid(row=3, column=2, sticky='e')

    # Frame 5 - contains only the button used to trigger a main action  
        self.activation_frame = ttl.ToggleFrame(self.main, text='', padx=0, pady=0, row=6)   
        
        self.train_button = tk.Button(self.activation_frame, text='TRAIN', padx=22, width=44, bg='deepskyblue2', underline=0, command=self.training)
        self.pred_button = tk.Button(self.activation_frame, text='PREDICT', padx=22, width=44, bg='mediumorchid2', state='disabled', command=lambda:None)
        
        self.train_button.grid(row=0, column=0) # overlap buttons, so they can displace one another
        self.pred_button.grid( row=0, column=0)
        
        self.switch_tpmode()
         
    # Packaging together some widgets and attributes, for ease of reference (also useful for self.reset() and self.isolate() methods)
        self.arrays      = (self.parameters, self.species, self.families, self.family_mapping) 
        self.switch_vars = (self.read_mode, self.input_mode, self.blank_var1, self.blank_var2, self.save_preset)
        self.frames      = (self.input_frame, self.selection_frame, self.hyperparam_frame, self.slicing_frame, self.param_frame, self.activation_frame)

        self.switch_mapping = {self.fam_switch : 'fam_training',
                               self.save_switch : 'save_weights',
                               self.stop_switch : 'early_stopping',
                               self.cycle_switch : 'cycle_fams'}
        self.hp_entry_mapping = {self.epoch_entry : 'num_epochs',
                                 self.batchsize_entry : 'batchsize',
                                 self.learnrate_entry : 'learnrate'}
        self.slice_entry_mapping = {self.lower_bound_entry : 'trimming_min', 
                                    self.slice_dec_entry : 'slice_decrement',
                                    self.n_slice_entry : 'num_slices'}
        self.entry_mapping = {**self.hp_entry_mapping, **self.slice_entry_mapping} # merged internal dict of all entries present  
        
        self.main.bind('<Key>', self.key_in_input) # activate internal conditional hotkey binding
        self.reset() # set menu to default configuration
        
    #General Methods
    def lift(self):
        '''Bring GUI window to front of tabs'''
        self.main.attributes('-topmost', True)
        self.main.attributes('-topmost', False)
    
    def isolate(self, on_frame):
        '''Enable just one frame.'''
        for frame in self.frames:
            if frame == on_frame:
                frame.enable()
            else:
                frame.disable()   
    
    def quit(self):
        '''Close the application, with confirm prompt'''
        if messagebox.askokcancel('Quit PLATINUMS?', 'Are you sure you want to close?'):
            self.main.destroy()
    
    def switch_tpmode(self):
        '''Used to switch the mode of the training button; planned future feature, WIP at the moment'''
        target_button = self.tpmode.get() and self.pred_button or self.train_button
        target_button.tkraise()
    
    def key_in_input(self, event):
        '''Hotkey binding wrapper for all frames - ensures actions are only available when the parent frame is enabled'''
        if self.input_frame.state == 'normal': # do not allow hotkeys to work if frame is disabled
            if event.char in ('m', 'p'): # bindings for input modes
                self.input_mode.set((event.char == 'm' and 'Manual Input' or 'Preset from File'))
                self.initial_input()
            elif event.char == 'c': # binding for confirmation
                self.confirm_input_mode()
        elif self.selection_frame.state == 'normal':
            opt_mapping = {'a' : 'Select All',
                           'f' : 'By Family', 
                           's' : 'By Species'}
            if event.char in opt_mapping:
                self.read_mode.set(opt_mapping[event.char]) # choose one of the read modes if the proper hotkey is selected
                self.further_sel() # open further selection menu if not just selecting all
            elif event.char == 'c':
                self.confirm_selections()
        #elif self.hyperparam_frame.state == 'normal' and event.char == 'c': # disabled for now, as the character "c" gets typed into entry
            #self.confirm_hp()
        elif self.param_frame.state == 'normal':
            switch_mapping = {'f' : self.fam_switch,
                              'w' : self.save_switch,
                              'e' : self.stop_switch,
                              'y' : self.cycle_switch}
            if event.char in switch_mapping:
                switch_mapping[event.char].toggle() # make parameter switches toggleable with keyboard
            elif event.char == 's': # add hotkey to save preset and proceed...
                self.save_preset.set(not self.save_preset.get()) # invert cycling status
            #elif event.char == 'n': # add hotkey to save preset and proceed...
            #    self.blank_var2.set(not self.blank_var2.get()) # invert cycling status
            elif event.char == 'c': # or just to proceed
                self.confirm_tparams()
        elif self.activation_frame.state == 'normal' and event.char == 't': # do not allow hotkeys to work if frame is disabled
            self.training()
    
    def reset(self):
        '''reset the GUI to the opening state, along with all variables'''
        for widget in self.main.winfo_children(): # close any TopLevel windows that may happen to be open when reset is hit (be it for selection or training)
            if isinstance(widget, tk.Toplevel):
                widget.destroy()
        
        for tk_switch_var in self.switch_vars:
            tk_switch_var.set(0)
        
        self.spectrum_size = 0
        for array in self.arrays:
            array.clear()
        
        for switch in self.switch_mapping.keys():
            switch.disable()
        
        for entry in self.entry_mapping.keys():
            entry.reset_default()   
        
        self.lift() # bring main window to forefront
        self.isolate(self.input_frame)

    # Frame 0 (Parameter Input) Methods
    def initial_input(self):
        '''Method called for initiating input (be it manual or from preset)'''
        if self.input_mode.get() == 'Manual Input': 
            self.parameters['data_files'] = [] # ensure a list exists AND is empty when selecting files
            ttl.SelectionWindow(self.main, self.input_frame, sorted(iumsutils.get_by_filetype('.json', self.data_path)),
                                self.parameters['data_files'], window_title='Select data file(s) to train over', ncols=4)    
            
        elif self.input_mode.get() == 'Preset from File': 
            preset_path = Path(filedialog.askopenfilename(title='Choose Training Preset', initialdir='./Training Presets', filetypes=(('JSONs', '*.json'),) ))
            try: 
                with preset_path.open() as preset_file: # load in the parameter preset (ALL settings)   
                    self.parameters = json.load(preset_file)   
            except PermissionError: # do nothing if user cancels selection
                self.input_mode.set(0) 
    
    def check_input_files(self):
        '''Performs check to determine whether the files chosen for training have compatible data''' 
        try:
            if not self.parameters['data_files']:
                messagebox.showerror('File Error', 'No data file(s) selected')
                self.reset()
            else:        
                for file in self.parameters['data_files']:
                    json_data = iumsutils.load_chem_json(self.data_path/f'{file}.json')
                    for data_property in ('spectrum_size', 'species', 'families', 'family_mapping'):
                        self.main.update() # update main wndow every cycle to ensure window/cursor doesn't freeze during checking
                        if not getattr(self, data_property): # check if the fields are empty, relies on attribute names being same as json keys - note lack fo default attr
                            setattr(self, data_property, json_data[data_property])
                        else:
                            if data_property == 'spectrum_size':
                                self.spectrum_size = min(self.spectrum_size, json_data['spectrum_size']) # ensures min_spectrum size is indeed the minimum across all files
                            elif json_data[data_property] != getattr(self, data_property):
                                messagebox.showerror('Property Mismatch', f'Attribute "{data_property}" in {file} does not match that of the other files chosen')
                                return # break out of loops
        except KeyError: # catch when the 'data_files' entry does not exist
            messagebox.showerror('Files Undefined', 'Property "data_files" missing from parameters, please check preset')
            self.reset()
    
    def confirm_input_mode(self):
        '''Performs final check over input and determines how to proceed appropriately'''
        if self.input_mode.get() == 'Manual Input':
            self.main.config(cursor='wait') # this make take a moment for large sets of files, indicate to user (via cursor) to be patient
            self.check_input_files()
            self.main.config(cursor='') # return cursor to normal
            self.isolate(self.selection_frame) 
        elif self.input_mode.get() == 'Preset from File':
            try: # filling in all the fields in the GUI based on the selected preset
                self.main.config(cursor='wait') # this make take a moment for large sets of files, indicate to user (via cursor) to be patient
                self.check_input_files()
                self.confirm_selections() # handles the case when family or all are passed explicitly as selections  
                self.read_mode.set(self.parameters['read_mode'])

                for switch, param in self.switch_mapping.items():
                    switch.apply_state(self.parameters[param]) # configure all the switch values in the GUI

                for entry, param in self.entry_mapping.items():
                    entry.set_value(self.parameters[param]) # configure all the entry values in the GUI

                self.check_trimming_bounds() # ensure that bounds actually make sense  
                self.main.config(cursor='') # return cursor to normal
                self.isolate(self.activation_frame)
            except KeyError as error: # gracefully handle the case when the preset does not contain the correct names
                messagebox.showerror('Preset Error', f'The parameter "{error}" is either missing or misnamed;\n Please check preset file for errors') 
                self.reset()
                
        else:
            messagebox.showerror('No input mode selected!', 'Please choose either "Manual" or "Preset" input mode')
            self.reset()
                    
        
    # Frame 1 (Evaluand Selection) Methods
    def further_sel(self): 
        '''logic for selection of evaluands to include in training, based on the chosen selection mode'''
        self.parameters['selections'] = [] # empties selections when selection menu is re-clicked (ensures no overwriting or mixing)
        self.parameters['read_mode'] = self.read_mode.get() # record the read mode to parameters with each selection
        
        if self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.selection_frame, self.species, self.parameters['selections'], window_title='Select Species to evaluate', ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.selection_frame, self.families, self.parameters['selections'], window_title='Select Families to evaluate over', ncols=3)
        # case for "Select All" is covered in confirm_selections() in order to be compatible with loading from preset

    def confirm_selections(self):
        '''Parsing and processing of the selections before proceeding'''
        if self.read_mode.get() == 'Select All':
            self.parameters['selections'] = self.species   
        elif self.read_mode.get() == 'By Family':  # pick out species by family if selection by family is made
            self.parameters['selections'] = [species for species in self.species if iumsutils.get_family(species) in self.parameters['selections']]    
        
        if not self.parameters.get('selections'): # if the selections parameter is still empty (or doesn't exist), instruct the user to make a choice
            messagebox.showerror('No evaluands selected', 'Please ensure one or more species have been selected for evaluation')
        else: 
            self.isolate(self.hyperparam_frame)
   
    # Frame 2 (hyperparameter) Methods
    def confirm_hp(self):
        '''Confirm the selected hyperparameters and proceed'''
        for hp_entry, param in self.hp_entry_mapping.items():
            self.parameters[param] = hp_entry.get_value()  
        self.isolate(self.slicing_frame) 
        
    # Frame 3 (slicing parameter) Methods
    def confirm_sparams(self):
        '''Confirm the selected slicing and proceed'''
        for slice_entry, param in self.slice_entry_mapping.items():
            self.parameters[param] = slice_entry.get_value()  
            
        self.check_trimming_bounds() # ensure bounds make sense, notify the user if they don't
        self.isolate(self.param_frame)     
     
    # Frame 4 (training parameter) Methods
    def check_trimming_bounds(self):
        '''Performs error check over the trimming bounds held internally to ensure that sensible bounds are chosen to ensure that training will not fail'''
        trimming_min    = self.parameters['trimming_min'] # have pulled these out here solely for readability
        num_slices      = self.parameters['num_slices']
        slice_decrement = self.parameters['slice_decrement']
        
        if trimming_min < 0 or type(trimming_min) != int: 
            messagebox.showerror('Boundary Value Error', 'Trimming Min must be a positive integer or zero')       
        elif self.spectrum_size - num_slices*slice_decrement <= trimming_min: # note that spectrum size here is the minimum size; this guarantees all datasets will pass
            messagebox.showerror('Boundary Mismatch', 'Upper limit will not always exceed lower;\nDecrease either slice decrement or lower bound')         
    
    def confirm_tparams(self):
        '''Confirm training parameter selections, perform pre-training error checks if necessary'''
        for switch, param in self.switch_mapping.items():
            self.parameters[param] = switch.value # configure parameters based on the switch values

        if self.save_preset.get(): # if the user has elected to save the current preset
            preset_path = Path(filedialog.asksaveasfilename(title='Save Preset to file', initialdir='./Training Presets',
                                                            defaultextension='.json', filetypes=[('JSONs', '*.json')] ))
            try: 
                with open(preset_path, 'w') as preset_file:
                    json.dump(self.parameters, preset_file)
            except PermissionError: # catch the case in which the user cancels the preset saving
                return           
            
        self.isolate(self.activation_frame) # make the training button clickable if all goes well
             
    # Frame 5: the training routine itself, this is where the magic happens
    def training(self, test_set_proportion=0.2, keras_verbosity=0): # use keras_verbosity of 2 for lengthy and detailed console printouts
        '''The function defining the neural network training sequence and parameters'''
        # UNPACKING INTERNAL PARAMETERS DICT AND INITIALIZING SOME VALUES
        overall_start_time = time() # get start of runtime for entire training routine
        plotutils.Base_RC.unit_circle = plotutils.Mapped_Unit_Circle(self.family_mapping) # set mapping for radar charts
        
        data_files       = self.parameters['data_files'] # shadow names to reduce number of lookups and for better code readability
        selections      = self.parameters['selections'] 
        num_epochs      = self.parameters['num_epochs'] 
        batchsize       = self.parameters['batchsize']
        learnrate       = self.parameters['learnrate'] 
         
        slice_decrement = self.parameters['slice_decrement'] 
        trimming_min    = self.parameters['trimming_min'] 
        n_slices        = self.parameters['num_slices'] + 1 # accounts for 0 indexing, makes input easier to understand
        
        cycle_fams      = self.parameters['cycle_fams']
        fam_training    = self.parameters['fam_training']
        save_weights    = self.parameters['save_weights']
        early_stopping  = self.parameters['early_stopping']      

        # FILE MANAGEMENT CODE, GUARANTEES THAT NO ACCIDENTAL OVERWRITES OCCUR AND THAT AN APPROPRIATE EMPTY FOLDER EXISTS TO WRITE RESULTS TO
        parent_name = simpledialog.askstring(title='Set Result Folder Name', prompt='Please enter a name to save training results under:',
                                             initialvalue=f'{num_epochs} epoch, {cycle_fams and "cycled" or (fam_training and "familiar" or "unfamiliar")}')   
        if not parent_name: # if user cancel operation or leaves field blank, reset train option
            self.activation_frame.enable()
            return
            
        else:  # otherwise, begin file checking loop  
            parent_folder = self.save_path/parent_name # same name applied to Path object, useful for incrementing file_id during overwrite checking
            file_id = 0
            while True: 
                try: # if no file exists, can make a new one and proceed without issue
                    parent_folder.mkdir(exist_ok=False) 
                    break
                except FileExistsError: # if one DOES exist, enter checking logic
                    if not any(parent_folder.iterdir()): # if the file exists but is empty, can also proceed with no issues
                        break 
                    elif (parent_folder/'Training Preset.json').exists(): # otherwise, if the folder has a preset present...
                        with (parent_folder/'Training Preset.json').open() as existing_preset_file: 
                            existing_preset = json.load(existing_preset_file)

                        if existing_preset != self.parameters: # ...and if the preset differs from the current preset, start checks over with incremented name
                            file_id += 1
                            parent_folder = self.save_path/f'{parent_name}({file_id})' # appends differentiating number to end of duplicated files 
                            continue # return to start of checking loop

                    # this branch is only executed if no preset file is found OR (implicitly) if the existing preset matches the one found
                    if messagebox.askyesno('Request Overwrite', 'Folder with same name and same (or no) training parameters found; Allow overwrite of folder?'):
                        try:
                            iumsutils.clear_folder(parent_folder) # empty the folder if permission is given
                            break
                        except PermissionError: # if emptying fails because user forgot to close files, notify them and exit training loop
                            messagebox.showerror('Overwrite Error!', f'{parent_folder}\nhas file(s) open and cannot be \
                                                  overwritten;\n\nPlease close all files and try training again')
                    self.activation_frame.enable()
                    return # final return branch executed in all cases except where user allows overwrite AND it is successful
                
        preset_path = parent_folder/'Training Preset.json'
        with preset_path.open(mode='w') as preset_file: 
            json.dump(self.parameters, preset_file) # save current training settings to a preset for reproducability
            
        self.activation_frame.disable()     # disable training button while training (an idiot-proofing measure)                 
        train_window = TrainingWindow(self) # pass the GUI app to the training window object to allow for menu manipulation        
        train_window.round_progress.set_max(len(selections)*n_slices*(1 + cycle_fams)) # compute and display the number of training rounds to occur
        train_window.file_num_progress.set_max(len(data_files))
        
        keras_callbacks = [TkEpochs(train_window)] # in every case, add the tkinter-keras interface callback to the callbacks list
        if early_stopping:
            early_stopping_callback = EarlyStopping(monitor='loss', mode='min', verbose=keras_verbosity, patience=16, restore_best_weights=True) # fine-tune patience, eventually
            keras_callbacks.append(early_stopping_callback) # add early stopping callback if called for
        
        # BEGINNING OF ACTUAL TRAINING CYCLE, ITERATE THROUGH EVERY DATA FILE PASSED
        point_ranges = set() # stores the slicing point ranges, used later to spliced together familiar and unfamiliar reulst files - is a set to avoid duplication
        for file_num, data_file in enumerate(data_files):
            train_window.reset() # clear training window between different files
            train_window.set_status('Initializing Training...') 
            
            train_window.set_file(data_file) # indicate the current file in the training window   
            train_window.file_num_progress.increment()

            results_folder = Path(parent_folder, f'{data_file} Results') # main folder with all results for a particular data file  
            results_folder.mkdir(exist_ok=True)        

            json_data = iumsutils.load_chem_json(self.data_path/f'{data_file}.json') # load in the data file for the current ecaluation round
            chem_data = json_data['chem_data']        # isolate the spectral/chemical data 
            spectrum_size = json_data['spectrum_size'] # set maximum length to be the full spectrum length of the current dataset

            # SECONDARY TRAINING LOOP, REGULATES CYCLING BETWEEN FAMILIARS AND UNFAMILIARS
            for familiar_cycling in range(1 + cycle_fams):    
                start_time = time() # log start time of each cycle, so that training duration can be computed afterwards  
                
                fam_training = (self.parameters['fam_training'] ^ familiar_cycling) # xor used to encode the inversion behavior on second cycle    
                fam_str = f'{fam_training and "F" or "Unf"}amiliar'  
                self.fam_switch.apply_state(fam_training)        # purely cosmetic, but allows the user to see that cycling is in fact occurring
                train_window.set_familiar_status(fam_str)

                curr_fam_folder = results_folder/fam_str # folder with the results from the current (un)familiar training, parent for all that follow 
                curr_fam_folder.mkdir(exist_ok=True)
                log_file_path = curr_fam_folder/'Log File.txt'
                log_file_path.touch() # make sure the log file exists

                # TERTIARY TRAINING LOOP, REGULATES SLICING
                for segment in range(n_slices): 
                    lower_bound, upper_bound = trimming_min, spectrum_size - slice_decrement*segment
                    point_range = f'Points {lower_bound}-{upper_bound}'
                    if lower_bound == 0 and upper_bound == spectrum_size: 
                        point_range = point_range + ' (Full Spectra)' # indicate whether the current slice is in fact truncated
                   
                    point_ranges.add(point_range) # add current range to point ranges (will not duplicate if cycling familiars, since point_ranges is a set)
                    train_window.set_slice(point_range)
                    curr_slice_folder = curr_fam_folder/point_range
                    curr_slice_folder.mkdir(exist_ok=True)

                    # INNERMOST TRAINING LOOP, CYCLES THROUGH ALL THE SELECTED EVALUANDS
                    predictions, round_summary = [
                        {family : 
                            {species : {}
                                for species in sorted((species for species in selections if iumsutils.get_family(species) == family), key=iumsutils.get_carbon_ordering)
                            }
                            for family in sorted(set(iumsutils.get_family(species) for species in selections))
                        } 
                    for i in range(2)]  # outer data structure is the same for both predictions and the round summary
                    
                    for evaluand_idx, evaluand in enumerate(selections):                  
                        train_window.set_status('Training...')
                        train_window.round_progress.increment() 
                        train_window.epoch_progress.set_max(num_epochs) # reset maximum number of epochs on progress bar to specified amount (done each round to handle early stopping)

                        curr_family  = iumsutils.get_family(evaluand)
                        eval_spectra = [instance.spectrum[lower_bound:upper_bound] for instance in chem_data if instance.species == evaluand]
                        eval_titles  = [instance.name for instance in chem_data if instance.species == evaluand]
                        train_window.set_evaluand(f'{evaluand} ({len(eval_titles)} instances found)')

                        # TRAIN/TEST SET CREATION, MODEL CREATION, AND MODEL TRAINING
                        if not fam_training or evaluand_idx == 0: # for familiars, model is only trained during first evaluand at each slice (since training set is identical throughout)
                            features = [instance.spectrum[lower_bound:upper_bound] for instance in chem_data if instance.species != evaluand or fam_training]
                            labels   = [instance.vector for instance in chem_data if instance.species != evaluand or fam_training]
                            x_train, x_test, y_train, y_test = map(np.array, train_test_split(features, labels, test_size=test_set_proportion)) # keras model only accepts numpy arrays

                            with tf.device('CPU:0'):     # eschews the requirement for a brand-new NVIDIA graphics card (which we don't have anyways)                      
                                model = Sequential()     # model block is created, layers are added to this block
                                model.add(Dense(256, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer, size depends on trimming
                                model.add(Dropout(0.5))                                          # dropout layer, to reduce overfit
                                model.add(Dense(256, activation='relu'))                         # 512 neuron hidden layer
                                model.add(Dense(len(self.family_mapping), activation='softmax')) # softmax gives probability distribution of identity over all families
                                model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learnrate), metrics=['accuracy']) 

                            hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=keras_callbacks, # actual model training occurs here
                                             verbose=keras_verbosity, epochs=num_epochs, batch_size=batchsize)
                            final_evals = model.evaluate(x_test, y_test, verbose=keras_verbosity)  # keras' self evaluation of loss and accuracy metric

                            if early_stopping and early_stopping_callback.stopped_epoch: # if early stopping has indeed been triggered:
                                train_window.epoch_progress.set_max(early_stopping_callback.stopped_epoch + 1) # set progress max to stopped epoch, to indicate that training is done   
                                with log_file_path.open(mode='a') as log_file: 
                                    log_file.write(f'{fam_training and fam_str or evaluand} training round stopped early at epoch {early_stopping_callback.stopped_epoch + 1}\n')

                            if save_weights and not train_window.end_training: # if saving is enabled, save the model to the current result directory
                                train_window.set_status('Saving Model...')
                                weights_folder = curr_slice_folder/f'{fam_training and fam_str or evaluand} Model Files'
                                model.save(str(weights_folder)) # path can only be str, for some reason

                                local_preset = {param : value for param, value in self.parameters.items()} # make a deep copy of the parameters that can be modified locally
                                local_preset['data_files']   = [data_file]  # only retrain using the current file
                                local_preset['fam_training'] = fam_training # to account for cycling
                                local_preset['cycle_fams']   = False        # ensure no cycling is performed for reproduction
                                local_preset['selections']   = [evaluand]   # only requires a single evaluand (and in the case of familiars, the particular one is not at all relevant)
                                local_preset['num_slices']   = 0            # no slices, only full data 
                                with open(weights_folder/f'Reproducability Preset ({point_range}).json', 'w') as weights_preset: 
                                    json.dump(local_preset, weights_preset)  # add a preset to each model file that allows for reproduction of ONLY that single model file                      

                        # ABORTION CHECK PRIOR TO WRITING FILES, WILL CEASE IF TRAINING IS ABORTED
                        if train_window.end_training:  
                            messagebox.showerror('Training has Stopped!', 'Training aborted by user;\nProceed from Progress Window', parent=train_window.training_window)
                            train_window.button_frame.enable()
                            return # without a return, aborting training only pauses one iteration of loop

                        # CREATION OF THE SPECIES SUMMARY OBJECT FOR THE CURRENT EVALUAND, PLOTTING OF EVALUATION RESULTS FOR THE CURRENT EVALUAND 
                        train_window.set_status('Plotting Results to Folder...')

                        for inst_name, pred in zip(eval_titles, model.predict(np.array(eval_spectra))): # file predictions into dict
                            predictions[curr_family][evaluand][inst_name] = [float(i) for i in pred] # convert to list of floats for JSON serialization 
                        
                        # obtain and file current evaluand score, plotting summary graphics in the process; overwrites empty dictionary
                        round_summary[curr_family][evaluand] = plotutils.plot_and_get_score(evaluand, eval_spectra, predictions, hist.history['loss'],
                                                                                             hist.history['accuracy'],final_evals, savedir=curr_slice_folder)                
                    # CALCULATION AND PROCESSING OF RESULTS TO PRODUCE SCORES AND SUMMARIES 
                    train_window.set_status(f'Unpacking Scores...')
                    
                    row_labels, score_list = [], []
                    for family, species_scores in round_summary.items(): # iterate through species score mapping (skips over omitted families)     
                        row_labels.append(family)
                        score_list.append(' ') # skip line in score column
                        
                        species_scores['AVERAGE'] = iumsutils.average(species_scores.values()) # compute and append average score to each
                        for species, score in species_scores.items():
                            row_labels.append(species)
                            score_list.append(score)
                        row_labels.append(' ')   
                        score_list.append(' ')   # leave a gap between each family
                                      
                    # WRITING/PLOTTING RESULTS TO FILES
                    train_window.set_status(f'Outputting Summaries of Results...')
                    
                    prediction_path = curr_slice_folder/'Prediction Values.json'
                    with prediction_path.open(mode='w', newline='') as pred_file:
                        json.dump(predictions, pred_file) # save the aavs for the current training

                    plotutils.single_plot(plotutils.Overlaid_Family_RC(predictions), curr_slice_folder/'Overall Summary', figsize=8) # save family-overlaid RC as visual result summary
                           
                    score_file_path = curr_slice_folder/f'{fam_str} Scores.csv'
                    iumsutils.add_csv_column(score_file_path, row_labels)
                    iumsutils.add_csv_column(score_file_path, score_list)
                    
                    comp_path = parent_folder/f'Compiled Results - {point_range}, {fam_str}.csv' # path to the relevant compiled results folder
                    if not comp_path.exists():
                        iumsutils.add_csv_column(comp_path, [fam_str, *row_labels]) # if the relevant file doesn't exists, make it and add row labels
                    iumsutils.add_csv_column(comp_path, [data_file, *score_list]) # add a column with the current scores, along with a dataset label, to the collated csv
                
                # LOGGING THE DURATION OF TRAINING OVER EACH DATASET
                with log_file_path.open(mode='a') as log_file:  # log the time taken to complete the training cycle (open in append mode to prevent overwriting)
                    model.summary(print_fn=lambda x : log_file.write(f'{x}\n')) # write model to log file (since model is same throughout, should only write model on the last pass)
                    log_file.write(f'\nTraining Time : {iumsutils.format_time(time() - start_time)}') # log the time taken for this particular training session as well
        
        # POST-TRAINING WRAPPING-UP   
        if cycle_fams: # if both familiar and unfamiliar traning are being performed, merge the score files together
            for point_range in point_ranges:
                fam_path   = parent_folder/f'Compiled Results - {point_range}, Familiar.csv' 
                unfam_path = parent_folder/f'Compiled Results - {point_range}, Unfamiliar.csv'
                
                with unfam_path.open('r') as unfam_file, fam_path.open('a') as fam_file: # read from unfamiliar scores and append to familiar scores
                    for row in unfam_file:
                        fam_file.write(row)  
                        
                unfam_path.unlink() # delete the unfamiliar file after copying the contents
                fam_path.rename(fam_path.parent/f'Compiled Results - {point_range}.csv' )  # get rid of "Familiar" affix after merging
        
        train_window.button_frame.enable()  # open up post-training options in the training window
        train_window.abort_button.configure(state='disabled') # disable the abort button (idiotproofing against its use after training)
        train_window.set_status(f'Finished in {iumsutils.format_time(time() - overall_start_time)}') # display the time taken for all trainings in the training window
        if messagebox.askyesno('Training Completed Succesfully!', 'Training finished; view training results in folder?', parent=train_window.training_window):
            os.startfile(parent_folder) # notify user that training has finished and prompt them to view the results in situ
        
if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
