# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of MS Neural Network applications
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 

# Built-in Imports
import json, shutil
from time import time                      
from pathlib import Path

# Built-In GUI imports
import tkinter as tk    
from tkinter import messagebox
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
tf.get_logger().setLevel('ERROR') # suppress deprecation warnings which seem to riddle all version of tensorflow


# SECTION 1 : custom classes needed to operate some features of the main GUI  ---------------------------------------------------                   
class TkEpochs(Callback):   
    '''A custom keras Callback which mediates interaction between the training window and the model training/Keras'''
    def __init__(self, training_window):
        super(TkEpochs, self).__init__()
        self.tw = training_window
    
    def on_epoch_begin(self, epoch, logs=None): # update the epoch progress bar at the start of each epoch
        self.tw.set_epoch_progress(epoch + 1)
        
    def on_epoch_end(self, epoch, logs=None):  # abort training if the abort flag has been raised by the training window
        if self.tw.end_training:
            self.model.stop_training = True
            self.tw.set_status('Training Aborted')
        
class TrainingWindow(): 
    '''The window which displays training progress, was easier to subclass outside of the main GUI class'''
    def __init__(self, main, total_rounds, num_epochs, train_funct, reset_main_funct, exit_funct):
        self.main = main
        self.training_window = tk.Toplevel(main)
        self.training_window.title('Training Progress')
        self.training_window.geometry('390x189')
        self.training_window.attributes('-topmost', True)
        self.end_training = False # flag for aborting training
        
        self.train_funct = train_funct
        self.reset_main_funct = reset_main_funct
        
        # Status Printouts
        self.status_frame = tk.Frame(self.training_window, bd=2, padx=9, relief='groove')
        self.status_frame.grid(row=0, column=0)
        
        self.evaluand_label = tk.Label(self.status_frame, text='Current Evaluand: ')
        self.curr_evaluand  = tk.Label(self.status_frame)
        self.slice_label    = tk.Label(self.status_frame, text='Current Data Slice: ')
        self.curr_slice     = tk.Label(self.status_frame)
        self.fam_label      = tk.Label(self.status_frame, text='Evaluation Type: ')
        self.curr_fam       = tk.Label(self.status_frame)
        self.round_label    = tk.Label(self.status_frame, text='Evaluation Round: ')
        self.round_progress = ttl.NumberedProgBar(self.status_frame, maximum=total_rounds, row=3, col=1)
        self.epoch_label    = tk.Label(self.status_frame, text='Training Epoch: ')
        self.epoch_progress = ttl.NumberedProgBar(self.status_frame, maximum=num_epochs, style_num=2, row=4, col=1) 
        self.status_label   = tk.Label(self.status_frame, text='Current Status: ')
        self.curr_status    = tk.Label(self.status_frame)
        
        self.evaluand_label.grid(row=0, column=0, sticky='w')
        self.curr_evaluand.grid( row=0, column=1, sticky='w')
        self.slice_label.grid(   row=1, column=0, sticky='w')
        self.curr_slice.grid(    row=1, column=1, sticky='w')
        self.fam_label.grid(     row=2, column=0, sticky='w')
        self.curr_fam.grid(      row=2, column=1, sticky='w')
        self.round_label.grid(   row=3, column=0, sticky='w')
        #self.round_progress, like all ttl widgets, has gridding built-in
        self.epoch_label.grid(   row=4, column=0, sticky='w')
        #self.epoch_progress, like all ttl widgets, has gridding built-in
        self.status_label.grid(  row=5, column=0, sticky='w')
        self.curr_status.grid(   row=5, column=1, sticky='w')
        
        self.reset() # ensure menu begins at default status when instantiated
    
        #Training Buttons
        self.button_frame        = ttl.ToggleFrame(self.training_window, text='', padx=0, pady=0, row=1)
        self.retrain_button      = tk.Button(self.button_frame, text='Retrain', width=17, bg='deepskyblue2', command=self.retrain)
        self.reset_parent_button = tk.Button(self.button_frame, text='Reset', width=17, bg='orange', command=self.reset_main)
        self.exit_button         = tk.Button(self.button_frame, text='Exit', width=17, bg='red', command=exit_funct)
        
        self.retrain_button.grid(     row=0, column=0)
        self.reset_parent_button.grid(row=0, column=1)
        self.exit_button.grid(        row=0, column=2) 
        
        self.button_frame.disable()
        
        # Abort Button, standalone and frameless
        self.abort_button = tk.Button(self.training_window, text='Abort Training', width=54, bg='sienna2', command=self.abort)
        self.abort_button.grid(row=2, column=0)
     
    def retrain(self):
        self.destroy()  # kill self (to avoid persistence issues)
        self.train_funct() # run the main window's training function
        
    def reset_main(self):
        self.destroy() # kill self (to avoid persistence issues)
        self.reset_main_funct() # reset the main window
    
    def abort(self):
        self.end_training = True
        self.button_frame.enable()
        self.reset()
        
    def reset(self):
        self.set_evaluand('---')
        self.set_slice('---')
        self.set_familiar_status('---')
        self.set_round_progress(0)
        self.set_epoch_progress(0)
        self.set_status('Standby')
    
    def set_readout(self, readout, value):
        '''Base method for updating a readout on the menu'''
        readout.configure(text=value)
        self.main.update()
    
    def set_evaluand(self, evaluand):
        self.set_readout(self.curr_evaluand, evaluand)
    
    def set_slice(self, data_slice):
        self.set_readout(self.curr_slice, data_slice)
    
    def set_status(self, status):
        self.set_readout(self.curr_status, status)
        
    def set_familiar_status(self, fam_status):
        self.set_readout(self.curr_fam, fam_status)
    
    def set_round_progress(self, curr_round):
        self.round_progress.set_progress(curr_round)
        self.main.update()
        
    def set_epoch_progress(self, curr_epoch):
        self.epoch_progress.set_progress(curr_epoch)
        self.main.update()
    
    def destroy(self): # wrapper for the builtin tkinter destroy() method 
        self.training_window.destroy()
        
        
# Section 2: Start of code for the actual GUI and application ---------------------------------------------------------------------------------------
class PLATINUMS_App:
    '''PLATIN-UMS : Prediction, Training, And Labelling INterface for Unlabelled Mobility Spectra'''
    def __init__(self, main):       
        #Main Window
        self.main = main
        self.main.title('PLATIN-UMS 4.7.2-alpha')
        self.main.geometry('445x503')
        self.parameters = {}

        # General Buttons
        self.exit_button  = tk.Button(self.main, text='Exit',  padx=22, pady=11, bg='red', command=self.shutdown)
        self.reset_button = tk.Button(self.main, text='Reset', padx=20, bg='orange', command=self.reset)
        self.exit_button.grid(row=0, column=4, sticky='s')
        self.reset_button.grid(row=5, column=4)
        
        self.tpmode = tk.BooleanVar() # option to switch from training to prediction mode, WIP and low priority for now
        self.tpmode_button = tk.Checkbutton(self.main, text='Predict', var=self.tpmode, command=self.switch_tpmode, state='disabled')
        self.tpmode_button.grid(row=2, column=4)
        
        
        # Frame 0
        self.input_frame = ttl.ToggleFrame(self.main, text='Select Parameter Input Method: ', padx=4)
        
        self.input_mode = tk.StringVar()     
        for i, mode in enumerate(('Manual Input', 'Preset from File')): # build selection type buttons sequentially, don't need to reference these w/in the class after building
            tk.Radiobutton(self.input_frame, text=mode, value=mode, padx=9, var=self.input_mode).grid(row=0, column=i)
        self.confirm_inputs = ttl.ConfirmButton(self.input_frame, command=self.confirm_input_mode, padx=4, row=0, col=2)
        
        
        #Frame 1
        self.file_frame  = ttl.ToggleFrame(self.main, text='Select Data File to Read: ', padx=5, row=1)
        self.chosen_file = tk.StringVar()
        self.chem_data, self.species, self.families, self.family_mapping, self.spectrum_size = [], [], [], {}, 0
        
        data_path = Path('TDMS Datasets') # ensure a datasets folder exists, so that app does not fail on __init__
        if not data_path.exists():
            data_path.mkdir()
            
        self.json_menu      = ttl.DynOptionMenu(self.file_frame, var=self.chosen_file, option_method=iumsutils.get_by_filetype,
                                                opargs=('.json', data_path), default='--Choose a JSON--', width=32, colspan=2)
        self.read_label     = tk.Label(self.file_frame, text='Read Status:')
        self.read_status    = ttl.StatusBox(self.file_frame, on_message='JSON Read!', off_message='No File Read', width=22, row=1, col=1)
        self.refresh_button = tk.Button(self.file_frame, text='Refresh JSONs', command=self.json_menu.update, padx=15)
        self.confirm_data   = ttl.ConfirmButton(self.file_frame, command=self.import_data, padx=5, row=1, col=2)
        
        self.refresh_button.grid(row=0, column=2, sticky='e')
        self.read_label.grid(    row=1, column=0)
        
        
        #Frame 2
        self.selection_frame = ttl.ToggleFrame(self.main, text='Select Instances to Evaluate: ', padx=5, row=2)
        self.read_mode   = tk.StringVar()
        
        for i, mode in enumerate(('Select All', 'By Family', 'By Species')): # build selection type buttons sequentially, don't need to reference these w/in the class after building
            tk.Radiobutton(self.selection_frame, text=mode, value=mode, var=self.read_mode, command=self.further_sel).grid(row=0, column=i)
        self.confirm_sels = ttl.ConfirmButton(self.selection_frame, command=self.confirm_selections, padx=4, row=0, col=3)
        
        
        #Frame 3
        self.hyperparam_frame    = ttl.ToggleFrame(self.main, text='Set Hyperparameters: ', padx=8, row=3)
        self.epoch_entry         = ttl.LabelledEntry(self.hyperparam_frame, text='Epochs:', var=tk.IntVar(),       width=18, default=2048)
        self.batchsize_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Batchsize:', var=tk.IntVar(),    width=18, default=32, row=1)
        self.learnrate_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Learnrate:', var=tk.DoubleVar(), width=18, default=2e-5, col=3)
        self.confirm_hyperparams = ttl.ConfirmButton(self.hyperparam_frame, command=self.confirm_hp, row=1, col=3, cs=2)
        
        
        #Frame 4
        self.param_frame = ttl.ToggleFrame(self.main, text='Set Training Parameters: ', padx=6, row=4) 
        self.fam_switch  = ttl.Switch(self.param_frame, text='Familiar Training :', row=0, col=1)
        self.save_switch = ttl.Switch(self.param_frame, text='Model Saving:', row=1, col=1)
        self.stop_switch = ttl.Switch(self.param_frame, text='Early Stopping: ',    row=2, col=1)
        self.trim_switch = ttl.Switch(self.param_frame, text='RIP Trimming: ',      row=3, col=1)

        self.cycle_fams   = tk.BooleanVar()
        self.cycle_button = tk.Checkbutton(self.param_frame, text='Cycle?', variable=self.cycle_fams)
        self.cycle_button.grid(row=0, column=3)
        
        self.upper_bound_entry      = ttl.LabelledEntry(self.param_frame, text='Upper Bound:', var=tk.IntVar(),      default=400, row=4, col=0)
        self.slice_decrement_entry  = ttl.LabelledEntry(self.param_frame, text='Slice Decrement:', var=tk.IntVar(),  default=20, row=4, col=2)
        self.lower_bound_entry      = ttl.LabelledEntry(self.param_frame, text='Lower Bound:', var=tk.IntVar(),      default=50, row=5, col=0)
        self.n_slice_entry          = ttl.LabelledEntry(self.param_frame, text='Number of Slices:', var=tk.IntVar(), default=1, row=5, col=2)
        self.trim_switch.dependents = (self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry, self.n_slice_entry)
        
        self.confirm_and_preset     = tk.Button(self.param_frame, text='Confirm and Save Preset', padx=12, command=self.confirm_tparams_and_preset)
        self.confirm_train_params   = ttl.ConfirmButton(self.param_frame, command=self.confirm_tparams, row=6, col=2, cs=2)
        
        self.confirm_and_preset.grid(row=6, column=0, columnspan=2, sticky='w')

        
        # Frame 5 - contains only the button used to trigger a main action  
        self.activation_frame = ttl.ToggleFrame(self.main, text='', padx=0, pady=0, row=5)   
        self.train_button = tk.Button(self.activation_frame, text='TRAIN', padx=22, width=44, bg='deepskyblue2', state='disabled', command=self.training)
        self.train_button.grid(row=0, column=0)
        self.species_summaries = []
        
        self.pred_button = tk.Button(self.activation_frame, text='PREDICT', padx=22, width=44, bg='mediumorchid2', state='disabled', command=lambda:None)
        self.pred_button.grid(row=0, column=0)
        self.switch_tpmode()
        
        
        # Packaging together some widgets and attributes, for ease of reference (also useful for self.reset() and self.isolate() methods)
        self.arrays   = (self.parameters, self.chem_data, self.species, self.families, self.family_mapping, self.species_summaries) 
        self.frames   = (self.input_frame, self.file_frame, self.selection_frame, self.hyperparam_frame, self.param_frame, self.activation_frame)
        self.switches = (self.fam_switch, self.save_switch, self.stop_switch, self.trim_switch)
        self.entries  = (self.epoch_entry, self.batchsize_entry, self.learnrate_entry, self.n_slice_entry, self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry)
        self.resetable_twidgets = (*self.entries, self.json_menu, self.read_status) # all of my custom widgets which have a "reset_default()" method built in
        
        self.reset() # isolate the first frame by default
        
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
    
    def shutdown(self):
        '''Close the application, with confirm prompt'''
        if messagebox.askokcancel('Exit', 'Are you sure you want to close?'):
            self.main.destroy()
    
    def switch_tpmode(self):
        '''Used to switch the mode of the training button; planned future feature, WIP at the moment'''
        target_button = self.tpmode.get() and self.pred_button or self.train_button
        target_button.tkraise()
    
    def reset(self):
        '''reset the GUI to the opening state, along with all variables'''
        self.isolate(self.input_frame)
        
        self.read_mode.set(None)
        self.input_mode.set(None)
        self.cycle_fams.set(False)
        self.spectrum_size = 0
        
        for array in self.arrays:
            array.clear()
        
        for switch in self.switches:
            switch.disable()
        
        for twidget in self.resetable_twidgets: # Tim Widgets, or just twidgets for short :)
            twidget.reset_default()
        
        self.lift() # bring main window to forefront

        
    # Frame 0 (Parameter Input) Methods   
    def confirm_input_mode(self):
        '''Confirm the method of inputting training parameters'''
        if self.input_mode.get() == 'Manual Input': # if manual mode is chosen, proceed through parameter setting
            self.isolate(self.file_frame)     
        elif self.input_mode.get() == 'Preset from File': # if present mode is chosen, perform all parameter-setting in a single step
            preset_path = Path(filedialog.askopenfilename(title='Choose Training Preset', initialdir='./Training Presets', filetypes=(('JSONs', '*.json'),) ))
            try: # NOTE: scope looks a bit odd but is correct, do not change indentation of try/excepts
                with open(preset_path, 'r') as preset_file:
                    self.parameters = json.load(preset_file) # load in the parameter preset (ALL settings)
                try:
                    # reading the present datafile, setting the file inputs fields accordingly
                    data_file = Path(self.parameters['data_file'])
                    self.chosen_file.set(data_file.name) 
                    self.read_data_file(data_file)

                    # making the evaluand selection
                    self.read_mode.set('By Species')
                    self.selections = self.parameters['selections']

                    # filling in the hyperparameter entries
                    self.epoch_entry.set_value(self.parameters['num_epochs'])
                    self.batchsize_entry.set_value(self.parameters['batchsize'])
                    self.learnrate_entry.set_value(self.parameters['learnrate'])

                    # setting the switches appropriately
                    self.cycle_fams.set(self.parameters['cycle_fams'])
                    self.fam_switch.apply_state( self.parameters['fam_training'])  
                    self.save_switch.apply_state(self.parameters['save_weights'])
                    self.stop_switch.apply_state(self.parameters['early_stopping'])
                    self.trim_switch.apply_state(self.parameters['trim_spectra'])

                    # filling in the slicing entry values
                    self.upper_bound_entry.set_value(self.parameters['trimming_max']) 
                    self.slice_decrement_entry.set_value(self.parameters['slice_decrement'])
                    self.lower_bound_entry.set_value(self.parameters['trimming_min'])     
                    self.n_slice_entry.set_value(self.parameters['num_slices'])   
                    self.check_trimming_bounds() # ensure that bounds actually make sense         
                    
                    self.isolate(self.activation_frame) # skip past manual input and move straight to the training button 
                except KeyError as error: # gracefully handle the case when the preset does not contain the correct names
                    messagebox.showerror('Preset Error', f'The parameter "{error}" is either missing or misnamed\n Please check preset file for errors')    
            except PermissionError: # do nothing if user cancels selection
                self.input_mode.set(None)      
        else:
            messagebox.showerror('No input mode selected!', 'Please choose either "Manual" or "Preset" input mode')
    
    
    # Frame 1 (Reading) Methods 
    def read_data_file(self, data_file):
        '''Read the contents of the chosen data file'''
        with open(data_file, 'r') as file:  # read in file data, discard counts for the time being
            json_data = json.load(file)
        self.chem_data = [iumsutils.Instance(*properties) for properties in json_data['chem_data']] # unpack data into Instance objects
        self.species = json_data['species']
        self.families = json_data['families']
        self.family_mapping = json_data['family_mapping']
        self.spectrum_size = json_data['spectrum_size']
        self.read_status.set_status(True)
    
    def import_data(self):
        '''Confirm the file chosen, read in the file, and proceed'''
        data_file = Path(self.chosen_file.get())
        if data_file == '--Choose a JSON--':
            messagebox.showerror('File Error', 'No JSON selected')
        else:
            self.parameters['data_file'] = data_file.name
            self.read_data_file(data_file)       
            self.upper_bound_entry.set_value(self.spectrum_size) # adjust the slicing upper bound to the size of spectra passed
            self.isolate(self.selection_frame)
    
    
    # Frame 2 (Evaluand Selection) Methods
    def further_sel(self): 
        '''logic for selection of evaluands to include in training, based on the chosen selection mode'''
        self.parameters['selections'] = []
        if self.read_mode.get() == 'Select All':
            self.selections = self.species
        elif self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.selection_frame, '950x190', self.species, self.parameters['selections'], window_title='Select Species to evaluate', ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.selection_frame, '270x85', self.families, self.parameters['selections'], window_title='Select Families to evaluate over', ncols=3)

    def confirm_selections(self):
        '''Confirm the species selected for evaluation and proceed'''
        if not self.parameters.get('selections'): # if the selections parameter is either empty or doesn't exist
            messagebox.showerror('No selections made', 'Please select species to evaluate')
        else:
            if self.read_mode.get() == 'By Family':  # pick out species by family if selection by family is made
                self.parameters['selections'] = [species for species in self.species if iumsutils.get_family(species) in self.parameters['selections']]
            self.isolate(self.hyperparam_frame)

    
    # Frame 3 (hyperparameter) Methods
    def confirm_hp(self):
        '''Confirm the selected hyperparameters and proceed'''
        self.parameters['num_epochs'] = self.epoch_entry.get_value()
        self.parameters['batchsize'] = self.batchsize_entry.get_value()
        self.parameters['learnrate'] = self.learnrate_entry.get_value()  
        self.isolate(self.param_frame)
        self.trim_switch.disable()   # ensures that the trimming menu stays greyed out, not necessary for the other switches 
    
    
    # Frame 4 (training parameter) Methods
    def check_trimming_bounds(self):
        '''Performs error check over the trimming bounds held internally to ensure that sensible bounds are chosen to ensure that training will not fail'''
        trimming_min, trimming_max = self.parameters['trimming_min'], self.parameters['trimming_max'] # have pulled these out solely for readability
        num_slices, slice_decrement =self.parameters['num_slices'], self.parameters['slice_decrement']
        
        if trimming_min < 0 or type(trimming_min) != int: 
            messagebox.showerror('Boundary Value Error', 'Trimming Min must be a positive integer')
        elif trimming_max < 0 or type(trimming_max) != int:
            messagebox.showerror('Boundary Value Error', 'Trimming Max must be a positive integer')
        elif trimming_max > self.spectrum_size:                            
            messagebox.showerror('Limit too high', 'Upper bound greater than data size, please decrease "Upper Bound"')
        elif trimming_max - num_slices*slice_decrement <= trimming_min:  
            messagebox.showerror('Boundary Mismatch', 'Upper limit will not always exceed lower;\nDecrease either slice decrement or lower bound')
    
    def confirm_tparams(self):
        '''Confirm training parameter selections, perform pre-training error checks if necessary'''
        self.parameters['fam_training']   = self.fam_switch.value
        self.parameters['cycle_fams']     = self.cycle_fams.get()
        self.parameters['save_weights']   = self.save_switch.value
        self.parameters['early_stopping'] = self.stop_switch.value
        
        self.parameters['trim_spectra']    = self.trim_switch.value        
        self.parameters['trimming_max']    = self.upper_bound_entry.get_value()
        self.parameters['slice_decrement'] = self.slice_decrement_entry.get_value()
        self.parameters['trimming_min']    = self.lower_bound_entry.get_value()
        self.parameters['num_slices']      = self.n_slice_entry.get_value()
        
        if self.trim_switch.value: # only perform error checks if trimming is enabled 
            self.check_trimming_bounds()
            
        self.isolate(self.activation_frame) # make the training button clickable
        
    def confirm_tparams_and_preset(self):
        '''Confirm training parameters in manual mode, as well as saving the configured training preset to the central preset folder'''
        self.confirm_tparams()
        preset_path = Path(filedialog.asksaveasfilename(title='Save Preset to file', initialdir='./Training Presets', defaultextension='.json', filetypes=[('JSONs', '*.json')] ))
        try: 
            with open(preset_path, 'w') as preset_file:
                json.dump(self.parameters, preset_file)
        except PermissionError: # catch the case in which the user cancels the preset saving
            self.isolate(self.param_frame) # re-enable the frame and allow the user to try again
        
    # Section 2A: the training routine itself, along with a dedicated reset method to avoid overwrites between subsequent trainings
    def training(self, num_spectra=2, test_set_proportion=0.2, verbosity=False):
        '''The neural net training function itself; this is where the fun happens'''
        # UNPACKING INTERNAL PARAMETERS FROM DICT INTO MORE EASILY REFERENCEABLE VARIABLES
        selections      = self.parameters['selections']
        data_file_name  = Path(self.parameters['data_file']).stem
        
        num_epochs      = self.parameters['num_epochs'] 
        batchsize       = self.parameters['batchsize']
        learnrate       = self.parameters['learnrate'] 
        
        trimming_max    = self.parameters['trimming_max'] 
        slice_decrement = self.parameters['slice_decrement'] 
        trimming_min    = self.parameters['trimming_min'] 
        num_slices      = self.parameters['num_slices'] 
        
        cycle_fams      = self.parameters['cycle_fams']
        fam_training    = self.parameters['fam_training']
        trim_spectra    = self.parameters['trim_spectra']
        save_weights    = self.parameters['save_weights']
        early_stopping  = self.parameters['early_stopping']      
        
        iumsutils.SpeciesSummary.family_mapping = self.family_mapping # MUST assign the current mapping to the species summary class
        total_rounds = len(selections) # determining the total number of training rounds which will be performed
        if trim_spectra:
            total_rounds *= (1 + num_slices)
        if cycle_fams:  
            total_rounds *= 2
            
        # PRE-TRAINING PREPARATIONS TO ENSURE INTERFACE CONTINUITY
        self.activation_frame.disable() # disable training button while training (an idiot-proofing measure)
        train_window = TrainingWindow(self.main, total_rounds, num_epochs, self.training, self.reset, self.shutdown)
        
        keras_callbacks = [TkEpochs(train_window)] # in every case, add the tkinter-keras interface callback to the callbacks list
        if early_stopping:
            early_stopping_callback = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=16, restore_best_weights=True) # fine-tune patience, eventually
            keras_callbacks.append(early_stopping_callback) # add early stopping callback if called for
        
        # FILE MANAGEMENT AND OVERWRITE CHECKS, ENSURES NO DATA GETS ACCIDENTALLY OVERWRITTEN AND TAKEN BURDEN OF ORGANIZATION OFF OF THE USER
        train_window.set_status('Creating Folders...') 
        results_folder = Path('Saved Training Results', f'{data_file_name} Results') # main folder with all results for a particular data file
        
        parent_folder, file_id = results_folder/f'{num_epochs}-epoch', 0
        while parent_folder.exists(): # logic to take care of numbering in case of file duplication
            try: # determine, from existing preset, if the old training is identical to the current one
                with open(parent_folder/'Training Preset.json', 'r') as existing_preset_file: # consider adding a case for when no preset exists
                    existing_preset = json.load(existing_preset_file)
                    prompt_overwrite = (existing_preset == self.parameters) # ask to overwrite only if training is identical
            except FileNotFoundError: # if no preset exists, for whatever reason, 
                prompt_overwrite = True
            
            if prompt_overwrite: 
                if messagebox.askyesno('Overwrite Prior Results?', 'Folder with same (or no) training parameters found;\nOverwrite old folder?', parent=train_window.training_window): 
                    shutil.rmtree(parent_folder, ignore_errors=True)
                    break # if a folder with the same preset is found and overwrite permission is given, delete the folder then exit the loop to create the folder
                else:
                    train_window.destroy()
                    self.activation_frame.enable() 
                    return  #terminate prematurely if overwrite permission is not given, reset to pre-training menu state
            else:
                file_id += 1
                parent_folder = Path(results_folder/f'{num_epochs}-epoch{bool(file_id)*f"({file_id})"}')
     
        try:
            parent_folder.mkdir(parents=True) # make the parent folder for this training session
            with open(parent_folder/'Training Preset.json', 'w') as preset_file: 
                json.dump(self.parameters, preset_file)   # save current training settings to a preset for reproducability
        except (FileExistsError, PermissionError): # catch the exception wherein mkdir fails because the folder is inadvertently open
            messagebox.showerror('Permission Error', f'{parent_folder} \nor one of its subfolders is open, cannot overwrite old files! \
                                \n\nPlease press "Retrain" to try again', parent=train_window.training_window)
            train_window.button_frame.enable()
            return
        
        # OUTERMOST TRAINING LOOP, REGULATES CYCLING BETWEEN FAMILIARS AND UNFAMILIARS
        for familiar_cycling in range(1 + int(self.cycle_fams.get())):  # for now, it is required that this is a "get()" due to toggling    
            start_time = time()    # log start time of each cycle, so that training duration can be computed afterwards  
            if familiar_cycling:
                fam_training = not fam_training # invert familiar state upon cycling (only locally, though, not in the parameters or preset)
                self.fam_switch.toggle()  # toggle the unfamiliar status the second time through (only occurs if cycling is enabled)
            
            fam_str = f'{fam_training and "F" or "Unf"}amiliar'  # some str formatting based on whether the current training type is familiar or unfamiliar
            train_window.set_familiar_status(fam_str)
            
            curr_fam_folder = parent_folder/fam_str # folder with the results from the current familiar training, parent for all that follow 
            if not curr_fam_folder.exists(): # if the folder doesn't exists, make it
                curr_fam_folder.mkdir() 
                
                log_file_path = curr_fam_folder/'Log File.txt'
                log_file_path.touch()   # create the log file, not yet logging anything
            
            # iNTERMEDIARY TRAINING LOOP, REGULATES SLICING
            for segment in range(1 + (trim_spectra and num_slices or 0)): # by default only train OVER full spectrum. If trimming is enabled, train an extra <number of slices> times
                if segment == 0:  # first segment will always be over the full spectrum, regardless of trimming
                    lower_bound, upper_bound = 0, self.spectrum_size
                    point_range = 'Full Spectra'
                else:
                    lower_bound, upper_bound = trimming_min, trimming_max - slice_decrement*(segment - 1) # trimming segments start at segment == 1, hence the -1
                    point_range = f'Points {lower_bound}-{upper_bound}'
                train_window.set_slice(point_range)
                
                self.species_summaries.clear() # empty the summaries list with each slice
                curr_slice_folder = curr_fam_folder/point_range
                if not curr_slice_folder.exists():                                                           
                    curr_slice_folder.mkdir(parents=True) # ensure the folder exists, obviously
                
                # INNERMOST TRAINING LOOP, CYCLES THROUGH ALL THE SELECTED EVALUANDS
                for evaluand_idx, evaluand in enumerate(selections):                  
                    train_window.set_status('Training...')
                    train_window.round_progress.increment() 
                    curr_family = iumsutils.get_family(evaluand)
 
                    eval_data = [(instance.name, instance.spectrum[lower_bound:upper_bound]) for instance in self.chem_data if instance.species == evaluand]
                    eval_titles, eval_spectra = map(list, zip(*eval_data)) # unpack the instance names and spectra via zip, convert zip tuples to lists to allow for insertion
                    eval_set_size = len(eval_data)
                    train_window.set_evaluand(f'{evaluand} ({eval_set_size} instances found)')

                    # TRAIN/TEST SET CREATION, MODEL CREATION, AND MODEL TRAINING
                    if not fam_training or evaluand_idx == 0: # for familiars, model is only trained during first evaluand at each slice (since training set is identical throughout)
                        training_data = zip( *((instance.spectrum[lower_bound:upper_bound], instance.vector) for instance in self.chem_data
                                                                                           if instance.species != evaluand or fam_training))               
                        x_train, x_test, y_train, y_test = map(np.array, train_test_split(*training_data, test_size=test_set_proportion)) # keras model only accepts numpy arrays
                        
                        with tf.device('CPU:0'):     # eschews the requirement for a brand-new NVIDIA graphics card (which we don't have anyways)                      
                            model = Sequential()     # model block is created, layers are added to this block
                            model.add(Dense(512, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer, size depends on trimming
                            model.add(Dropout(0.5))                                    # dropout layer, to reduce overfit
                            model.add(Dense(512, activation='relu'))                   # 512 neuron hidden layer
                            model.add(Dense(len(self.families), activation='softmax')) # softmax gives probability distribution of identity over all families
                            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learnrate), metrics=['accuracy']) 
                        
                        if verbosity: # optional printout to console of some useful information about the model and data
                            print(f'\'{len(training_data)} total instances in training/test sets;')
                            for x, y, group in ((x_train, y_train, "training"), (x_test, y_test, "test")):
                                print(f'{x.shape[0]} features & {y.shape[0]} labels in {group} set ({round(100*x.shape[0]/len(training_data), 2)}% of the data)')

                        # actual model training occurs here
                        verbose = (verbosity and 2 or 0) # translate the passed "verbosity" argument into Keras verbosity
                        hist = model.fit(x_train, y_train, callbacks=keras_callbacks, verbose=verbose, epochs=num_epochs, batch_size=batchsize)
                        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verbose)  # keras' self evaluation of loss and accuracy metric
                        
                        if early_stopping and early_stopping_callback.stopped_epoch: # if early stopping has indeed been triggered:
                            with open(log_file_path, 'a') as log_file: # if early stopping has occurred, log it
                                log_file.write(f'{fam_training and fam_str or evaluand} training round stopped early at epoch {early_stopping_callback.stopped_epoch}\n')
                        
                        if save_weights and not train_window.end_training: # if saving is enabled, save the model to the current result directory
                            train_window.set_status('Saving Model...')
                            weights_folder = curr_slice_folder/f'{fam_training and fam_str or evaluand} Model Files'
                            model.save(str(weights_folder)) # path can only be str, for some reason
                            
                            # add a preset to each model file that allows for reproduction of ONLY that single model file
                            local_preset = {param : value for param, value in self.parameters.items()} # make a deep copy of the parameters that can be modified only locally
                            local_preset['cycle_fams'] = False 
                            local_preset['fam_training'] = fam_training
                            if not fam_training:
                                local_preset['selections'] = [evaluand] # only use single evaluand for unfamiliars (full set for familiars)
                            local_preset['trimming_max'] = upper_bound 
                            local_preset['trimming_min'] = lower_bound 
                            local_preset['num_slices'] = 1
                            
                            with open(weights_folder/f'Reproducability Preset ({point_range}).json', 'w') as weights_preset: 
                                json.dump(local_preset, weights_preset) # add a preset just for the current                        
                    
                    # ABORTION CHECK PRIOR TO WRITING FILES, WILL CEASE IF TRAINING IS ABORTED
                    if train_window.end_training:  
                        messagebox.showerror('Training has Stopped!', 'Training aborted by user;\nProceed from Progress Window', parent=train_window.training_window)
                        train_window.button_frame.enable()
                        return # without a return, aborting training only pauses one iteration of loop
                    
                    # PACKAGING OF EXTRANEOUS PLOTS
                    spectra_plots = [iumsutils.BundledPlot(data=spectrum, title=name, x_data=range(lower_bound, upper_bound)) for name, spectrum in eval_data[:num_spectra]]
                    loss_plot     = iumsutils.BundledPlot(data=hist.history['loss'], title=f'Training Loss (Final = {test_loss:0.2f})', plot_type='m') 
                    accuracy_plot = iumsutils.BundledPlot(data=hist.history['accuracy'], title=f'Training Accuracy (Final = {100*test_acc:0.2f})', plot_type='m') 
                    extra_plots   = [*spectra_plots, loss_plot, accuracy_plot]
                    
                    # CREATION OF THE SPECIES SUMMARY OBJECT FOR THE CURRENT EVALUAND, PLOTTING OF EVALUATION RESULTS FOR THE CURRENT EVALUAND                
                    curr_spec_sum = iumsutils.SpeciesSummary(evaluand) # SpeciesSummary objects handle calculation and plotting of individual evaluand results
                    curr_spec_sum.add_all_insts(eval_titles, model.predict(np.array(eval_spectra))) # channel the instance names and predictions into the summary and process them
                    
                    train_window.set_status('Plotting Results to Folders...')
                    curr_spec_sum.graph(curr_slice_folder/evaluand, prepended_plots=extra_plots) # results are also processed before graphing (two birds with one stone)
                    self.species_summaries.append(curr_spec_sum) # collect all the species summaries for this slice, unpack them after the end of the slice

                # DISTRIBUTION OF SUMMARY DATA TO APPROPRIATE RESPECTIVE FOLDERS 
                train_window.set_status(f'Distributing Results for {point_range}, {fam_str}...')
                iumsutils.unpack_summaries(self.species_summaries, save_dir=curr_slice_folder, indicator=fam_str)
            
            with open(log_file_path, 'a') as log_file:  # log the time taken to complete the training cycle (open in append mode to prevent overwriting)
                model.summary(print_fn=lambda x : log_file.write(f'{x}\n')) # write model to log file (since model is same throughout, should only write model on the last pass)
                log_file.write(f'\nTraining Time : {iumsutils.format_time(time() - start_time)}')   
        
        # POST-TRAINING WRAPPING-UP
        train_window.button_frame.enable()  # open up post-training options in the training window
        train_window.abort_button.configure(state='disabled') # disable the abort button (idiotproofing against its use after training)
        train_window.set_status('Finished')
        messagebox.showinfo('Training Completed Succesfully!', f'Training results can be found in folder:\n{parent_folder}', parent=train_window.training_window)
        
if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
