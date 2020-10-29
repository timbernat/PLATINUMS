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
    '''A custom keras Callback which interfaces between the training window and the Keras model training session'''
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
    '''The window which displays training progress and information, subclassed TopLevel allows it to be separate from the main GUI'''
    def __init__(self, main, train_funct, reset_main_funct, exit_funct):
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
        self.round_progress = ttl.NumberedProgBar(self.status_frame, row=3, col=1)
        self.epoch_label    = tk.Label(self.status_frame, text='Training Epoch: ')
        self.epoch_progress = ttl.NumberedProgBar(self.status_frame, style_num=2, row=4, col=1) 
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
        self.button_frame      = ttl.ToggleFrame(self.training_window, text='', padx=0, pady=0, row=1)
        self.retrain_button    = tk.Button(self.button_frame, text='Retrain', width=17, bg='deepskyblue2', command=self.retrain)
        self.reset_main_button = tk.Button(self.button_frame, text='Reset', width=17, bg='orange', command=self.reset_main)
        self.exit_button       = tk.Button(self.button_frame, text='Exit', width=17, bg='red', command=exit_funct)
        
        self.retrain_button.grid(   row=0, column=0)
        self.reset_main_button.grid(row=0, column=1)
        self.exit_button.grid(      row=0, column=2) 
        
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
    '''PLATINUMS : Prediction, Training, And Labelling INterface for Unlabelled Mobility Spectra'''
    def __init__(self, main):       
    #Main Window
        self.main = main
        self.main.title('PLATINUMS 4.7.5-alpha')
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
        
        self.data_path = Path('TDMS Datasets') # this is the folder where PLATINUMS will check and read data files
        self.data_path.mkdir(exist_ok=True)  # will make this directory if it doesn't already exist
            
        self.json_menu      = ttl.DynOptionMenu(self.file_frame, var=self.chosen_file, option_method=iumsutils.get_by_filetype,
                                                opargs=('.json', self.data_path), default='--Choose a JSON--', width=32, colspan=2)
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
        self.save_switch = ttl.Switch(self.param_frame, text='Model Saving:',       row=1, col=1)
        self.stop_switch = ttl.Switch(self.param_frame, text='Early Stopping: ',    row=2, col=1)
        self.trim_switch = ttl.Switch(self.param_frame, text='RIP Trimming: ',      row=3, col=1)

        self.cycle_fams   = tk.BooleanVar()
        self.cycle_button = tk.Checkbutton(self.param_frame, text='Cycle?', variable=self.cycle_fams)
        self.cycle_button.grid(row=0, column=3)
        
        self.upper_bound_entry      = ttl.LabelledEntry(self.param_frame, text='Upper Bound:',    var=tk.IntVar(), default=400, row=4, col=0)
        self.slice_decrement_entry  = ttl.LabelledEntry(self.param_frame, text='Slice Decrement:', var=tk.IntVar(), default=20, row=4, col=2)
        self.lower_bound_entry      = ttl.LabelledEntry(self.param_frame, text='Lower Bound:',     var=tk.IntVar(), default=50, row=5, col=0)
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
        self.twidgets = (*self.entries, self.json_menu, self.read_status) # all of my custom widgets WITH a "reset_default()" method built in
        
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
        for tk_switch_var in (self.read_mode, self.input_mode, self.cycle_fams):
            tk_switch_var.set(0)
        
        for array in self.arrays:
            array.clear()
        
        for switch in self.switches:
            switch.disable()
        
        for twidget in self.twidgets: # Tim Widgets, or just twidgets for short :)
            twidget.reset_default()
        
        self.lift() # bring main window to forefront
        self.isolate(self.input_frame)
          
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
                    self.chosen_file.set(self.parameters['data_file']) 
                    self.read_data_file()

                    # making the evaluand selection
                    self.read_mode.set(self.parameters['read_mode'])
                    self.confirm_selections() # handles the logic of assigning 

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
                    messagebox.showerror('Preset Error', f'The parameter "{error}" is either missing or misnamed;\n Please check preset file for errors')    
            except PermissionError: # do nothing if user cancels selection
                self.input_mode.set(None)      
        else:
            messagebox.showerror('No input mode selected!', 'Please choose either "Manual" or "Preset" input mode')
    
    # Frame 1 (Reading) Methods 
    def read_data_file(self):
        '''Read the contents of the chosen data file'''
        with open(self.data_path/self.chosen_file.get(), 'r') as file:  # read in file data, discard counts for the time being
            json_data = json.load(file)
        self.chem_data = [iumsutils.Instance(*properties) for properties in json_data['chem_data']] # unpack data into Instance objects
        self.species = json_data['species']
        self.families = json_data['families']
        self.family_mapping = json_data['family_mapping']
        self.spectrum_size = json_data['spectrum_size']
        self.read_status.set_status(True)
    
    def import_data(self):
        '''Confirm the file chosen, read in the file, and proceed'''
        if self.chosen_file.get() == '--Choose a JSON--':
            messagebox.showerror('File Error', 'No JSON selected')
        else:
            self.parameters['data_file'] = self.chosen_file.get()
            self.read_data_file()       
            self.upper_bound_entry.set_value(self.spectrum_size) # adjust the slicing upper bound to the size of spectra passed
            self.isolate(self.selection_frame)
    
    # Frame 2 (Evaluand Selection) Methods
    def further_sel(self): 
        '''logic for selection of evaluands to include in training, based on the chosen selection mode'''
        self.parameters['selections'] = [] # empties selections when selection menu is re-clicked (ensures no overwriting or mixing)
        self.parameters['read_mode'] = self.read_mode.get() # record the read mode to parameters with each selection
        
        if self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.selection_frame, '950x190', self.species, self.parameters['selections'], window_title='Select Species to evaluate', ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.selection_frame, '270x85', self.families, self.parameters['selections'], window_title='Select Families to evaluate over', ncols=3)
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
        
        
    # Section 2A: the training routine itself, this is where the magic happens
    def training(self, num_spectra=2, test_set_proportion=0.2, verbosity=False):
        '''The function defining the neural network training sequence and parameters'''
        # UNPACKING INTERNAL PARAMETERS DICT AND INITIALIZING SOME VALUES
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
        
        iumsutils.SpeciesSummary.family_mapping = self.family_mapping # use the current mapping for producing summaries later
        
        # FILE MANAGEMENT CODE, GUARANTEES THAT NO ACCIDENTAL OVERWRITES OCCUR AND THAT AN APPRORIATE EMPTY FOLDER EXISTS TO WRITE RESULTS TO
        results_folder = Path('Saved Training Results', f'{data_file_name} Results') # main folder with all results for a particular data file        
        parent_name = f'{num_epochs}-epoch, {cycle_fams and "cycled" or (fam_training and "familiar" or "unfamiliar")}' # name of folder with results for this training session
        parent_folder = results_folder/parent_name # same name applied to a Path object, useful for incrementing file_id during ensuing overwrite checking
        
        file_id = 0
        while parent_folder: 
            try: # if no file exists, can make a new one and proceed without issue
                parent_folder.mkdir(exist_ok=False) 
                break
            except FileExistsError: # if one DOES exist, enter checking logic
                if not any(parent_folder.iterdir()): # if the file exists but is empty, can also proceed with no issues
                    break 
                elif (parent_folder/'Training Preset.json').exists(): # otherwise, if the folder has a preset present...
                    with open(parent_folder/'Training Preset.json', 'r') as existing_preset_file: 
                        existing_preset = json.load(existing_preset_file)

                    if existing_preset != self.parameters: # ...and if the preset is different to the current preset, start the checks over with incremented name
                        file_id += 1
                        parent_folder = Path(results_folder/f'{parent_name}({file_id})') # appends number to end of duplicated files (similar to Windows file management)
                        continue

                # this branch is only executed if no preset file is found OR (implicitly) if the existing preset matches the one found
                if messagebox.askyesno('Request Overwrite', 'Folder with same name and same (or no) training parameters found; Allow overwrite of folder?'):
                    try:
                        iumsutils.clear_folder(parent_folder) # empty the folder if permission is given
                    except PermissionError: # if emptying fails because the user forgot to close files, notify them and quit
                        messagebox.showerror('Overwrite Error!', f'{parent_folder}\nhas file(s) open and cannot be overwritten;\n\nPlease close all files and try training again')
                        self.activation_frame.enable() # particular to the case where "retrain" button is used instead
                        return
             
        with open(parent_folder/'Training Preset.json', 'w') as preset_file: # save current training settings to a preset for reproducability
            json.dump(self.parameters, preset_file)   
        
        self.activation_frame.disable() # disable training button while training (an idiot-proofing measure)                 
        train_window = TrainingWindow(self.main, train_funct=self.training, reset_main_funct=self.reset, exit_funct=self.shutdown)
        train_window.set_status('Initializing Training...') # create the training window UI object and begin preparing training
        train_window.round_progress.set_max(len(selections)*(1 + trim_spectra*num_slices)*(1 + cycle_fams)) # compute and display the number of training rounds to occur

        keras_callbacks = [TkEpochs(train_window)] # in every case, add the tkinter-keras interface callback to the callbacks list
        if early_stopping:
            early_stopping_callback = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=16, restore_best_weights=True) # fine-tune patience, eventually
            keras_callbacks.append(early_stopping_callback) # add early stopping callback if called for
        
        # OUTERMOST TRAINING LOOP, REGULATES CYCLING BETWEEN FAMILIARS AND UNFAMILIARS
        for familiar_cycling in range(1 + cycle_fams):    
            start_time = time()    # log start time of each cycle, so that training duration can be computed afterwards  
            if familiar_cycling:
                fam_training = not fam_training # invert familiar state upon cycling (only locally, though, not in the parameters or preset)
                self.fam_switch.toggle()  # cosmetic, but allows the user to see that cycling is in fact occurring
            
            fam_str = f'{fam_training and "F" or "Unf"}amiliar'  # some str formatting based on whether the current training type is familiar or unfamiliar
            train_window.set_familiar_status(fam_str)
            
            curr_fam_folder = parent_folder/fam_str # folder with the results from the current familiar training, parent for all that follow 
            curr_fam_folder.mkdir(exist_ok=True)
                
            log_file_path = curr_fam_folder/'Log File.txt'
            log_file_path.touch()   # create the log file, not yet logging anything
            
            # iNTERMEDIARY TRAINING LOOP, REGULATES SLICING
            for segment in range(1 + trim_spectra*num_slices): # by default only train OVER full spectrum. If trimming is enabled, train an extra <number of slices> times
                if segment == 0:  # first segment will always be over the full spectrum, regardless of trimming
                    lower_bound, upper_bound = 0, self.spectrum_size
                    point_range = 'Full Spectra'
                else:
                    lower_bound, upper_bound = trimming_min, trimming_max - slice_decrement*(segment - 1) # trimming segments start at segment == 1, hence the -1
                    point_range = f'Points {lower_bound}-{upper_bound}'
                train_window.set_slice(point_range)
                
                self.species_summaries.clear() # empty the summaries list with each slice
                curr_slice_folder = curr_fam_folder/point_range
                curr_slice_folder.mkdir(exist_ok=True)
                
                # INNERMOST TRAINING LOOP, CYCLES THROUGH ALL THE SELECTED EVALUANDS
                for evaluand_idx, evaluand in enumerate(selections):                  
                    train_window.set_status('Training...')
                    train_window.round_progress.increment() 
                    train_window.epoch_progress.set_max(num_epochs) # reset maximum number of epochs on progress bar to specified amount (done each round to handle early stopping)
                    
                    curr_family = iumsutils.get_family(evaluand)
                    eval_data = [(instance.name, instance.spectrum[lower_bound:upper_bound]) for instance in self.chem_data if instance.species == evaluand]
                    eval_titles, eval_spectra = map(list, zip(*eval_data)) # unpack the instance names and spectra via zip, convert zip tuples to lists to allow for insertion
                    eval_set_size = len(eval_data)
                    train_window.set_evaluand(f'{evaluand} ({eval_set_size} instances found)')

                    # TRAIN/TEST SET CREATION, MODEL CREATION, AND MODEL TRAINING
                    if not fam_training or evaluand_idx == 0: # for familiars, model is only trained during first evaluand at each slice (since training set is identical throughout)
                        features = [instance.spectrum[lower_bound:upper_bound] for instance in self.chem_data if instance.species != evaluand or fam_training]
                        labels   = [instance.vector for instance in self.chem_data if instance.species != evaluand or fam_training]
                        x_train, x_test, y_train, y_test = map(np.array, train_test_split(features, labels, test_size=test_set_proportion)) # keras model only accepts numpy arrays
                        
                        with tf.device('CPU:0'):     # eschews the requirement for a brand-new NVIDIA graphics card (which we don't have anyways)                      
                            model = Sequential()     # model block is created, layers are added to this block
                            model.add(Dense(256, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer, size depends on trimming
                            model.add(Dropout(0.5))                                    # dropout layer, to reduce overfit
                            model.add(Dense(256, activation='relu'))                   # 512 neuron hidden layer
                            model.add(Dense(len(self.families), activation='softmax')) # softmax gives probability distribution of identity over all families
                            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learnrate), metrics=['accuracy']) 

                        # actual model training occurs here
                        verbose = (verbosity and 2 or 0) # translate the passed "verbosity" argument into Keras verbosity
                        hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=keras_callbacks, verbose=verbose, epochs=num_epochs, batch_size=batchsize)
                        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verbose)  # keras' self evaluation of loss and accuracy metric
                        self.hist = hist
                        
                        if early_stopping and early_stopping_callback.stopped_epoch: # if early stopping has indeed been triggered:
                            train_window.epoch_progress.set_max(early_stopping_callback.stopped_epoch + 1) # set progress max to the stopped epoch, to indicate that training is done
                            
                            with open(log_file_path, 'a') as log_file: # if early stopping has occurred, log it
                                log_file.write(f'{fam_training and fam_str or evaluand} training round stopped early at epoch {early_stopping_callback.stopped_epoch + 1}\n')
                        
                        if save_weights and not train_window.end_training: # if saving is enabled, save the model to the current result directory
                            train_window.set_status('Saving Model...')
                            weights_folder = curr_slice_folder/f'{fam_training and fam_str or evaluand} Model Files'
                            model.save(str(weights_folder)) # path can only be str, for some reason
                            
                            # add a preset to each model file that allows for reproduction of ONLY that single model file
                            local_preset = {param : value for param, value in self.parameters.items()} # make a deep copy of the parameters that can be modified only locally
                            local_preset['cycle_fams']   = False 
                            local_preset['selections']   = [evaluand] # only requires a single evaluand (in the case of familiars, the particular one is not at all relevant)
                            local_preset['trimming_max'] = upper_bound # should revise these for the future
                            local_preset['trimming_min'] = lower_bound 
                            local_preset['num_slices']   = 1
                            
                            with open(weights_folder/f'Reproducability Preset ({point_range}).json', 'w') as weights_preset: 
                                json.dump(local_preset, weights_preset) # add a preset just for the current                        
                    
                    # ABORTION CHECK PRIOR TO WRITING FILES, WILL CEASE IF TRAINING IS ABORTED
                    if train_window.end_training:  
                        messagebox.showerror('Training has Stopped!', 'Training aborted by user;\nProceed from Progress Window', parent=train_window.training_window)
                        train_window.button_frame.enable()
                        return # without a return, aborting training only pauses one iteration of loop
                    
                    # PACKAGING OF EXTRANEOUS PLOTS
                    species_data = [tuple(map(funct, zip(*eval_spectra))) for funct in (min, iumsutils.average, max)] # transpose data, calculate pointwise minima, average, and maxima 
                    spectra_plot = iumsutils.BundledPlot(data=species_data, title=f'Agg. Spectrum of {evaluand}', x_data=range(lower_bound, upper_bound), plot_type='v')
                    #spectra_plots = [iumsutils.BundledPlot(data=spectrum, title=name, x_data=range(lower_bound, upper_bound)) for name, spectrum in eval_data[:num_spectra]]
                    self.spectra_plot = spectra_plot
                    
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verbose)  # keras' self evaluation of loss and accuracy metric
                    loss_plot     = iumsutils.BundledPlot(data=hist.history['loss'], title=f'Training Loss (Final = {test_loss:0.2f})', plot_type='m') 
                    accuracy_plot = iumsutils.BundledPlot(data=hist.history['accuracy'], title=f'Training Accuracy (Final = {test_acc:0.2f})', plot_type='m') 
                    extra_plots   = [spectra_plot, loss_plot, accuracy_plot]
                    
                    # CREATION OF THE SPECIES SUMMARY OBJECT FOR THE CURRENT EVALUAND, PLOTTING OF EVALUATION RESULTS FOR THE CURRENT EVALUAND                
                    curr_spec_sum = iumsutils.SpeciesSummary(evaluand) # SpeciesSummary objects handle calculation and plotting of individual evaluand results
                    curr_spec_sum.add_all_insts(eval_titles, model.predict(np.array(eval_spectra))) # channel the instance names and predictions into the summary and process them
                    
                    train_window.set_status('Plotting Results to Folder...')
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
