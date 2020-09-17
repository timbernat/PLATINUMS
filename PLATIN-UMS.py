# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of IMS Neural Network applications
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 

# Built-in Imports
import json, shutil
from time import time                      
from pathlib import Path
import tkinter as tk    # Built-In GUI imports
from tkinter import messagebox

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
    def __init__(self, train_window):
        super(TkEpochs, self).__init__()
        self.tw = train_window
    
    def on_epoch_begin(self, epoch, logs=None): # update the epoch progress bar at the start of each epoch
        self.tw.set_epoch_progress(epoch + 1)
        
    def on_epoch_end(self, epoch, logs=None):  # abort training if the abort flag has been raised by the training window
        if self.tw.end_training:
            self.model.stop_training = True
        
class TrainingWindow(): 
    '''The window which displays training progress, was easier to subclass outside of the main GUI class'''
    def __init__(self, main, total_rounds, num_epochs, train_funct, reset_funct, exit_funct, train_button):
        self.total_rounds = total_rounds
        self.num_epochs = num_epochs
        self.main = main
        self.train_button = train_button
        self.training_window = tk.Toplevel(main)
        self.training_window.title('Training Progress')
        self.training_window.geometry('390x189')
        self.training_window.attributes('-topmost', True)
        self.end_training = False
        
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
        self.round_progress = ttl.NumberedProgBar(self.status_frame, total=total_rounds, row=3, col=1)
        self.epoch_label    = tk.Label(self.status_frame, text='Training Epoch: ')
        self.epoch_progress = ttl.NumberedProgBar(self.status_frame, total=num_epochs, style_num=2, row=4, col=1) 
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
        self.button_frame   = ttl.ToggleFrame(self.training_window, text='', padx=0, pady=0, row=1)
        self.retrain_button = tk.Button(self.button_frame, text='Retrain', width=17, bg='dodger blue', command=train_funct)
        self.reinput_button = tk.Button(self.button_frame, text='Reset', width=17, bg='orange', command=reset_funct)
        self.exit_button    = tk.Button(self.button_frame, text='Exit', width=17, bg='red', command=exit_funct)
        
        self.retrain_button.grid(row=0, column=0)
        self.reinput_button.grid(row=0, column=1)
        self.exit_button.grid(   row=0, column=2) 
        
        self.button_frame.disable()
        
        # Abort Button, standalone and frameless
        self.abort_button = tk.Button(self.training_window, text='Abort Training', width=54, bg='sienna2', command=self.abort)
        self.abort_button.grid(row=2, column=0)
        
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
        
        
# Section 2: Start of code for the actual GUI and application ---------------------------------------------------------------------------------------
class PLATINUMS_App:
    '''PLATIN-UMS : Prediction, Training, And Labelling INterface for Unlabelled Mobility Spectra'''
    def __init__(self, main):       
        #Main Window
        self.main = main
        self.main.title('PLATIN-UMS 4.4.5-alpha')
        self.main.geometry('445x422')

        # General Buttons
        self.exit_button  = tk.Button(self.main, text='Exit',  padx=22, pady=22, bg='red', command=self.shutdown)
        self.reset_button = tk.Button(self.main, text='Reset', padx=20, bg='orange', command=self.reset)
        self.exit_button.grid(row=0, column=4)
        self.reset_button.grid(row=4, column=4)
        
        self.tpmode = tk.BooleanVar() # option to switch from training to prediction mode, WIP and low priority for now
        self.tpmode_button = tk.Checkbutton(self.main, text='Predict', var=self.tpmode, command=self.switch_tpmode, state='disabled')
        self.tpmode_button.grid(row=2, column=4)
        
        #Frame 1
        self.data_frame  = ttl.ToggleFrame(self.main, text='Select Data File to Read: ', padx=5, pady=5)
        self.chosen_file = tk.StringVar()
        self.data_file = None
        self.chem_data, self.species, self.families, self.family_mapping, self.spectrum_size = {}, [], [], {}, 0
        
        self.json_menu      = ttl.DynOptionMenu(self.data_frame, var=self.chosen_file, option_method=iumsutils.get_by_filetype,
                                                opargs=('.json',), default='--Choose a JSON--', width=32, colspan=2)
        self.read_label     = tk.Label(self.data_frame, text='Read Status:')
        self.read_status    = ttl.StatusBox(self.data_frame, on_message='JSON Read!', off_message='No File Read', width=22, row=1, col=1)
        self.refresh_button = tk.Button(self.data_frame, text='Refresh JSONs', command=self.json_menu.update, padx=15)
        self.confirm_data   = ttl.ConfirmButton(self.data_frame, command=self.import_data, padx=5, row=1, col=2, sticky='e')
        
        self.refresh_button.grid(row=0, column=2, sticky='e')
        self.read_label.grid(    row=1, column=0)
        
        #Frame 2
        self.input_frame = ttl.ToggleFrame(self.main, text='Select Instances to Evaluate: ', padx=5, pady=5, row=1)
        self.read_mode   = tk.StringVar()
        self.read_mode.set(None)
        self.selections  = []
        
        for i, mode in enumerate(('Select All', 'By Family', 'By Species')): # build the selection type buttons sequentially, don't need to reference these w/in the class after building
            tk.Radiobutton(self.input_frame, text=mode, value=mode, var=self.read_mode, command=self.further_sel).grid(row=0, column=i)
        self.confirm_sels = ttl.ConfirmButton(self.input_frame, command=self.confirm_inputs, row=0, col=3, sticky='e')
        self.input_frame.disable()
        
        #Frame 3
        self.hyperparams = {}
        
        self.hyperparam_frame    = ttl.ToggleFrame(self.main, text='Set Hyperparameters: ', padx=8, pady=5, row=2)
        self.epoch_entry         = ttl.LabelledEntry(self.hyperparam_frame, text='Epochs:', var=tk.IntVar(),       width=19, default=2048)
        self.batchsize_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Batchsize:', var=tk.IntVar(),    width=19, default=32, row=1)
        self.learnrate_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Learnrate:', var=tk.DoubleVar(), width=18, default=2e-5, col=3)
        self.confirm_hyperparams = ttl.ConfirmButton(self.hyperparam_frame, command=self.confirm_hp, row=1, col=3, cs=2, sticky='e')
        self.hyperparam_frame.disable()
        
        #Frame 4
        self.trimming_min =    None
        self.trimming_max =    None
        self.slice_decrement = None
        self.num_slices =      None
        self.keras_callbacks = []
        
        self.param_frame = ttl.ToggleFrame(self.main, text='Set Training Parameters: ', padx=9, pady=5, row=3) 
        self.fam_switch  = ttl.Switch(self.param_frame, text='Familiar Training :', row=0, col=1)
        self.stop_switch = ttl.Switch(self.param_frame, text='Early Stopping: ',    row=1, col=1)
        self.trim_switch = ttl.Switch(self.param_frame, text='RIP Trimming: ',      row=2, col=1)

        self.cycle_fams   = tk.IntVar()
        self.cycle_button = tk.Checkbutton(self.param_frame, text='Cycle?', variable=self.cycle_fams)
        self.cycle_button.grid(row=0, column=3)
        
        self.upper_bound_entry      = ttl.LabelledEntry(self.param_frame, text='Upper Bound:', var=tk.IntVar(),      default=400, row=3, col=0)
        self.slice_decrement_entry  = ttl.LabelledEntry(self.param_frame, text='Slice Decrement:', var=tk.IntVar(),  default=20, row=3, col=2)
        self.lower_bound_entry      = ttl.LabelledEntry(self.param_frame, text='Lower Bound:', var=tk.IntVar(),      default=50, row=4, col=0)
        self.n_slice_entry          = ttl.LabelledEntry(self.param_frame, text='Number of Slices:', var=tk.IntVar(), default=1, row=4, col=2)
        self.trim_switch.dependents = (self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry, self.n_slice_entry)
        
        self.save_weights = tk.IntVar()
        self.save_button  = tk.Checkbutton(self.param_frame, text='Save Models after Training?', variable=self.save_weights)
        self.save_button.grid(row=5, column=0, columnspan=2, sticky='w')
        
        self.confirm_train_params   = ttl.ConfirmButton(self.param_frame, command=self.confirm_tparams, row=5, col=2, cs=2, sticky='e')
        self.param_frame.disable()

        # Frame 5 - contains only the button used to trigger a main action  
        self.train_window = None
        self.summaries    = {}
        
        self.activation_frame = ttl.ToggleFrame(self.main, text='', padx=0, pady=0, row=4)
        
        self.train_button = tk.Button(self.activation_frame, text='TRAIN', padx=20, width=45, bg='dodger blue', state='disabled', command=self.training)
        self.train_button.grid(row=0, column=0)
        self.pred_button = tk.Button(self.activation_frame, text='PREDICT', padx=20, width=45, bg='mediumorchid2', state='disabled', command=lambda:None)
        self.pred_button.grid(row=0, column=0)
        self.switch_tpmode()
        
        # packaging some similar widgets together to make certain GUI operations easier
        self.frames   = (self.data_frame, self.input_frame, self.hyperparam_frame, self.param_frame, self.activation_frame)
        self.switches = (self.fam_switch, self.stop_switch, self.trim_switch)
        self.entries  = (self.epoch_entry, self.batchsize_entry, self.learnrate_entry, self.n_slice_entry, self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry)
        
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
    
    def update_data_file(self):
        '''Used to assign the currently selected data file to an internal attribute'''
        self.data_file = Path(self.chosen_file.get())
    
    def switch_tpmode(self):
        target_button = self.tpmode.get() and self.pred_button or self.train_button
        target_button.tkraise()
    
    def reset(self):
        '''reset the GUI to the opening state, along with all variables'''
        self.isolate(self.data_frame)
        
        self.json_menu.reset_default()
        self.update_data_file()
        self.read_status.set_status(False)

        self.read_mode.set(None)
        self.cycle_fams.set(0)
        self.save_weights.set(0)
        self.spectrum_size = 0
        
        for switch in self.switches:
            switch.disable()
        
        for entry in self.entries:
            entry.reset_default()
        
        # include all array attributes for resetting here (excluding self.summaries and self.keras_callbacks, which are already covered by self.reset_training())
        for array in (self.chem_data, self.species, self.families, self.family_mapping, self.selections, self.hyperparams):
            array.clear()
        
        self.reset_training() 
        self.main.lift() # bring main window to forefront

                
    #Frame 1 (Reading) Methods 
    def import_data(self):
        '''Read in data based on the selected data file'''
        self.update_data_file()
        if self.data_file == '--Choose a JSON--':
            messagebox.showerror('File Error', 'No JSON selected')
        else:
            with open(self.data_file, 'r') as data_file:  # unpack and dicard the counts for now
                self.chem_data, self.species, self.families, self.family_mapping, self.spectrum_size, *counts = json.load(data_file).values()
            self.upper_bound_entry.set_value(self.spectrum_size) # adjust the slicing upper bound to the size of spectra passed
            self.read_status.set_status(True)
            self.isolate(self.input_frame)
    
    
    #Frame 2 (Input) Methods
    def further_sel(self): 
        '''logic for selection of evaluands to include in training, based on the chosen selection mode'''
        self.selections.clear()
        if self.read_mode.get() == 'Select All':
            self.selections = self.species
        elif self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.input_frame, '950x190', self.species, self.selections, window_title='Select Species to evaluate', ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.input_frame, '270x85', self.families, self.selections, window_title='Select Families to evaluate over', ncols=3)

    def confirm_inputs(self):
        '''Confirm species input selections'''
        if not self.selections: 
            messagebox.showerror('No selections made', 'Please select species to evaluate')
        else:
            if self.read_mode.get() == 'By Family':  # pick out species by family if selection by family is made
                self.selections = [species for species in self.species if iumsutils.get_family(species) in self.selections]
            self.isolate(self.hyperparam_frame)

    
    # Frame 3 (hyperparameter) Methods
    def confirm_hp(self):
        '''Confirm hyperparameter selections'''
        self.hyperparams['Number of Epochs'] = self.epoch_entry.get_value()
        self.hyperparams['Batch Size'] = self.batchsize_entry.get_value()
        self.hyperparams['Learn Rate'] = self.learnrate_entry.get_value()  
        self.isolate(self.param_frame)
        self.trim_switch.disable()   # ensures that the trimming menu stays greyed out, not necessary for the other switches 
    
    
    #Frame 4 (training parameter) Methods
    def confirm_tparams(self):
        '''Confirm training parameter selections, perform pre-training error checks if necessary'''
        if self.stop_switch.value:
            self.keras_callbacks.append(EarlyStopping(monitor='loss', mode='min', verbose=1, patience=8))  # optimize patience, eventually
        
        if not self.trim_switch.value: # if RIP trimming is not selected
            self.param_frame.disable()
            self.isolate(self.activation_frame) # make the training button clickable 
        else:  # this comment is a watermark - 2020, timotej bernat
            self.num_slices      = self.n_slice_entry.get_value()
            self.slice_decrement = self.slice_decrement_entry.get_value()
            self.trimming_max    = self.upper_bound_entry.get_value()
            self.trimming_min    = self.lower_bound_entry.get_value()
                
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
                self.isolate(self.activation_frame) # make the training button clickable
        
        
    # Section 2A: the training routine itself, along with a dedicated reset method to avoid overwrites between subsequent trainings
    def training(self, num_spectra=2, test_set_proportion=0.2, verbosity=False):
        '''The neural net training function itself; this is where the fun happens'''
        total_rounds = len(self.selections)
        if self.trim_switch.value:  # if RIP trimming is enabled
            total_rounds += len(self.selections)*self.num_slices
        if self.cycle_fams.get():   # if familiar cycling is enabled
            total_rounds *= 2

        # PRE-TRAINING PREPARATIONS TO ENSURE INTERFACE CONTINUITY
        self.reset_training()  # ensure no previous training states are kept before beginning (applies to retrain option specifically)
        self.train_button.configure(state='disabled') # disable training button while training (an idiot-proofing measure)
        self.train_window = TrainingWindow(self.main, total_rounds, self.hyperparams['Number of Epochs'], self.training, self.reset, self.shutdown, self.train_button)
        self.keras_callbacks.append(TkEpochs(self.train_window))
        
        # ACTUAL TRAINING CYCLE BEGINS
        for familiar_cycling in range(1 + self.cycle_fams.get()):  # if cycling is enabled, run throught the training twice, toggling familiar status in between      
            if familiar_cycling:
                self.fam_switch.toggle()  # toggle the unfamiliar status the second time through (only occurs if cycling is enabled)
                self.summaries.clear()
            
            # FILE MANAGEMENT FOR TRAIN SETTINGS RECORD AND RESULTS FOLDERS
            start_time = time()    # log start of runtime, will re-log if cycling is enabled
            fam_training = self.fam_switch.value    
            fam_str = f'{fam_training and "F" or "Unf"}amiliar'  # some str formatting based on whether the current training type is familiar or unfamiliar
            self.train_window.set_familiar_status(fam_str)
            
            self.train_window.set_status('Creating Folders...') 
            results_folder_name = f'{self.hyperparams["Number of Epochs"]}-epoch {fam_str}'
            results_folder = Path('Saved Training Results', f'{self.data_file.stem} Results', results_folder_name)
            if results_folder.exists():   # prompt user to overwrite file if one already exists
                if messagebox.askyesno('Duplicates Found', 'Folder with same data settings found;\nOverwrite old folder?'): 
                    shutil.rmtree(results_folder, ignore_errors=True)
                else:
                    self.reset_training()
                    return  #terminate prematurely if overwrite permission is not given
            try:
                results_folder.mkdir(parents=True)
            except PermissionError: # catch the exception wherein mkdir fails because the folder is inadvertently open
                messagebox.showerror('Permission Error', f'Cannot delete/write folders while {results_folder_name} is open!\
                                    \nNavigate to another directory, then click "Retrain"', parent=self.train_window.training_window)
                self.train_window.button_frame.enable()
                return
                
            with open(results_folder/'Training Settings.txt', 'a') as settings_file:  # record info about training parameters to a text file in the results folder
                settings_file.write(f'Source File : {self.data_file.name}\n\n')
                settings_file.write(f'Familiar Training : {fam_training}\n')
                for hyperparam, value in self.hyperparams.items():
                    settings_file.write(f'{hyperparam} : {value}\n')
            
            # INNER LAYERS OF TRAINING LOOP
            for segment in range(1 + (self.trim_switch.value and self.num_slices or 0)): # by default only train once with the full spectrum. If trimming is enabled...
                for evaluand_idx, evaluand in enumerate(self.selections):                ## ...then train an extra <number of slices> times (DON"T DO ARITHMETICALLY!)
                # INITIALIZATION OF SOME INFORMATION AND DATA FOR THE CURRENT ROUND            
                    self.train_window.set_status('Training...')
                    curr_family = iumsutils.get_family(evaluand)
                    self.train_window.round_progress.increment() # increment round progress

                    if segment == 0:  # first segment will always be over the full spectrum, regardless of trimming
                        lower_bound, upper_bound = 0, self.spectrum_size
                        point_range = 'Full Spectra'
                    else:
                        lower_bound, upper_bound = self.trimming_min, self.trimming_max - self.slice_decrement*(segment - 1) # trimming segments start at segment == 1, hence the -1
                        point_range = f'Points {lower_bound}-{upper_bound}'
                    self.train_window.set_slice(point_range)                 
 
                    eval_data = [ (instance, spectrum[lower_bound:upper_bound]) for instance, (spectrum, vector) in self.chem_data[evaluand].items()]
                    plot_list = [ ((range(lower_bound, upper_bound), spectrum), instance, 's') for instance, spectrum in eval_data[:num_spectra]]
                    eval_titles, eval_spectra = map(list, zip(*eval_data)) # unpack the instance names and spectra via zip, convert zip tuples to lists to allow for insertion
                    eval_set_size = len(eval_data)
                    self.train_window.set_evaluand(f'{evaluand} ({eval_set_size} instances found)')

                    # TRAIN/TEST SET CREATION, MODEL CREATION, AND MODEL TRAINING
                    if not fam_training or evaluand_idx == 0: # for familiars, a model is only trained during the first evaluand at each slice (since training set is identical throughout familiars)
                        training_data = zip( *((spectrum[lower_bound:upper_bound], vector) for species, instances in self.chem_data.items()
                                                                                           for (spectrum, vector) in instances.values()
                                                                                           if species != evaluand or fam_training))               
                        x_train, x_test, y_train, y_test = map(np.array, train_test_split(*training_data, test_size=test_set_proportion)) # keras model only accepts numpy arrays
                        
                        with tf.device('CPU:0'):     # eschews the requirement for a brand-new NVIDIA graphics card (which we don't have anyways)                      
                            model = Sequential()     # model block is created, layers are added to this block
                            model.add(Dense(512, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer, size depends on trimming
                            model.add(Dropout(0.5))                                    # dropout layer, to reduce overfit
                            model.add(Dense(512, activation='relu'))                   # 512 neuron hidden layer
                            model.add(Dense(len(self.families), activation='softmax')) # softmax gives probability distribution of identity over all families
                            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.hyperparams['Learn Rate']), metrics=['accuracy']) 
                        
                        if verbosity: # optional printout to console of some useful information about the model and data
                            print(f'\'{len(training_data)} total instances in training/test sets;')
                            for x, y, group in ((x_train, y_train, "training"), (x_test, y_test, "test")):
                                print(f'{x.shape[0]} features & {y.shape[0]} labels in {group} set ({round(100*x.shape[0]/len(training_data), 2)}% of the data)')
                            model.summary()

                        # actual model training occurs here
                        hist = model.fit(x_train, y_train, callbacks=self.keras_callbacks, verbose=verbosity and 2 or 0, 
                                         epochs=self.hyperparams['Number of Epochs'], batch_size=self.hyperparams['Batch Size'])
                        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=(verbosity and 2 or 0))  # keras' self evaluation of loss and accuracy metric
                        
                        if self.save_weights.get(): # if saving is enabled, save the model to the current result directory
                            self.train_window.set_status('Saving Model...')
                            weights_folder = results_folder/point_range/f'{fam_training and fam_str or evaluand} Model Files'
                            model.save(str(weights_folder)) # path can only be str, for some reason
                            
                            shutil.copy(results_folder/'Training Settings.txt', weights_folder) # copy training settings file to the model folder for removability
                            with open(weights_folder/'Training Settings.txt', 'a') as weight_settings_file:
                                weight_settings_file.write(f'Input Spectrum Slice: {point_range}') # specify the range over which the model can actually be used to predict

                    if self.train_window.end_training:  # escape training loop if training is aborted
                            messagebox.showerror('Training has Stopped!', 'Training aborted by user;\nProceed from Progress Window', parent=self.train_window.training_window)
                            self.train_window.button_frame.enable()
                            return                    # without a return, aborting training only pauses one iteration of loop        
                        
                    # PREDICTION OVER EVALUATION SET, EVALUATION OF PERFORMANCE                 
                    predictions = list(model.predict(np.array(eval_spectra)))
                    targets, num_correct = [], 0   # produce prediction values using the model and determine the accuracy of these predictions
                    for prediction in predictions:
                        target = prediction[self.family_mapping[curr_family].index(1)]
                        targets.append(target)

                        if max(prediction) == target:
                            num_correct += 1

                    targets.sort(reverse=True)
                    fermi_data = iumsutils.normalized(targets)

                    # PACKAGING OF ALL PLOTS, APART FROM THE EVALUATION SPECTRA
                    loss_plot     = (hist.history['loss'], 'Training Loss (Final = %0.2f)' % test_loss, 'm') 
                    accuracy_plot = (hist.history['accuracy'], 'Training Accuracy (Final = %0.2f%%)' % (100 * test_acc), 'm') 
                    fermi_plot    = (fermi_data, f'{evaluand}, {num_correct}/{eval_set_size} correct', 'f')  

                    predictions.insert(0, [iumsutils.average(column) for column in zip(*predictions)]) # prepend standardized sum of predictions to predictions
                    eval_titles.insert(0, 'Standardized Summation') # prepend label to the above list to the titles list
                    prediction_plots = [((self.family_mapping.keys(), prediction), eval_titles[i], 'p') for i, prediction in enumerate(predictions)]  

                    for plot in (loss_plot, accuracy_plot, fermi_plot, *prediction_plots): # collate together the plot data tuples (will implement as objects)
                        plot_list.append(plot)    

                # ORGANIZATION AND ADDITION OF RELEVANT DATA TO THE SUMMARY DICT
                    if point_range not in self.summaries:    # adding relevant data to the summary dict                                 
                        self.summaries[point_range] = ( [], {} )
                    fermi_data, score_data = self.summaries[point_range]
                    fermi_data.append(fermi_plot)

                    if curr_family not in score_data:
                        score_data[curr_family] = ( [], [] )
                    names, scores = score_data[curr_family]
                    names.append(evaluand)
                    scores.append(round(num_correct/eval_set_size, 4))

                # CREATION OF FOLDERS, IF NECESSARY, AND PLOTS FOR THE CURRENT ROUND
                    self.train_window.set_status('Writing Results to Folders...')    # creating folders as necessary, writing results to folders 
                    point_folder = results_folder/point_range # folder containing results for the current spectrum slice 
                    if not point_folder.exists():                                                           
                        point_folder.mkdir(parents=True)
                    iumsutils.adagraph(plot_list, save_dir=results_folder/point_range/evaluand)
        
            # DISTRIBUTION OF SUMMARY DATA TO APPROPRIATE RESPECTIVE FOLDERS
            self.train_window.set_status('Distributing Result Summaries...')  
            for point_range, (fermi_data, score_data) in self.summaries.items(): 
                iumsutils.adagraph(fermi_data, ncols=5, save_dir=results_folder/point_range/'Fermi Summary.png')

                with open(results_folder/point_range/'Scores.txt', 'a') as score_file:
                    for family, (names, scores) in score_data.items():
                        family_header = f'{"-"*20}\n{family}\n{"-"*20}\n'  # an underlined heading for each family
                        score_file.write(family_header)   

                        processed_scores = sorted(zip(names, scores), key=lambda x : x[1], reverse=True) # zip scores together and sort in ascending order by score
                        processed_scores.append( ('AVERAGE', iumsutils.average(scores, precision=4)) )   # add the average score to the score list for each family

                        for name, score in processed_scores:
                            score_file.write(f'{name} : {score}\n')
                        score_file.write('\n')   # leave a gap between each family
            
            with open(results_folder/'Training Settings.txt', 'a') as settings_file:  # log the training time in the Train Settings file
                settings_file.write(f'\nTraining Time : {iumsutils.format_time(time() - start_time)}')   
        
        # POST-TRAINING WRAPPING-UP
        self.train_window.button_frame.enable()  # open up post-training options in the training window
        self.train_window.abort_button.configure(state='disabled') # disable the abort button (idiotproofing against its use after training)
        self.train_window.set_status('Finished')
        messagebox.showinfo('Training Completed Succesfully!', f'Training results can be found in folder:\n{results_folder.parents[0]}', parent=self.train_window.training_window) 
    
    def reset_training(self):
        '''Dedicated reset method for the training cycle, necessary to allow for cycling and retraining'''
        if self.train_window: # if a window already exists
            self.train_window.training_window.destroy()
            self.train_window = None
        self.summaries.clear()
        self.keras_callbacks.clear()
        
if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
