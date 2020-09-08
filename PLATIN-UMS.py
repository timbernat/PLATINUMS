# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of IMS Neural Network applications
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 

# Built-in Imports
import json
from time import time                      
from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from collections import Counter
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
        self.status_frame = tk.Frame(self.training_window, bd=2, padx=20, relief='groove')
        self.status_frame.grid(row=0, column=0)
        
        self.member_label   = tk.Label(self.status_frame, text='Current Species: ')
        self.curr_member    = tk.Label(self.status_frame)
        self.slice_label    = tk.Label(self.status_frame, text='Current Data Slice: ')
        self.curr_slice     = tk.Label(self.status_frame)
        self.fam_label      = tk.Label(self.status_frame, text='Training Type: ')
        self.curr_fam       = tk.Label(self.status_frame)
        self.round_label    = tk.Label(self.status_frame, text='Training Round: ')
        self.round_progress = ttl.NumberedProgBar(self.status_frame, total_rounds, row=3, col=1)
        self.epoch_label    = tk.Label(self.status_frame, text='Training Epoch: ')
        self.epoch_progress = ttl.NumberedProgBar(self.status_frame, num_epochs, style_num=2, row=4, col=1) 
        self.status_label   = tk.Label(self.status_frame, text='Current Status: ')
        self.curr_status    = tk.Label(self.status_frame)
        
        self.member_label.grid(  row=0, column=0)
        self.curr_member.grid(   row=0, column=1, sticky='w')
        self.slice_label.grid(   row=1, column=0)
        self.curr_slice.grid(    row=1, column=1, sticky='w')
        self.fam_label.grid(     row=2, column=0)
        self.curr_fam.grid(      row=2, column=1, sticky='w')
        self.round_label.grid(   row=3, column=0)
        #self.round_progress, like all ttl widgets, has gridding built-in
        self.epoch_label.grid(   row=4, column=0)
        #self.epoch_progress, like all ttl widgets, has gridding built-in
        self.status_label.grid(  row=5, column=0)
        self.curr_status.grid(   row=5, column=1, sticky='w')
        
        self.reset() # ensure menu begins at default status when instantiated
    
        #Training Buttons
        self.button_frame   = ttl.ToggleFrame(self.training_window, '', padx=0, pady=0, row=1)
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
        self.set_member('---')
        self.set_slice('---')
        self.set_familiar_status('---')
        self.set_round_progress(0)
        self.set_epoch_progress(0)
        self.set_status('Standby')
    
    def set_readout(self, readout, value):
        '''Base method for updating a readout on the menu'''
        readout.configure(text=value)
        self.main.update()
    
    def set_member(self, member):
        self.set_readout(self.curr_member, member)
    
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
        
    def destroy(self):
        self.train_button.configure(state='normal')
        self.training_window.destroy()
        
# Section 2: Start of code for the actual GUI and application ---------------------------------------------------------------------------------------
class PLATINUMS_App:
    def __init__(self, main):
        
        #Main Window
        self.main = main
        self.main.title('PLATIN-UMS 4.3.9-alpha')
        self.main.geometry('445x420')

        #Frame 1
        self.data_frame  = ttl.ToggleFrame(self.main, 'Select JSON to Read: ', padx=21, pady=5)
        self.chosen_file = tk.StringVar()
        self.data_file = None
        self.chem_data, self.species, self.families, self.family_mapping, self.spectrum_size, self.species_count = {}, [], [], {}, 0, {}
        
        self.json_menu      = ttl.DynOptionMenu(self.data_frame, self.chosen_file, lambda : iumsutils.get_by_filetype('.json'), default='--Choose a JSON--', width=28, colspan=2)
        self.read_label     = tk.Label(self.data_frame, text='Read Status:')
        self.read_status    = ttl.StatusBox(self.data_frame, on_message='JSON Read!', off_message='No File Read', row=1, col=1)
        self.refresh_button = tk.Button(self.data_frame, text='Refresh JSONs', command=self.json_menu.update, padx=11)
        self.confirm_data   = ttl.ConfirmButton(self.data_frame, self.import_data, padx=2, row=1, col=2)
        
        self.refresh_button.grid(row=0, column=2)
        self.read_label.grid(    row=1, column=0)
        
        #Frame 2
        self.input_frame = ttl.ToggleFrame(self.main, 'Select Input Mode: ', padx=5, pady=5, row=1)
        self.read_mode   = tk.StringVar()
        self.read_mode.set(None)
        self.selections  = []
        
        for i, mode in enumerate(('Select All', 'By Family', 'By Species')):
            button = tk.Radiobutton(self.input_frame, text=mode, value=mode, var=self.read_mode, command=self.further_sel)
            button.grid(row=0, column=i)
        self.confirm_sels = ttl.ConfirmButton(self.input_frame, self.confirm_inputs, row=0, col=3, sticky='e')
        self.input_frame.disable()
        
        #Frame 3
        self.hyperparams = {}
        
        self.hyperparam_frame    = ttl.ToggleFrame(self.main, 'Set Hyperparameters: ', padx=8, pady=5, row=2)
        self.epoch_entry         = ttl.LabelledEntry(self.hyperparam_frame, 'Epochs:', tk.IntVar(), width=19, default=2048)
        self.batchsize_entry     = ttl.LabelledEntry(self.hyperparam_frame, 'Batchsize:', tk.IntVar(), width=19, default=32, row=1)
        self.learnrate_entry     = ttl.LabelledEntry(self.hyperparam_frame, 'Learnrate:', tk.DoubleVar(), width=18, default=2e-5, col=3)
        self.confirm_hyperparams = ttl.ConfirmButton(self.hyperparam_frame, self.confirm_hp, row=1, col=3, cs=2, sticky='e')
        self.hyperparam_frame.disable()
        
        #Frame 4
        self.trimming_min = None
        self.trimming_max = None
        self.slice_decrement = None
        self.num_slices = None
        self.keras_callbacks = []
        
        self.param_frame = ttl.ToggleFrame(self.main, 'Set Training Parameters: ', padx=9, pady=5, row=3) 
        self.fam_switch  = ttl.Switch(self.param_frame, 'Familiar Training :', row=0, col=1)
        self.stop_switch = ttl.Switch(self.param_frame, 'Early Stopping: ', row=1, col=1)
        self.trim_switch = ttl.Switch(self.param_frame, 'RIP Trimming: ', row=2, col=1)

        self.cycle_fams   = tk.IntVar()
        self.cycle_button = tk.Checkbutton(self.param_frame, text='Cycle?', variable=self.cycle_fams)
        self.cycle_button.grid(row=0, column=3)
        
        self.upper_bound_entry      = ttl.LabelledEntry(self.param_frame, 'Upper Bound:', tk.IntVar(), default=400, row=3, col=0)
        self.slice_decrement_entry  = ttl.LabelledEntry(self.param_frame, 'Slice Decrement:', tk.IntVar(), default=20, row=3, col=2)
        self.lower_bound_entry      = ttl.LabelledEntry(self.param_frame, 'Lower Bound:', tk.IntVar(), default=50, row=4, col=0)
        self.n_slice_entry          = ttl.LabelledEntry(self.param_frame, 'Number of Slices:', tk.IntVar(), default=1, row=4, col=2)
        self.trim_switch.dependents = (self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry, self.n_slice_entry)
        
        self.save_weights = tk.IntVar()
        self.save_button  = tk.Checkbutton(self.param_frame, text='Save Models after Training?', variable=self.save_weights)
        self.save_button.grid(row=5, column=0, columnspan=2, sticky='w')
        
        self.confirm_train_params   = ttl.ConfirmButton(self.param_frame, self.confirm_tparams, row=5, col=2, cs=2, sticky='e')
        self.param_frame.disable()

        #General/Misc
        self.frames   = (self.data_frame, self.input_frame, self.hyperparam_frame, self.param_frame)
        self.switches = (self.fam_switch, self.stop_switch, self.trim_switch)
        self.entries  = (self.epoch_entry, self.batchsize_entry, self.learnrate_entry, self.n_slice_entry,
                        self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry)
        self.arrays   = (self.chem_data, self.species, self.families, self.family_mapping, self.species_count, self.selections, self.hyperparams)
        # consider eliminating arrays and moving all lists directly to the reset method

        self.exit_button  = tk.Button(self.main, text='Exit', padx=22, pady=22, bg='red', command=self.shutdown)
        self.reset_button = tk.Button(self.main, text='Reset', padx=20, bg='orange', command=self.reset)
        self.exit_button.grid(row=0, column=4)
        self.reset_button.grid(row=4, column=4)
        
        self.train_button = tk.Button(self.main, text='TRAIN', padx=20, width=45, bg='dodger blue', state='disabled', command=self.training)
        self.train_button.grid(row=4, column=0)
        self.train_window = None
        self.summaries    = {} # deliberately NOT a member of self.arrays because of the self.reset_training() method

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
    
    def reset(self):
        '''reset the GUI to the opening state, along with all variables'''
        self.isolate(self.data_frame)
        
        self.read_status.set_status(False)
        self.train_button.configure(state='disabled')
        self.json_menu.reset_default()
        self.update_data_file()
        self.read_mode.set(None)
        self.cycle_fams.set(0)
        self.save_weights.set(0)
        self.spectrum_size = 0
        
        for switch in self.switches:
            switch.disable()
        
        for entry in self.entries:
            entry.reset_default()
        
        for array in self.arrays:
            array.clear()
        
        self.reset_training()  # keras callbacks and summaries, while not in self.arrays, are cleared by this call
        self.main.lift()

                
    #Frame 1 (Reading) Methods 
    def import_data(self):
        '''Read in data based on the selected data file'''
        self.update_data_file()
        if self.data_file == '--Choose a JSON--':
            messagebox.showerror('File Error', 'No JSON selected')
        else:
            with open(self.data_file, 'r') as data_file:
                self.chem_data, self.species, self.families, self.family_mapping, self.spectrum_size, self.species_count = json.load(data_file).values()
            self.upper_bound_entry.set_value(self.spectrum_size) # adjust the slicing upper bound to the size of spectra passed
            self.read_status.set_status(True)
            self.isolate(self.input_frame)
    
    
    #Frame 2 (Input) Methods
    def further_sel(self): 
        '''logic for selection of members to include in training, based on the chosen selection mode'''
        self.selections.clear()
        if self.read_mode.get() == 'Select All':
            self.selections = self.species
        elif self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.input_frame, '1000x210', self.species, self.selections, ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.input_frame, '270x85', self.families, self.selections, ncols=3)

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
            self.train_button.configure(state='normal')
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
                self.train_button.configure(state='normal')
        
        
    # Section 2A: the training routine itself, along with a dedicated reset method to avoid overwrites between subsequent trainings
    def training(self, num_spectra=2, verbosity=False):
        '''The neural net training function itself; this is where the fun happens'''
        total_rounds = len(self.selections)
        if self.trim_switch.value:  # if RIP trimming is enabled
            total_rounds += len(self.selections)*self.num_slices
        if self.cycle_fams.get():   # if familiar cycling is enabled
            total_rounds *= 2

        # PRE-TRAINING PREPARATIONS TO ENSURE INTERFACE CONTINUITY
        self.reset_training()  # ensure no previous training states are kept before beginning (applies to retrain option specifically)
        self.train_button.configure(state='disabled') # disable training button while training (for idiot-proofing purposes)
        self.train_window = TrainingWindow(self.main, total_rounds, self.hyperparams['Number of Epochs'], self.training, self.reset, self.shutdown, self.train_button)
        self.keras_callbacks.append(TkEpochs(self.train_window))
        
        # ACTUAL TRAINING CYCLE BEGINS
        save_weights = self.save_weights.get() #  flag for whether or not to save model weights after each training loop
        for familiar_cycling in range(1 + self.cycle_fams.get()):  # if cycling is enabled, run throught the training twice, toggling familiar status in between      
            if familiar_cycling:
                self.fam_switch.toggle()  # toggle the unfamiliar status the second time through (only occurs if cycling is enabled)
                self.summaries.clear()
            
            # FILE MANAGEMENT FOR TRAIN SETTINGS RECORD AND RESULTS FOLDERS
            start_time = time()    # log start of runtime, will re-log if cycling is enabled
            fam_training = self.fam_switch.value    
            familiar_str = f'{fam_training and "F" or "Unf"}amiliar'  # some str formatting based on whether the current training type is familiar or unfamiliar
            self.train_window.set_familiar_status(familiar_str)
            
            self.train_window.set_status('Creating Folders...') 
            results_folder = Path('Saved Training Results', f'{self.data_file.stem} Results', f'{self.hyperparams["Number of Epochs"]}-epoch {familiar_str}')
            if results_folder.exists():   # prompt user to overwrite file if one already exists
                if messagebox.askyesno('Duplicates Found', 'Folder with same data settings found;\nOverwrite old folder?'): 
                    rmtree(results_folder, ignore_errors=True)
                else:
                    self.reset_training()
                    return  #terminate prematurely if overwrite permission is not given
            results_folder.mkdir(parents=True)

            with open(results_folder/'Training Settings.txt', 'a') as settings_file:  # make these variables more compact at some point
                settings_file.write(f'Source File : {self.data_file.name}\n\n')
                settings_file.write(f'Familiar Training : {fam_training}\n')
                for hyperparam, value in self.hyperparams.items():
                    settings_file.write(f'{hyperparam} : {value}\n')
            
            # INNER LAYERS OF TRAINING LOOP
            for member in self.selections:         # iterate over all selected species
                for select_RIP in range(1 + int(self.trim_switch.value)):   # if trimming is enabled, will re-cycle through with trimming
                    for segment in range(select_RIP and self.num_slices or 1):  # perform as many slices as are specified (with no trimming, just 1)
                    # INITIALIZATION OF SOME INFORMATION REGARDING THE CURRENT ROUND
                        self.train_window.set_status('Training...')
                        curr_family = iumsutils.get_family(member)
                        self.train_window.round_progress.increment() # increment round progress

                        if select_RIP:
                            lower_bound, upper_bound = self.trimming_min, self.trimming_max - self.slice_decrement*segment
                            point_range = f'Points {lower_bound}-{upper_bound}'
                        else:
                            lower_bound, upper_bound =  0, self.spectrum_size
                            point_range = 'Full Spectra'
                        self.train_window.set_slice(point_range)                 

                    # EVALUATION AND TRAINING SET SELECTION AND SPLITTING 
                        plot_list = []
                        eval_data, eval_titles = [], [] # the first plot, after results are produced, will be the summation plot
                        features, labels, occurrences = [], [], Counter()
                        train_set_size, eval_set_size = 0, 0

                        for instance, (data, vector) in self.chem_data.items():
                            data = data[lower_bound:upper_bound]                      
                            if iumsutils.isolate_species(instance) == member:  # add all instances of the current species to the evaluation set
                                eval_set_size += 1
                                eval_data.append(data)
                                eval_titles.append(instance)

                                if eval_set_size <= num_spectra:  # add the specified number of sample spectra to the list of plots
                                    plot_list.append( ((range(lower_bound, upper_bound), data), instance, 's'))

                            if iumsutils.isolate_species(instance) != member or fam_training:  # add any instance to the training set, unless its a member
                                train_set_size += 1                                            # of the current species and unfamiliar training is enabled
                                features.append(data)
                                labels.append(vector)
                                occurrences[iumsutils.get_family(instance)] += 1

                        self.train_window.set_member(f'{member} ({eval_set_size} instances found)')                      
                        x_train, x_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.2)                     

                        if verbosity:   # optional printout to console, gives overview of the training data and the model settings
                            for (x, y, group) in ((x_train, y_train, "training"), (x_test, y_test, "test")):
                                print(f'{x.shape[0]} features & {y.shape[0]} labels in {group} set ({round(100*x.shape[0]/train_set_size, 2)}% of the data)')
                            print(f'\n{len(self.chem_data)} features total. Of the {train_set_size} instances in training dataset:')
                            for family in self.family_mapping.keys():
                                print(f'\t{round(100*occurrences[family]/train_set_size, 2)}% of data are {family}')

                    # MODEL CREATION AND TRAINING
                        with tf.device('CPU:0'):     # eschews the requirement for a brand-new NVIDIA graphics card (which we don't have anyways)                      
                            model = Sequential()     # model block is created, layers are added to this block
                            model.add(Dense(512, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer, size depends on trimming
                            model.add(Dropout(0.5))                                    # dropout layer, to reduce overfit
                            model.add(Dense(512, activation='relu'))                   # 512 neuron hidden layer
                            model.add(Dense(len(self.families), activation='softmax')) # softmax gives prob. dist. of identity over all families
                            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.hyperparams['Learn Rate']), metrics=['accuracy']) 
                        if verbosity:
                            model.summary()

                        # model training occurs here
                        hist = model.fit(x_train, y_train, callbacks=self.keras_callbacks, verbose=verbosity and 2 or 0, 
                                         epochs=self.hyperparams['Number of Epochs'], batch_size=self.hyperparams['Batch Size'])
                        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=(verbosity and 2 or 0))  # keras' self evaluation of loss and accuracy metrics

                        if self.train_window.end_training:  # condition to escape training loop if training is aborted
                            messagebox.showerror('Training has Stopped!', 'Training aborted by user;\nProceed from Progress Window')
                            self.train_window.button_frame.enable()
                            return     # without this, aborting training only pauses one iteration of loop
                        
                        if save_weights: # if training has not been aborted and saving is enabled, save the model to the current result directory
                            self.train_window.set_status('Saving Model...')
                            model.save(str(results_folder/point_range/f'{member} Model Files')) # path can only be str, for some reason

                    # PREDICTION OVER EVALUATION SET, EVALUATION OF PERFORMANCE                 
                        targets, num_correct = [], 0    # produce prediction values using the model and determine the accuracy of these predictions
                        predictions = [list(prediction) for prediction in model.predict(np.array(eval_data))]

                        for prediction in predictions:
                            target_index = self.family_mapping[curr_family].index(1)
                            target = prediction[target_index]
                            targets.append(target)

                            if max(prediction) == target:
                                num_correct += 1
                                
                        targets.sort(reverse=True)
                        fermi_data = iumsutils.normalized(targets)
                        

                    # PACKAGING OF ALL PLOTS, APART FROM THE EVALUATION SPECTRA
                        loss_plot     = (hist.history['loss'], 'Training Loss (Final = %0.2f)' % test_loss, 'm') 
                        accuracy_plot = (hist.history['accuracy'], 'Training Accuracy (Final = %0.2f%%)' % (100 * test_acc), 'm') 
                        fermi_plot    = (fermi_data, f'{member}, {num_correct}/{eval_set_size} correct', 'f')  
                        
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
                        names.append(member)
                        scores.append(round(num_correct/eval_set_size, 4))

                    # CREATION OF FOLDERS, IF NECESSARY, AND PLOTS FOR THE CURRENT ROUND
                        self.train_window.set_status('Writing Results to Folders...')    # creating folders as necessary, writing results to folders 
                        point_folder = results_folder/point_range # folder containing results for the current spectrum slice 
                        if not point_folder.exists():                                                           
                            point_folder.mkdir(parents=True)
                        iumsutils.adagraph(plot_list, 6, save_dir=results_folder/point_range/member)
        
            # DISTRIBUTION OF SUMMARY DATA TO APPROPRIATE RESPECTIVE FOLDERS
            self.train_window.set_status('Distributing Result Summaries...')  
            for point_range, (fermi_data, score_data) in self.summaries.items(): 
                iumsutils.adagraph(fermi_data, 5, save_dir=results_folder/point_range/'Fermi Summary.png')

                with open(results_folder/point_range/'Scores.txt', 'a') as score_file:
                    for family, (names, scores) in score_data.items():
                        family_header = f'{"-"*20}\n{family}\n{"-"*20}\n'  # an underlined heading for each family
                        score_file.write(family_header)   

                        processed_scores = sorted(zip(names, scores), key=lambda x : x[1], reverse=True)  # zip scores together and sort in ascending order by score
                        processed_scores.append( ('AVERAGE : ', iumsutils.average(scores)) )

                        for name, score in processed_scores:
                            score_file.write(f'{name} : {score}\n')
                        score_file.write('\n')   # leave a gap between each family
            
            with open(results_folder/'Training Settings.txt', 'a') as settings_file:  # log the training time in the Train Settings file
                runtime = timedelta(seconds=round(time() - start_time))
                settings_file.write(f'\nTraining Time : {runtime}')   
        
        # POST-TRAINING WRAPPING-UP
        self.train_window.button_frame.enable()  # open up post-training options in the training window
        self.train_window.abort_button.configure(state='disabled') # disable the abort button (idiotproofing against its use after training)
        self.train_window.set_status('Finished')
        messagebox.showinfo('Training Completed Succesfully!', f'Training results can be found in folder:\n{results_folder.parents[0]}',
                            parent=self.train_window.training_window)  # ensure the message originates from the training window's root
    
    def reset_training(self):
        '''Dedicated reset method for the training cycle, necessary to allow for cycling and retraining'''
        if self.train_window: # if a window already exists
            self.train_window.destroy()
            self.train_window = None
        self.summaries.clear()
        self.keras_callbacks.clear()
        

if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
