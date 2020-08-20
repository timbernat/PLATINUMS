# GUI imports
import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar

# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of IMS Neural Network applications
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 

# Built-in Imports
import csv, math, os, re
from time import time                      
from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from collections import Counter

 # PIP-installed Imports                
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.backends.backend_tkagg

# Neural Net Libraries
import tensorflow as tf
from tensorflow.keras import metrics, Input                   
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split


# SECTION 1 : custom classes needed to operate some features of the main GUI  ---------------------------------------------------                   
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
        self.training_window.geometry('398x163')
        self.end_training = False
        
        # Status Printouts
        self.status_frame = tk.Frame(self.training_window, bd=2, padx=13, relief='groove')
        self.status_frame.grid(row=0, column=0)
        
        self.member_label =   tk.Label(self.status_frame, text='Current Species: ')
        self.curr_member =    tk.Label(self.status_frame)
        self.slice_label =    tk.Label(self.status_frame, text='Current Data Slice: ')
        self.curr_slice =     tk.Label(self.status_frame)
        self.fam_label =      tk.Label(self.status_frame, text='Training Type: ')
        self.curr_fam =       tk.Label(self.status_frame)
        self.round_label =    tk.Label(self.status_frame)
        self.round_progress = Progressbar(self.status_frame, orient='horizontal', length=240, maximum=total_rounds)
        self.epoch_label =    tk.Label(self.status_frame)
        self.epoch_progress = Progressbar(self.status_frame, orient='horizontal', length=240, maximum=num_epochs) 
        self.status_label =   tk.Label(self.status_frame, text='Current Status: ')
        self.curr_status =    tk.Label(self.status_frame)
        
        self.member_label.grid(  row=0, column=0)
        self.curr_member.grid(   row=0, column=1, sticky='w')
        self.slice_label.grid(   row=1, column=0)
        self.curr_slice.grid(    row=1, column=1, sticky='w')
        self.fam_label.grid(     row=2, column=0)
        self.curr_fam.grid(      row=2, column=1, sticky='w')
        self.round_label.grid(   row=3, column=0)
        self.round_progress.grid(row=3, column=1, sticky='w')
        self.epoch_label.grid(   row=4, column=0)
        self.epoch_progress.grid(row=4, column=1, sticky='w')
        self.status_label.grid(  row=5, column=0)
        self.curr_status.grid(   row=5, column=1, sticky='w')
        
        self.reset()
    
        #Training Buttons
        self.button_frame =   ttl.ToggleFrame(self.training_window, '', padx=0, pady=0, row=1)
        self.retrain_button = tk.Button(self.button_frame, text='Retrain', width=17, bg='dodger blue', command=train_funct)
        self.reinput_button = tk.Button(self.button_frame, text='Reset', width=17, bg='orange', command=reset_funct)
        self.abort_button =   tk.Button(self.button_frame, text='Abort Training', width=17, bg='red', command=self.abort)
        
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
        
# Section 2: Start of code for the actual GUI and application ------------------------------------------------------------------------------------------------------------
class PLATINUMS_App:
    def __init__(self, main):
        
        #Main Window
        self.main = main
        self.main.title('PLATIN-UMS 4.2.5-alpha')
        self.main.geometry('445x420')

        #Frame 1
        self.data_frame = ttl.ToggleFrame(self.main, 'Select CSV to Read: ', padx=21, pady=5, row=0)
        self.chosen_file = tk.StringVar()
        self.chem_data = {}
        self.all_species = set()
        self.families = set()
        self.family_mapping = {}
        self.spectrum_size = None
        
        self.csv_menu =       ttl.DynOptionMenu(self.data_frame, self.chosen_file, iumsutils.get_csvs, default='--Choose a CSV--', width=28, colspan=2)
        self.read_label =     tk.Label(self.data_frame, text='Read Status:')
        self.read_status =    ttl.StatusBox(self.data_frame, on_message='CSV Read!', off_message='No File Read', row=1, col=1)
        self.refresh_button = tk.Button(self.data_frame, text='Refresh CSVs', command=self.csv_menu.update, padx=15)
        self.confirm_data =   ttl.ConfirmButton(self.data_frame, self.import_data, padx=2, row=1, col=2)
        
        self.refresh_button.grid(row=0, column=2)
        self.read_label.grid(row=1, column=0)
        
        #Frame 2
        self.input_frame = ttl.ToggleFrame(self.main, 'Select Input Mode: ', padx=5, pady=5, row=1)
        self.read_mode = tk.StringVar()
        self.read_mode.set(None)
        self.selections = []
        
        for i, mode in enumerate( ('Select All', 'By Family', 'By Species') ):
            button = tk.Radiobutton(self.input_frame, text=mode, value=mode, var=self.read_mode, command=self.further_sel)
            button.grid(row=0, column=i)
        self.confirm_sels = ttl.ConfirmButton(self.input_frame, self.confirm_inputs, row=0, col=3, sticky='e')
        self.input_frame.disable()
        
        #Frame 3
        self.num_epochs = None
        self.batchsize = None
        self.learnrate = None
        
        self.hyperparam_frame =    ttl.ToggleFrame(self.main, 'Set Hyperparameters: ', padx=8, pady=5, row=2)
        self.epoch_entry =         ttl.LabelledEntry(self.hyperparam_frame, 'Epochs:', tk.IntVar(), width=19, default=2048)
        self.batchsize_entry =     ttl.LabelledEntry(self.hyperparam_frame, 'Batchsize:', tk.IntVar(), width=19, default=32, row=1)
        self.learnrate_entry =     ttl.LabelledEntry(self.hyperparam_frame, 'Learnrate:', tk.DoubleVar(), width=18, default=2e-5, col=3)
        self.confirm_hyperparams = ttl.ConfirmButton(self.hyperparam_frame, self.confirm_hp, row=1, col=3, cs=2, sticky='e')
        self.hyperparam_frame.disable()
        
        #Frame 4
        self.trimming_min = None
        self.trimming_max = None
        self.slice_decrement = None
        self.num_slices = None
        
        self.param_frame = ttl.ToggleFrame(self.main, 'Set Training Parameters: ', padx=9, pady=5, row=3) 
        self.fam_switch =  ttl.Switch(self.param_frame, 'Familiar Training :', row=0, col=1)
        self.stop_switch = ttl.Switch(self.param_frame, 'Early Stopping: ', row=1, col=1)
        self.trim_switch = ttl.Switch(self.param_frame, 'RIP Trimming: ', row=2, col=1)

        self.cycle_fams = tk.IntVar()
        self.cycle_button = tk.Checkbutton(self.param_frame, text='Cycle', variable=self.cycle_fams)
        self.cycle_button.grid(row=0, column=3)
        
        self.upper_bound_entry =     ttl.LabelledEntry(self.param_frame, 'Upper Bound:', tk.IntVar(), default=400, row=3, col=0)
        self.slice_decrement_entry = ttl.LabelledEntry(self.param_frame, 'Slice Decrement:', tk.IntVar(), default=20, row=3, col=2)
        self.lower_bound_entry =     ttl.LabelledEntry(self.param_frame, 'Lower Bound:', tk.IntVar(), default=50, row=4, col=0)
        self.n_slice_entry =         ttl.LabelledEntry(self.param_frame, 'Number of Slices:', tk.IntVar(), default=1, row=4, col=2)
        self.confirm_train_params =  ttl.ConfirmButton(self.param_frame, self.confirm_tparams, row=5, col=2, cs=2, sticky='e')

        self.trim_switch.dependents = (self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry, self.n_slice_entry)
        self.keras_callbacks = []
        self.param_frame.disable()

        #General/Misc
        self.frames = (self.data_frame, self.input_frame, self.hyperparam_frame, self.param_frame)
        self.switches = (self.fam_switch, self.stop_switch, self.trim_switch)
        self.entries = (self.epoch_entry, self.batchsize_entry, self.learnrate_entry, self.n_slice_entry, self.upper_bound_entry, self.slice_decrement_entry, self.lower_bound_entry)
        self.arrays = (self.chem_data, self.selections, self.family_mapping)

        self.exit_button =  tk.Button(self.main, text='Exit', padx=22, pady=22, bg='red', command=self.shutdown)
        self.reset_button = tk.Button(self.main, text='Reset', padx=20, bg='orange', command=self.reset)
        self.exit_button.grid(row=0, column=4)
        self.reset_button.grid(row=4, column=4)
        
        self.train_button = tk.Button(self.main, text='TRAIN', padx=20, width=45, bg='dodger blue', state='disabled', command=self.begin_training)
        self.train_button.grid(row=4, column=0)
        self.train_window = None
        self.summaries = {}

    
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
        
        self.read_status.set_status(False)
        self.train_button.configure(state='disabled')
        self.csv_menu.reset_default()
        self.read_mode.set(None)
        self.cycle_fams.set(0)
        
        for switch in self.switches:
            switch.disable()
        
        for entry in self.entries:
            entry.reset_default()
        
        for array in self.arrays:
            array.clear()
        self.all_species = set()
        self.families = set()
        
        self.reset_training()  # keras callbacks and summaries, while not in self.arrays, are cleared by this call

                
    #Frame 1 (Reading) Methods 
    def read_chem_data(self): 
        '''Used to read and format the data from the csv provided into a form usable by the training program
        Returns the read data (with vector) and sorted lists of the species and families found in the data'''
        with open(self.chosen_file.get(), 'r') as file:
            for row in csv.reader(file):
                instance = row[0]
                spectrum_data = [float(i) for i in row[1:]]  # convert data point from str to floats
                
                self.chem_data[instance] = spectrum_data
                self.all_species.add( iumsutils.isolate_species(instance) )
                self.families.add( iumsutils.get_family(instance) )
                if not self.spectrum_size:
                    self.spectrum_size = len(spectrum_data)

        self.upper_bound_entry.set_value(self.spectrum_size)
        self.all_species, self.families = sorted(self.all_species), sorted(self.families)  # sort and convert to lists
        
        for index, family in enumerate(self.families):
            one_hot_vector = tuple(i == index and 1 or 0 for i in range(len(self.families)) )
            self.family_mapping[family] = one_hot_vector
                                   
        for instance, data in self.chem_data.items():  # add mapping vector to all data entries
            vector = self.family_mapping[iumsutils.get_family(instance)]
            self.chem_data[instance] = (data, vector)
    
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
        '''logic for selection of members to include in training, based on the chosen selection mode'''
        self.selections.clear()
        if self.read_mode.get() == 'Select All':
            self.selections = self.all_species
        elif self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.input_frame, '1000x210', self.all_species, self.selections, ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.input_frame, '270x85', self.families, self.selections, ncols=3)

    def confirm_inputs(self):
        '''Confirm species input selections'''
        if self.selections == []:
            messagebox.showerror('No selections made', 'Please select species to evaluate')
        else:
            if self.read_mode.get() == 'By Family':  # pick out species by family if selection by family is made
                self.selections = [species for species in self.all_species if iumsutils.get_family(species) in self.selections]
            self.isolate(self.hyperparam_frame)

    
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
            self.keras_callbacks.append(EarlyStopping(monitor='loss', mode='min', verbose=1, patience=8))  # optimize patience, eventually
        
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
        total_rounds = len(self.selections)
        if self.trim_switch.value:  #if RIP trimming is enabled
            total_rounds += len(self.selections)*self.num_slices

        for familiar_cycling in range(1 + self.cycle_fams.get()):  # similar to RIP trimming True/False loop, enables cycling thru both fams and unfams
            if familiar_cycling:
                self.fam_switch.toggle()
            self.reset_training()
            self.train_button.configure(state='disabled')
            self.train_window = TrainingWindow(self.main, total_rounds, self.num_epochs, self.begin_training, self.reset, self.train_button)
            self.keras_callbacks.append(TkEpochs(self.train_window))
            
            end_notification = (self.cycle_fams.get() == familiar_cycling)  # wont show end notification (and pause training) if not on last cycle
            self.training(end_notification=end_notification)
     
    # Section 2A: the training routine code itself 
    def training(self, end_notification=True, num_spectra=2, verbosity=False):
        '''The neural net training function itself'''
        start_time = time()    # log start of runtime
        current_round = 0      # initialize dummy round counter at round 0
        fam_training = self.fam_switch.value 
        
        familiar_str = '{}amiliar'.format(fam_training and 'F' or 'Unf')  # some str formatting based on whether the current training type is familiar or unfamiliar
        self.train_window.set_familiar_status(familiar_str)
        results_folder = Path('Saved Training Results', 'All Spectra', 'Training Results, {}-epoch {}'.format(self.num_epochs, familiar_str))
        
        if os.path.exists(results_folder):   # prompt user to overwrite file if one already exists
            if messagebox.askyesno('Duplicates Found', 'Folder with same data settings found;\nOverwrite old folder?'):
                rmtree(results_folder, ignore_errors=True)
            else:
                self.reset_training()
                return  #terminate prematurely if overwrite permission is not given
        os.makedirs(results_folder)

        with open(results_folder/'Training Settings.txt', 'a') as settings_file:  # make these variables more compact at some point
            settings_file.write('Familiar Training : {}\n'.format(fam_training))
            settings_file.write('Number of Epochs : {}\n'.format(self.num_epochs))
            settings_file.write('Batchsize : {}\n'.format(self.batchsize))
            settings_file.write('Learn Rate : {}\n'.format(self.learnrate))
                  
        for instance, member in enumerate(self.selections):     # iterate over all selected species
            for select_RIP in range(1 + int(self.trim_switch.value)):     # if trimming is enabled, will re-cycle through with trimming
                for segment in range(select_RIP and self.num_slices or 1):  # perform as many slices as are specified (with no trimming, just 1)
                # INITIALIZE SOME INFORMATION REGARDING THE CURRENT ROUND
                    self.train_window.set_status('Training...')
                    curr_family = iumsutils.get_family(member)
                    
                    current_round += 1
                    self.train_window.set_round_progress(current_round)
                                
                    if select_RIP:
                        lower_bound, upper_bound = self.trimming_min, self.trimming_max - self.slice_decrement*segment
                        point_range = 'Points {}-{}'.format(lower_bound, upper_bound)
                    else:
                        lower_bound, upper_bound =  0, self.spectrum_size
                        point_range = 'Full Spectra'
                    self.train_window.set_slice(point_range)                 

                # EVALUATION AND TRAINING SET SELECTION AND SPLITTING 
                    plot_list = []
                    eval_data, eval_titles = [], []
                    features, labels, occurrences = [], [], Counter()
                    train_set_size, eval_set_size = 0, 0
                         
                    for instance, (data, vector) in self.chem_data.items():
                        data = data[lower_bound:upper_bound]                      
                        if iumsutils.isolate_species(instance) == member:  # add all instances of the current species to the evaluation set
                            eval_set_size += 1
                            eval_data.append(data)
                            eval_titles.append(instance)
                                                        
                            if eval_set_size <= num_spectra:  # add sample spectra to the list of plots up to the number assigned
                                plot_list.append((data, instance, 's'))
                       
                        if iumsutils.isolate_species(instance) != member or fam_training:  # add any instance to the training set, unless its a member                                    
                            train_set_size += 1                                       # of the current species and unfamiliar trainin is enabled
                            features.append(data)
                            labels.append(vector)
                            occurrences[iumsutils.get_family(instance)] += 1
                            
                    self.train_window.set_member( '{} ({} instances found)'.format(member, eval_set_size) )                      
                    x_train, x_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.2)                     

                    if verbosity:   # optional prinout, gives overview of the training data and the model settings
                        for (x, y, group) in ( (x_train, y_train, 'training'), (x_test, y_test, 'test') ):
                            print('{} features/labels in {} set ({} of the data)'.format(
                                  (x.shape[0], y.shape[0]), group, round(100 * x.shape[0]/train_set_size, 2)) )
                        print('\n{} features total. Of the {} instances in training dataset:'.format(len(self.chem_data), train_set_size) )
                        for family in self.family_mapping.keys():
                            print('    {}% of data are {}'.format( round(100 * occurrences[family]/train_set_size, 2), family) )
                  
                # MODEL CREATION AND TRAINING
                    with tf.device('CPU:0'):                            
                        model = Sequential()                              # model block is created, layers are created/added in this block
                        model.add(Dense(512, input_dim=(upper_bound - lower_bound), activation='relu'))  # 512 neuron input layer
                        model.add(Dropout(0.5))                                    # dropout layer, to reduce overfit
                        model.add(Dense(512, activation='relu'))                   # 512 neuron hidden layer
                        model.add(Dense(len(self.families), activation='softmax')) # softmax gives prob. dist. of identity over all families
                        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learnrate), metrics=['accuracy']) 
                    if verbosity:
                        model.summary()
                      
                    hist = model.fit(x_train, y_train, epochs=self.num_epochs, batch_size=self.batchsize, callbacks=self.keras_callbacks, verbose=verbosity and 2 or 0)  
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=(verbosity and 2 or 0))  # keras' self evaluation of loss and accuracy metrics
                    
                    if self.train_window.end_training:  # condition to escape training loop of training is aborted
                        messagebox.showerror('Training has Stopped', 'Training aborted by user;\nProceed from Progress Window')
                        self.train_window.button_frame.enable()
                        return     # without this, aborting training only pauses one iteration of loop

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
                    fermi_data = [AAV/max(targets) for AAV in targets]
                    
                # PACKAGING OF ALL PLOTS, APART FROM THE SAMPLE SPECTRA
                    loss_plot = (hist.history['loss'], 'Training Loss (Final = %0.2f)' % test_loss, 'm') 
                    accuracy_plot = (hist.history['accuracy'], 'Training Accuracy (Final = %0.2f%%)' % (100 * test_acc), 'm') 
                    fermi_plot = (fermi_data, '{}, {}/{} correct'.format(member, num_correct, eval_set_size), 'f')  
                    summation_plot = ([iumsutils.average(column) for column in zip(*predictions)], 'Standardized Summation', 'p')
                    prediction_plots =  zip(predictions, eval_titles, tuple('p' for i in predictions))   # all the prediction plots                    
                    
                    for plot in (loss_plot, accuracy_plot, fermi_plot, summation_plot, *prediction_plots): 
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
                    scores.append(num_correct/eval_set_size)

                # CREATION OF FOLDERS, IF NECESSARY, AND PLOTS FOR THE CURRENT ROUND
                    self.train_window.set_status('Writing Results to Folders...')    # creating folders as necessary, writing results to folders 
                    if not os.path.exists(results_folder/point_range):                                                           
                        os.makedirs(results_folder/point_range)
                    self.adagraph(plot_list, 6, results_folder/point_range/member, lower_bound=lower_bound, upper_bound=upper_bound)
        
        # DISTRIBUTION OF SUMMARY DATA TO APPROPRIATE RESPECTIVE FOLDERS
        self.train_window.set_status('Distributing Result Summaries...')  
        for point_range, (fermi_data, score_data) in self.summaries.items(): 
            self.adagraph(fermi_data, 5, results_folder/point_range/'Fermi Summary.png')
                
            with open(results_folder/point_range/'Scores.txt', 'a') as score_file:
                for family, (names, scores) in score_data.items():
                    family_header = '{}\n{}\n{}\n'.format('-'*20, family, '-'*20)  # an underlined heading for each family
                    score_file.write(family_header)   

                    processed_scores = sorted(zip(names, scores), key=lambda x : x[1], reverse=True)  # zip the scores together, then sort them in ascending order by score
                    processed_scores.append( ('AVERAGE : ', iumsutils.average(scores)) )

                    for name, score in processed_scores:
                        score_file.write('{} : {}\n'.format(name, score))
                    score_file.write('\n')   # leave a gap between each family
        
        self.train_window.button_frame.enable()
        self.train_window.set_status('Finished')
        
        runtime = timedelta(seconds=round(time() - start_time))   
        with open(results_folder/'Training Settings.txt', 'a') as settings_file:
            settings_file.write('\nTraining Time : {}'.format(runtime))   # log the runtime in the Train Settings file
        
        if end_notification:
            messagebox.showinfo('Training Complete', 'Routine completed successfully\nResults can be found in "Training Results" folders')
    
    
    def reset_training(self):
        if self.train_window: # if a window already exists
            self.train_window.destroy()
            self.train_window = None
        self.summaries.clear()
        self.keras_callbacks.clear()
    
    def adagraph(self, plot_list, ncols, save_dir, lower_bound=None, upper_bound=None):  # ADD AXIS LABELS AND ELIMINATE BOUNDS/CLASS REFERENCES
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
