# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of MS Neural Network applications
import plotutils           # library of plotting utilities useful for summarizing and visualizing training results
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 

# Built-in Imports
import os, json, time                    
from pathlib import Path

# Built-In GUI imports
import tkinter as tk   
from tkinter import messagebox

# PIP-installed Imports                
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split

# Neural Net Libraries
import tensorflow as tf                    
from tensorflow.keras.optimizers import Adam, RMSprop, SGD # use sgd for NW param optimizer (RMS is loss metric, NOT optimizer)
from tensorflow.keras.metrics import RootMeanSquaredError, CategoricalAccuracy 
#from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy # builtin reference by string supported
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
tf.get_logger().setLevel('ERROR') # suppress deprecation warnings which seem to riddle tensorflow


# SECTION 1 : custom classes needed to operate some features of the main GUI  ---------------------------------------------------                   
class TWInter(Callback):   
    '''A callback for interfacing between the keras' model training and the training window GUI'''
    def __init__(self, training_window, metric='loss'):
        self.tw = training_window
        self.metric = metric
    
    def on_epoch_begin(self, epoch, logs=None): # update the epoch progress bar at the start of each epoch
        if epoch == 0: # reset progress bar with each new training
            self.tw.epoch_progress.reset()      
        self.tw.epoch_progress.increment() # increment progress each epoch
        self.tw.app.main.update() # allows for mobility of the main window while training
        
    def on_epoch_end(self, epoch, logs=None):  # abort training if the abort flag has been raised by the training window
        self.tw.loss_plot.update(epoch, logs.get(self.metric))
        if self.tw.abort_flag:
            self.model.stop_training = True # done in case training is occurring when abort is issued
            
class Convergence(Callback):   
    '''A callback which facilitates early stopping of training when loss convergence is reached'''
    def __init__(self, threshold, metric='loss'):
        self.threshold = threshold
        self.stopped_epoch = None
        self.metric = metric
        
    def on_epoch_end(self, epoch, logs=None):  # abort training if the abort flag has been raised by the training window
        if logs.get(self.metric) < self.threshold:
            self.model.stop_training = True # immediately terminate training
            self.stopped_epoch = epoch + 1  # flag current epoch (1-indexed) as the point of termination
            
    def reset_flag(self):
        self.stopped_epoch = None

class TrainingWindow(): 
    '''The window which displays training progress and information, subclassed TopLevel allows it to be separate from the main GUI'''
    def __init__(self, frontend_app, parent_name, nw_compat=False):
    # MIRRORING RELEVANT TRAINING PARAMETERS AND GUI FEATURES FROM MAIN APP 
        self.app    = frontend_app # the main app GUI itself (as an object)
        self.main   = frontend_app.main # mirroring relevant methods and attributes to the current trainer object      
        self.parent_name = parent_name # the path to the folder in which training results will be saved
        
        self.nw_compat = nw_compat
        self.metric = (nw_compat and 'rmse' or 'loss')
        
        self.mapping   = frontend_app.family_mapping       
        self.evaluands = frontend_app.evaluands            
        plotutils.Base_RC.set_uc_mapping(self.mapping) # set unit circle mapping for radar charts
        
        # param names: 'data_files','num_epochs', 'batchsize','learnrate', 'slice_decrement','trimming_min','num_slices','cycle_fams','fam_training','save_weights','convergence','tolerance'
        self.parameters = frontend_app.parameters # copy over parameters dict for reference when writing presets
        for param, value in frontend_app.parameters.items():
            setattr(self, param, value) # individually copy over parameters internally for reference

        # determining which callbacks to inject into training
        self.keras_callbacks = [TWInter(self, metric=self.metric)] # in every case, add the tkinter-keras interface callback to the callbacks list   
        if self.convergence:
            self.num_epochs = 1000000 # set epoch size to very large number to account for unconstrained traning
            self.convergence_callback = Convergence(self.tolerance, metric=self.metric) # named so this parameter can be referenced within code after stopping
            self.keras_callbacks.append(self.convergence_callback)
        
    # CREATING THE TRAINING WINDOW ITSELF
        self.training_window = tk.Toplevel(self.main)
        self.training_window.title('Training Progress')
        self.training_window.attributes('-topmost', True)
        
    # TKINTER OBJECTS FOR STATUS DISPLAY
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
        
        # Dynamically-updatable loss plot
        self.loss_plot = ttl.DynamicPlot(self.training_window, 'Progress Monitor', 'Training Epoch', 'Training Loss', row=0, col=1, rs=3, cs=1)
        
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
        
    # TRAINING BUTTONS
        self.button_frame = ttl.ToggleFrame(self.training_window, text='', padx=0, pady=0, row=1) # leave commands as they are (internal methods destroy train window also) 
        
        self.retrain_button    = tk.Button(self.button_frame, text='Retrain', width=17, underline=2, bg='deepskyblue2', command=self.training) 
        self.reset_main_button = tk.Button(self.button_frame, text='Reset',   width=17, underline=0, bg='orange'      , command=self.reset_main)
        self.quit_button       = tk.Button(self.button_frame, text='Quit',    width=17, underline=0, bg='red'         , command=self.quit_main)
        
        self.retrain_button.grid(   row=0, column=0)
        self.reset_main_button.grid(row=0, column=1)
        self.quit_button.grid(      row=0, column=2) 
           
        self.abort_flag = False # intended to keep track of whether training was terminated by user or by callbacks
        self.abort_button = tk.Button(self.training_window, text='Abort Training', width=54, underline=1, bg='sienna2', command=self.raise_abort_flag)
        self.abort_button.grid(row=2, column=0, pady=(0,2)) # Abort Button IS standalone and frameless
        
        self.training_window.bind('<Key>', self.key_bind)
    
    def __del__(self):
        print('Training Window destructor called')
    
    # DEFINE DYNAMIC KEYBINDINGS
    def key_bind(self, event):
        '''command to bind hotkeys, contingent on menu enabled status'''
        if self.button_frame.state == 'normal':
            if event.char == 't':
                self.training()
            elif event.char == 'r':
                self.reset_main()
            elif event.char == 'q':
                self.quit()
        elif self.abort_button.cget('state') == 'normal' and event.char == 'b':
            self.abort()
    
    # TRAINING BUTTON METHODS       
    def reset_main(self):
        self.training_window.destroy()   # kill self (to avoid persistence issues)
        self.app.reset() # reset the main window
        
    def quit_main(self):    
        self.loss_plot.__del__()
        self.training_window.destroy() 
        self.app.quit()
    
    def abort(self):   
        self.set_status('Training Aborted')       
        self.abort_button.configure(state='disabled')
        self.button_frame.enable()
        messagebox.showerror('Training has Stopped!', 'Training aborted by user;\nProceed from Progress Window', parent=self.training_window)
      
    def raise_abort_flag(self):
        self.abort_flag = True
        
    def lower_abort_flag(self):
        self.abort_flag = False
    
    def reset(self):
        for prog_bar in self.prog_bars:
            prog_bar.set_progress(0)
            self.main.update() # progress bars sometimes appear frozen when no update occurs???
            
        for readout in self.readouts:
            self.set_readout(readout, '---')
        
        self.lower_abort_flag()
        self.set_status('Standby')
        self.abort_button.configure(state='normal')
        self.button_frame.disable()
    
    # READOUT ADJUSTMENT METHODS
    def set_readout(self, readout, value):
        '''Base method for updating a readout on the menu'''
        readout.configure(text=value)
        self.main.update()
    
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
     
    # ROBUST FILE MANAGEMENT CHECK TO AVOID OVERWRITING AND TO ENSURE AN EMPTY FOLDER WITH THE APPROPRIATE NAME EXISTS PRIOR TO TRAINING
    def prepare_result_folder(self, save_folder, parent_name):
        '''Handles folder naming and overwriting pre-training to guarantee results have a unique folder into which they can be written'''    
        parent_folder = save_folder/parent_name # same name applied to Path object, useful for incrementing file_id during overwrite checking
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
                        parent_folder = save_folder/f'{parent_name}({file_id})' # appends differentiating number to end of duplicated files 
                        continue # return to start of checking loop

                # this branch is only executed if no preset file is found OR (implicitly) if the existing preset matches the one found
                if messagebox.askyesno('Request Overwrite', 'Folder with same name and training parameters found; Allow overwrite of folder?', parent=self.training_window):
                    try:
                        iumsutils.clear_folder(parent_folder) # empty the folder if permission is given
                        break
                    except PermissionError: # if emptying fails because user forgot to close files, notify them and exit training loop
                        messagebox.showerror('Overwrite Error', f'Folder "{parent_folder}"\nstill has file(s) open, close files and retrain', parent=self.training_window)
                else:
                    return # return None if overwrite is not allowed
        return parent_folder # return path-like object pointing to folder if successful
        
    # MAIN TRAINING METHODS -- THIS IS WHERE THE MAGIC HAPPENS! 
    def training(self, test_set_proportion=0.2, keras_verbosity=0): # use keras_verbosity of 2 for lengthy and detailed console printouts
        '''The function defining the neural network training sequence and parameters'''
        # INITIAL FILE MANAGEMENT AND PREPARATION
        parent_folder = self.prepare_result_folder(save_folder=self.app.save_path, parent_name=self.parent_name)
        if not parent_folder:
            return # if user does not allow overwrite, do not begin the training cycle
        
        preset_path = parent_folder/'Training Preset.json' # save current training settings to a preset for reproducability
        with preset_path.open(mode='w') as preset_file: 
            json.dump(self.parameters, preset_file) 
            
        self.reset() # ensure menu begins at default status when instantiated
        self.round_progress.set_max(len(self.evaluands)*(self.num_slices + 1)*(self.cycle_fams + 1)) # compute and display the number of training rounds to occur
        self.file_num_progress.set_max(len(self.parameters['data_files'])) # display number of data files to be iterated through       

        # BEGINNING OF ACTUAL TRAINING CYCLE, ITERATE THROUGH EVERY DATA FILE PASSED
        overall_start_time = time.time() # get start of runtime for entire training routine
        point_ranges = set() # stores the slicing point ranges, used later to spliced together familiar and unfamiliar reulst files - is a set to avoid duplication
        for file_num, data_file in enumerate(self.data_files):
            self.reset() # clear training window between different files
            self.set_status('Initializing Training...')            
            self.set_file(data_file) # indicate the current file in the training window   
            self.file_num_progress.increment()

            results_folder = parent_folder/f'{data_file} Results' # main folder with all results for a particular data file  
            results_folder.mkdir(exist_ok=True)        

            json_data = iumsutils.load_chem_json(self.app.data_path/f'{data_file}.json') # load in the data file for the current ecaluation round
            chem_data = json_data['chem_data']        # isolate the spectral/chemical data 
            spectrum_size = json_data['spectrum_size'] # set maximum length to be the full spectrum length of the current dataset

            # SECONDARY TRAINING LOOP, REGULATES CYCLING BETWEEN FAMILIARS AND UNFAMILIARS
            for familiar_cycling in range(self.cycle_fams + 1):    
                start_time = time.time() # log start time of each cycle, so that training duration can be computed afterwards  
                
                local_fam_training = (self.fam_training ^ familiar_cycling) # xor used to encode the inversion behavior on second cycle    
                fam_str = f'{local_fam_training and "F" or "Unf"}amiliar'  
                self.app.fam_switch.set_value(local_fam_training)        # purely cosmetic, but allows the user to see that cycling is in fact occurring
                self.set_familiar_status(fam_str)

                curr_fam_folder = results_folder/fam_str # folder with the results from the current (un)familiar training, parent for all that follow 
                curr_fam_folder.mkdir(exist_ok=True)
                
                log_file_path = curr_fam_folder/'Log File.txt'
                log_file_path.touch() # make sure the log file exists
                
                loss_file_path = curr_fam_folder/'Training Losses.json'
                loss_file_path.touch() # make sure the log file exists
                losses = {}
                
                # TERTIARY TRAINING LOOP, REGULATES SLICING
                for segment in range(self.num_slices + 1): 
                    lower_bound, upper_bound = self.trimming_min, spectrum_size - self.slice_decrement*segment
                    point_range = f'Points {lower_bound}-{upper_bound}'
                    if lower_bound == 0 and upper_bound == spectrum_size: 
                        point_range = point_range + ' (Full Spectra)' # indicate whether the current slice is in fact truncated
                   
                    point_ranges.add(point_range) # add current range to point ranges (will not duplicate if cycling familiars, since point_ranges is a set)
                    self.set_slice(point_range)
                    curr_slice_folder = curr_fam_folder/point_range
                    curr_slice_folder.mkdir(exist_ok=True)

                    # INNERMOST TRAINING LOOP, CYCLES THROUGH ALL THE SELECTED EVALUANDS
                    predictions, round_summary = [
                        {family : 
                            {species : {}
                                for species in sorted((species for species in self.evaluands if iumsutils.get_family(species) == family), key=iumsutils.get_carbon_ordering)
                            }
                            for family in sorted(set(iumsutils.get_family(species) for species in self.evaluands))
                        } 
                    for i in range(2)]  # outer data structure is the same for both predictions and the round summary
                    
                    for evaluand_idx, evaluand in enumerate(self.evaluands):                  
                        self.set_status('Training...')
                        self.round_progress.increment() 

                        curr_family  = iumsutils.get_family(evaluand)
                        header = (local_fam_training and fam_str or evaluand) # label unfamiliar trainings with the current evaluand and familiar trainings as simply "familiar"
                        evals, non_evals = iumsutils.partition(chem_data, condition=lambda inst : inst.species == evaluand) # partition the dataset into instances belonging and not belonging to the current evaluand species
                        self.set_evaluand(f'{evaluand} ({len(evals)} instances found)')
                        
                        eval_spectra = [instance.spectrum[lower_bound:upper_bound] for instance in evals]
                        eval_titles  = [instance.name for instance in evals]
                        
                        # TRAIN/TEST SET CREATION, MODEL CREATION, AND MODEL TRAINING
                        if not local_fam_training or evaluand_idx == 0: # train over all evaluands for unfamiliars, but only first evaluand for familiars (training set is same for familiars)
                            self.epoch_progress.set_max(self.num_epochs) # reset maximum number of epochs on progress bar to specified amount (done only during training rounds)
                            
                            trainees = (local_fam_training and chem_data or non_evals) # train over all instances for familiars, or non-evaluands instances for unfamiliars 
                            features = np.array([instance.spectrum[lower_bound:upper_bound] for instance in trainees]) # extract features and labels, casting them to np arrays for training
                            labels   = np.array([instance.vector for instance in trainees]) 
                            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_set_proportion) # split features and labels into train and test sets based on the predefined proportion

                            self.loss_plot.reset(cutoff=(self.convergence and self.tolerance or None)) # resetting the loss plot each training round
                            with tf.device('CPU:0'):     # eschews the requirement for a brand-new NVIDIA graphics card (which we don't have anyways)                      
                                model = Sequential()     # model block is created, layers are added to this block
                                if self.nw_compat:            # provide option to switch between NeuralWare model configuration (for comparison) or optimized standard configuration
                                    model.add(Dense(150, input_dim=(upper_bound - lower_bound), activation='tanh'))  # 512 neuron input layer, size depends on trimming
                                    model.add(Dense(len(self.mapping), activation='softmax')) # softmax gives probability distribution of identity over all families
                                    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.learnrate), metrics=[RootMeanSquaredError(name='rmse')]) 
                                else:   
                                    model.add(Dense(256, input_dim=(upper_bound - lower_bound), activation='relu'))  # 256 neuron input layer, size depends on trimming      
                                    model.add(Dropout(0.5))                                   # dropout layer, to reduce overfit
                                    model.add(Dense(256, activation='relu'))                  # 256 neuron hidden layer
                                    model.add(Dense(len(self.mapping), activation='softmax')) # softmax gives probability distribution of identity over all families
                                    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learnrate), metrics=[CategoricalAccuracy(name='cat_acc')])      
                            hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=self.keras_callbacks, verbose=keras_verbosity, epochs=self.num_epochs, batch_size=self.batchsize)
                        
                        # CHECKS TO DETERMINE WHETHER TO ABORT TRAINING, STOP TRAINING DUE TO CONVERGENCE, OR PROCEED AS PLANNED
                        if self.abort_flag: # check if training has been aborted by user as well
                            self.abort()
                            return # without a return, aborting training only pauses one iteration of loop                      
                        else:
                            if model.stop_training and self.convergence: # check if training stopped due to convergence being reached. 
                                self.epoch_progress.set_max(self.convergence_callback.stopped_epoch) # set progress max to stopped epoch, to indicate that training is done   
                                with log_file_path.open(mode='a') as log_file: 
                                    log_file.write(f'{header} training round stopped early at epoch {self.convergence_callback.stopped_epoch}\n')
                                self.convergence_callback.reset_flag() # reset stopped epoch to None for next training round 

                            if self.save_weights: # otherwise, save the model to the current result directory if saving is enabled
                                self.set_status('Saving Model...')
                                weights_folder = curr_slice_folder/f'{header} Model Files'
                                model.save(str(weights_folder)) # path can only be str, for some reason

                                local_preset = {param : value for param, value in self.parameters.items()} # make a deep copy of the parameters that can be modified locally
                                local_preset['data_files'  ] = [data_file]  # only retrain using the current file
                                local_preset['fam_training'] = local_fam_training # to account for current familiar or unfamiliar status
                                local_preset['cycle_fams'  ] = False        # ensure no cycling is performed for reproduction
                                local_preset['read_mode'   ] = 'By Species' # ensure only species-wise reading is being performed
                                local_preset['selections'  ] = [evaluand]   # only requires a single evaluand (and in the case of familiars, the particular one is not at all relevant)
                                local_preset['num_slices'  ] = 0            # no slices, only full data 
                                with open(weights_folder/f'Reproducability Preset ({point_range}).json', 'w') as weights_preset: 
                                    json.dump(local_preset, weights_preset)  # add a preset to each model file that allows for reproduction of ONLY that single model file                      

                            # CREATION OF THE SPECIES SUMMARY OBJECT FOR THE CURRENT EVALUAND, PLOTTING OF EVALUATION RESULTS FOR THE CURRENT EVALUAND 
                            self.set_status('Plotting Results to Folder...')
                            for inst_name, pred in zip(eval_titles, model.predict(np.array(eval_spectra))): # file predictions into dict
                                predictions[curr_family][evaluand][inst_name] = [float(i) for i in pred] # convert from np array of numpy float to list of floats for JSON serialization 

                            # obtain and file current evaluand score, plotting summary graphics in the process; overwrites empty dictionary
                            final_evals = model.evaluate(x_test, y_test, verbose=keras_verbosity)  # keras' self evaluation of loss and metrics - !NOTE! - resets model.stop_training flag
                            loss, metric = hist.history['loss'], hist.history[self.nw_compat and 'rmse' or 'cat_acc'] # get loss and metric history from the training run 
                            round_summary[curr_family][evaluand] = plotutils.plot_and_get_score(evaluand, eval_spectra, predictions, loss, metric, final_evals, savedir=curr_slice_folder)   
                            losses[header] = loss # add current loss to dict for the current training round                         
                                                     
                    # CALCULATION AND PROCESSING OF RESULTS TO PRODUCE SCORES AND SUMMARIES 
                    self.set_status(f'Unpacking Scores...')     
                    with loss_file_path.open('w') as loss_file: # write losses for trainings to file
                        json.dump(losses, loss_file)
                    
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
                    self.set_status(f'Outputting Summaries of Results...')
                    
                    prediction_path = curr_slice_folder/'Prediction Values.json'
                    with prediction_path.open(mode='w', newline='') as pred_file:
                        json.dump(predictions, pred_file) # save the aavs for the current training

                    plotutils.single_plot(plotutils.Overlaid_Family_RC(predictions, title=f'{data_file} Overall Summary'), curr_slice_folder/'Overall Summary', figsize=8) 
                           
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
                    log_file.write(f'\nTraining Time : {iumsutils.format_time(time.time() - start_time)}') # log the time taken for this particular training session as well
        
        # POST-TRAINING WRAPPING-UP   
        if self.cycle_fams: # if both familiar and unfamiliar traning are being performed, merge the score files together
            for point_range in point_ranges:
                fam_path   = parent_folder/f'Compiled Results - {point_range}, Familiar.csv' 
                unfam_path = parent_folder/f'Compiled Results - {point_range}, Unfamiliar.csv'
                
                with unfam_path.open('r') as unfam_file, fam_path.open('a') as fam_file: # read from unfamiliar scores and append to familiar scores
                    for row in unfam_file:
                        fam_file.write(row)  
                        
                unfam_path.unlink() # delete the unfamiliar file after copying the contents
                fam_path.rename(fam_path.parent/f'Compiled Results - {point_range}.csv' )  # get rid of "Familiar" affix after merging
        
        self.button_frame.enable()  # open up post-training options in the training window
        self.abort_button.configure(state='disabled') # disable the abort button (idiotproofing against its use after training)
        
        self.set_status(f'Finished in {iumsutils.format_time(time.time() - overall_start_time)}') # display the time taken for all trainings in the training window
        if messagebox.askyesno('Training Completed Succesfully!', 'Training finished; view training results in folder?', parent=self.training_window):
            os.startfile(parent_folder) # notify user that training has finished and prompt them to view the results in situ