# Custom Imports
import iumsutils           # library of functions specific to my "-IUMS" class of MS Neural Network applications
import TimTkLib as ttl     # library of custom tkinter widgets I've written to make GUI assembly more straightforward 
from PLATINUMS_Trainer import TrainingWindow # the trainer component of the app

# Built-in Imports    
import json
from pathlib import Path 

# Built-in GUI Imports 
import tkinter as tk   
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog


class PLATINUMS_App:
    '''PLATINUMS : Prediction, Training, And Labelling INterface for Unlabelled Mobility Spectra'''
    default_paths = {'data_path'    : Path('TDMS Datasets'),  # configure names of default file locations here
                     'preset_path'  : Path('Training Presets'),
                     'save_path'    : Path('Saved Training Results'),
                     'weights_path' : Path('Saved Models')}
    
    def __init__(self, main):       
    #Main Window
        self.main = main
        self.main.title('PLATINUMS v-7.0.0a') # set name with version here
        self.parameters = {}
        
        for path_type, path in self.default_paths.items(): # on creation, reference class-wide default paths to set up appropriate folders
            path.mkdir(exist_ok=True) # create the path
            setattr(self, path_type, path) # map said path to an internal object attribute

        self.tpmode = tk.BooleanVar() # option to switch from training to prediction mode, WIP and disabled for now
        
        self.quit_button   = tk.Button(self.main, text='Quit', underline=0, padx=22, pady=11, bg='red', command=self.quit)
        self.tpmode_button = tk.Checkbutton(self.main, text='Predict', var=self.tpmode, underline=2, command=self.switch_tpmode, state='disabled')
        self.reset_button  = tk.Button(self.main, text='Reset', underline=0, padx=20, bg='orange', command=self.reset)
        
        self.quit_button.grid(  row=0, column=4, sticky='s')
        self.tpmode_button.grid(row=2, column=4)
        self.reset_button.grid( row=6, column=4, padx=2)
        
        # Primary menu method - universally keybound, irrespective of which cell is enabled
        self.main.bind('q', lambda event : self.quit()) 
        self.main.bind('e', lambda event : self.tpmode_button.invoke()) # doesn't work while button is disabled, due to invoke (not by accident)
        self.main.bind('r', lambda event : self.reset())  
        
    # Frame 0
        self.spectrum_size  = 0  # initialize empty variables for various data attributes
        self.species        = []
        self.families       = []
        self.evaluands      = []
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
            tk.Radiobutton(self.selection_frame, text=mode, value=mode, var=self.read_mode, underline=underline, command=self.evaluand_selection).grid(row=0, column=i)
        
        self.confirm_sels = tk.Button(self.selection_frame, text='Confirm Selection', command=self.confirm_evaluand_selection, bg='deepskyblue2', underline=0, padx=4)
        self.confirm_sels.grid(row=0, column=3)
        
    #Frame 2
        self.hyperparam_frame    = ttl.ToggleFrame(self.main, text='Set Hyperparameters: ', padx=8, row=3)
        
        self.epoch_entry         = ttl.LabelledEntry(self.hyperparam_frame, text='Epochs:',    var=tk.IntVar(),    default=2048, width=16, row=0, col=0)
        self.batchsize_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Batchsize:', var=tk.IntVar(),    default=32,   width=16, row=1, col=0)
        self.learnrate_entry     = ttl.LabelledEntry(self.hyperparam_frame, text='Learnrate:', var=tk.DoubleVar(), default=2e-5, width=18, row=0, col=2, spacing=(10,0))
        
        self.confirm_hyperparams = tk.Button(self.hyperparam_frame, text='Confirm Selection', command=self.confirm_hp, padx=5, bg='deepskyblue2', underline=0)
        self.confirm_hyperparams.grid(row=1, column=3, columnspan=2, sticky='e')
        
    #Frame 3
        self.slicing_frame       = ttl.ToggleFrame(self.main, text='Set Slicing Parameters: ', padx=7, row=4)  
        
        self.n_slice_entry       = ttl.LabelledEntry(self.slicing_frame, text='Slices:',    var=tk.IntVar(), default=0,  width=18, row=0, col=0)
        self.lower_bound_entry   = ttl.LabelledEntry(self.slicing_frame, text='Bottom:',    var=tk.IntVar(), default=0,  width=18, row=1, col=0)
        self.slice_dec_entry     = ttl.LabelledEntry(self.slicing_frame, text='Decrement:', var=tk.IntVar(), default=10, width=18, row=0, col=2, spacing=(0, 0)) 
        
        self.confirm_sliceparams = tk.Button(self.slicing_frame, text='Confirm Selection', command=self.confirm_sparams, padx=5, bg='deepskyblue2', underline=0)
        self.confirm_sliceparams.grid(row=1, column=3, columnspan=2, sticky='e')
        
    #Frame 4
        self.param_frame  = ttl.ToggleFrame(self.main, text='Set Training Parameters:', padx=6, row=5)
        
        self.tolerance_entry = ttl.LabelledEntry(self.param_frame, text='Tolerance:', var=tk.DoubleVar(), default=0.01, width=16, row=2, col=2, spacing=(6, 0))      
        self.fam_switch   = ttl.Switch(self.param_frame, text='Familiar Training:',  underline=0, row=0, col=0)
        self.cycle_switch = ttl.Switch(self.param_frame, text='Cycle Familiars: ',   underline=1, row=1, col=0)
        self.convergence_switch  = ttl.Switch(self.param_frame, text='Convergence:', underline=3, row=2, col=0, dependents=(self.tolerance_entry,))
        self.save_switch  = ttl.Switch(self.param_frame, text='Save Weights:',       underline=5, row=3, col=0) 
        
        self.blank_var   = tk.BooleanVar()
        self.save_preset = tk.BooleanVar()     
 
        self.blank_option_button = tk.Checkbutton(self.param_frame, text=' '*21, var=self.blank_var, padx=5) # leaving room for future expnadability
        self.save_preset_button   = tk.Checkbutton(self.param_frame, text='Save Preset?', var=self.save_preset, underline=0, padx=5)  
        
        self.blank_option_button.grid(row=0, column=2, columnspan=2, sticky='e')
        self.save_preset_button.grid( row=1, column=2, columnspan=2, sticky='e')
        
        self.confirm_train_params = tk.Button(self.param_frame, text='Confirm Selection', command=self.confirm_tparams, bg='deepskyblue2', underline=0, padx=6)
        self.confirm_train_params.grid(row=3, column=2, columnspan=2, sticky='e')

    # Frame 5 - contains only the button used to trigger a main action  
        self.activation_frame = ttl.ToggleFrame(self.main, text='', padx=0, pady=0, row=6)   
        
        self.train_button = tk.Button(self.activation_frame, text='TRAIN', padx=22, width=44, bg='deepskyblue2', underline=0, command=self.training)
        self.pred_button = tk.Button(self.activation_frame, text='PREDICT', padx=22, width=44, bg='mediumorchid2', state='disabled', command=lambda:None)
        
        self.train_button.grid(row=0, column=0) # overlap buttons, so they can displace one another
        self.pred_button.grid( row=0, column=0)
        
        self.switch_tpmode()
         
    # Packaging together some widgets and attributes, for ease of reference (also useful for self.reset() and self.isolate() methods)
        self.arrays      = (self.parameters, self.species, self.families, self.evaluands, self.family_mapping) 
        self.switch_vars = (self.read_mode, self.input_mode, self.blank_var, self.save_preset)
        self.frames      = (self.input_frame, self.selection_frame, self.hyperparam_frame, self.slicing_frame, self.param_frame, self.activation_frame)

        self.tparam_mapping = {self.convergence_switch : 'convergence',
                               self.tolerance_entry : 'tolerance', # strictly an entry, not a switch, but makes code cleaner of placed here
                               self.fam_switch : 'fam_training',
                               self.cycle_switch : 'cycle_fams',
                               self.save_switch : 'save_weights'}
        self.hp_entry_mapping = {self.epoch_entry : 'num_epochs',
                                 self.batchsize_entry : 'batchsize',
                                 self.learnrate_entry : 'learnrate'} 
        self.slice_entry_mapping = {self.lower_bound_entry : 'trimming_min', 
                                    self.slice_dec_entry : 'slice_decrement',
                                    self.n_slice_entry : 'num_slices'}
        self.field_mapping = {**self.tparam_mapping, **self.hp_entry_mapping, **self.slice_entry_mapping} # merged internal dict of all field which can be filled out within the app
        
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
                self.evaluand_selection() # open further selection menu if not just selecting all
            elif event.char == 'c':
                self.confirm_evaluand_selection()
        elif self.hyperparam_frame.state == 'normal' and event.char == 'c': 
            self.confirm_hp()
        elif self.slicing_frame.state == 'normal' and event.char == 'c': 
            self.confirm_sparams()
        elif self.param_frame.state == 'normal':
            switch_mapping = {'f' : self.fam_switch,
                              'w' : self.save_switch,
                              'v' : self.convergence_switch,
                              'y' : self.cycle_switch}
            if event.char in switch_mapping:
                switch_mapping[event.char].toggle() # make parameter switches toggleable with keyboard
            elif event.char == 's': # add hotkey to save preset and proceed...
                self.save_preset.set(not self.save_preset.get()) # invert cycling status
            #elif event.char == 'n': # add hotkey to save preset and proceed...
            #    self.blank_var2.set(not self.blank_var2.get()) # invert status
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
        
        for field in self.field_mapping.keys():
            field.reset_default() # reset defaults for all switches and entries  
        
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
            except json.decoder.JSONDecodeError:        
                messagebox.showerror('Preset Format Error', 'Incorrect formatting in preset file;\n Please check preset for errors and try again')        
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
                        if not getattr(self, data_property): # check if the fields are empty, relies on attribute names being same as json keys - note lack of default attr
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
        if self.input_mode.get() not in ('Manual Input', 'Preset from File'):
            messagebox.showerror('No input mode selected!', 'Please choose either "Manual" or "Preset" input mode')
            self.reset()
        else:
            self.main.config(cursor='wait') # this make take a moment for large sets of files, indicate to user (via cursor) to be patient
            self.check_input_files()
            
            if self.input_mode.get() == 'Manual Input':                
                self.isolate(self.selection_frame)
            elif self.input_mode.get() == 'Preset from File': # could simply leave as "else", but is more explicit as is
                
                for field, param in self.field_mapping.items():
                    try:
                        field.set_value(self.parameters[param]) # configure all the switch values in the GUI
                    except KeyError as error:
                        messagebox.showerror('Preset Error', f'The parameter "{error}" is either missing or misnamed;\n Please check preset file for errors') 
                        break
                  
                self.read_mode.set(self.parameters['read_mode'])
                self.confirm_evaluand_selection() # assigns evaluands based on the given read mode and selections buffer
                    
                self.check_trimming_bounds() # ensure that bounds actually make sense  
                self.isolate(self.activation_frame)
            self.main.config(cursor='') # return cursor to normal
        
    # Frame 1 (Evaluand Selection) Methods
    def evaluand_selection(self): 
        '''logic for selection of evaluands to include in training, based on the chosen selection mode'''
        self.parameters['selections'] = [] # empties internal evaluand buffer when selection menu is re-clicked (ensures no overwriting or mixing, also serves as first-time list init)
        self.parameters['read_mode']  = self.read_mode.get() # record the read mode to parameters with each selection
        
        if self.read_mode.get() == 'By Species':
            ttl.SelectionWindow(self.main, self.selection_frame, self.species, self.parameters['selections'], window_title='Select Species to evaluate', ncols=8)
        elif self.read_mode.get() == 'By Family':
            ttl.SelectionWindow(self.main, self.selection_frame, self.families, self.parameters['selections'], window_title='Select Families to evaluate over', ncols=3)
        elif self.read_mode.get() == 'Select All':
            self.parameters['selections'].append('All Species') # add a flag to make clear that all species are being selected

    def confirm_evaluand_selection(self):
        '''Parsing and processing of the selections before proceeding'''
        if self.read_mode.get() == 'By Species':
            self.evaluands = self.parameters['selections']      
        elif self.read_mode.get() == 'By Family':  # pick out species by family if selection by family is made
            self.evaluands = [species for species in self.species if iumsutils.get_family(species) in self.parameters['selections']]  
        elif self.read_mode.get() == 'Select All':
            self.evaluands = self.species 
        
        if not self.evaluands: # if the evaluands is still empty (or doesn't exist), instruct the user to make a choice
            messagebox.showerror('No evaluands selected', 'Please ensure one or more species have been selected for evaluation')
        else: 
            self.isolate(self.hyperparam_frame)
            self.tolerance_entry.configure(state='disabled') # ensure tolerance dependent remains disabled
   
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
        self.convergence_switch.disable() # ensure convergence is disabled by default
     
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
        for field, param in self.tparam_mapping.items():
            self.parameters[param] = field.get_value() # configure parameters based on the switch values

        if self.save_preset.get(): # if the user has elected to save the current preset
            preset_path = Path( filedialog.asksaveasfilename(title='Save Preset to file', initialdir='./Training Presets', defaultextension='.json', filetypes=[('JSONs', '*.json')]) )
            try: 
                with open(preset_path, 'w') as preset_file:
                    json.dump(self.parameters, preset_file)
            except PermissionError: # catch the case in which the user cancels the preset saving
                return           
            
        self.isolate(self.activation_frame) # make the training button clickable if all goes well
             
    # Frame 5: the training routine itself, this is where the magic happens
    def training(self):
        '''Initialize a PLATINUMS_Trainer using the inputted parameters'''
        if self.parameters['convergence']:
            stop_str = f'{self.parameters["tolerance"]} Error'
        else:
            stop_str = f'{self.parameters["num_epochs"]} Epoch'
            
        if self.parameters["cycle_fams"]:
            training_type_str = 'cycled'
        else:
            training_type_str = (self.parameters["fam_training"] and 'familiar' or 'unfamiliar')
            
        parent_name = simpledialog.askstring(title='Set Result Folder Name', prompt='Please enter a name to save training results under:', initialvalue=f'{stop_str}, {training_type_str}') 
        if not parent_name: # if file determination returns empty due to user cancellation/omission, exit training routine
            return
        else:
            self.activation_frame.disable()
            train_window = TrainingWindow(self, parent_name, nw_compat=False) # pass the GUI app to the training window object to allow for menu manipulation     
            train_window.training(keras_verbosity=0)

if __name__ == '__main__':        
    main_window = tk.Tk()
    app = PLATINUMS_App(main_window)
    main_window.mainloop()
