'''Some Custom widget classes I've defined to make working with tkinter a bit more palatable.
Widgets, in the order they appear, are ConfirmButton, StatusBox, DynOptionMenu, ToggleFrame, 
NumberedProgBar, LabelledEntry, Switch, GroupableCheck, CheckPanel, and SelectionWindow'''
import tkinter as tk
import tkinter.ttk as ttk


class ConfirmButton: 
    '''A nice, basic blue confirmation button, will execute the function passed when pressed (remember to omit parenthesis!)'''
    def __init__(self, frame, command, padx=5, pady=1, underline=None, row=0, col=0, cs=1, color='deepskyblue2', sticky='e'):
        self.button = tk.Button(frame, text='Confirm Selection', command=command, bg=color, underline=underline, padx=padx, pady=pady)
        self.button.grid(row=row, column=col, columnspan=cs, sticky=sticky)
        
        
class StatusBox:
    '''A simple label which changes color and gives indictation of the status of something'''
    def __init__(self, frame, on_message='On', off_message='Off', default_status=False, width=17, padx=0, row=0, col=0):
        self.on_message = on_message
        self.off_message = off_message
        
        self.status_box = tk.Label(frame, width=width, padx=padx)
        self.status_box.grid(row=row, column=col)
        
        self.default = default_status
        self.reset_default()
        
    def set_status(self, status):
        if type(status) != bool:
            raise Exception(TypeError)    
        elif status:
            self.status_box.configure(bg='green2', text=self.on_message)
        else:
            self.status_box.configure(bg='light gray', text=self.off_message)
            
    def reset_default(self):
        self.set_status(self.default)
            
            
class DynOptionMenu:
    '''My addon to the TKinter OptionMenu, adds methods to conveniently update menu contents'''
    def __init__(self, frame, var, option_method, opargs=(), default=None, width=10, row=0, col=0, colspan=1):
        self.option_method = option_method
        self.opargs = opargs # any additional arguments that need to be passed to the option-getting method
        self.default = default
        self.menu = tk.OptionMenu(frame, var, (None,) )
        self.menu.configure(width=width)
        self.menu.grid(row=row, column=col, columnspan=colspan)
        
        self.var = var
        self.contents = self.menu.children['menu']
        self.update()
        
    def enable(self):
        self.menu.configure(state='normal')
        
    def disable(self):
        self.menu.configure(state='disabled')
    
    def reset_default(self):
        self.var.set(self.default)
    
    def update(self):
        self.contents.delete(0, 'end')
        for option in self.option_method(*self.opargs):
            self.contents.add_command(label=option, command=lambda x=option: self.var.set(x))
        self.reset_default()
        
class NumberedProgBar():
    '''Progress bar which displays the numerical proportion complete (out of the set maximum) in the middle of the bar'''
    def __init__(self, frame, maximum=100, default=0, style_num=1, length=260, row=0, col=0, cs=1):
        self.curr_val = None
        self.default = default
        self.maximum = maximum
        self.style = ttk.Style(frame)
        
        self.style_name = f'NumberedProgBar{style_num}'
        self.style.layout(self.style_name, 
             [('Horizontal.Progressbar.trough', {'children': [('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})],
                                                 'sticky': 'nswe'}),
              ('Horizontal.Progressbar.label', {'sticky': ''})]) 
        
        self.prog_bar = ttk.Progressbar(frame, style=self.style_name, orient='horizontal', length=length, maximum=maximum)
        self.prog_bar.grid(row=row, column=col, columnspan=cs)
        self.reset()
        
    def configure(self, **kwargs): # wrapper for tkinter "configure()" method
        self.prog_bar.configure(**kwargs)
            
    def reset(self):
        self.set_progress(self.default)
        
    def set_progress(self, val):
        if val > self.maximum:
            raise ValueError('Current progress value exceeds maximum') # ensure that the progressbar is not set beyond the maximum
        else:
            self.curr_val = val
            self.configure(value=self.curr_val)
            self.style.configure(self.style_name, text=f'{self.curr_val}/{self.maximum}')
            
    def set_max(self, new_max):
        '''change the maximum value of the progress bar (including the label)'''
        self.maximum = new_max
        self.configure(maximum = new_max)
        self.set_progress(self.curr_val) # ensures that the display updates without incrementing the count, and performs max value check to boot
        
    def increment(self):
        if self.curr_val == self.maximum:
            return # don't increment when full
        else:
            self.set_progress(self.curr_val+1) 


class ToggleFrame(tk.LabelFrame):
    '''A frame whose contents can be easily disabled or enabled, If starting disabled, must put "self.disable()"
    AFTER all widgets have been added to the frame'''
    def __init__(self, window, text, default_state='normal', padx=5, pady=5, row=0, col=0):
        tk.LabelFrame.__init__(self, window, text=text, padx=padx, pady=pady, bd=2, relief='groove')
        self.grid(row=row, column=col)
        
        self.main = window
        self.state = default_state
        self.apply_state(default_state)
    
    def apply_state(self, new_state):
        self.state = new_state
        for widget in self.winfo_children():
            try:
                widget.configure(state = self.state)
            except tk.TclError: # tkinter.ttk widgets are enabled and disabled completely differently, for no good reason
                state = ((self.state == 'normal') and '!disabled' or self.state)
                widget.state([state])
            
    def enable(self):
        self.apply_state('normal')
     
    def disable(self):
        self.apply_state('disabled')
    
    def toggle(self):
        if self.state == 'normal':
            self.disable()
        else:
            self.enable()
      
    
class LabelledEntry:
    '''An entry with an adjacent label to the right. Use "self.get_value()" method to retrieve state of
    variable. Be sure to leave two columns worth of space for this widget'''
    def __init__(self, frame, text, var, state='normal', default=None, underline=None, width=10, spacing=(0,0), row=0, col=0):           
        self.var = var
        
        left, right = spacing # unpack padding to be placed on either side of the object
        self.label = tk.Label(frame, text=text, padx=2, underline=underline, state=state)
        self.entry = tk.Entry(frame, width=width, textvariable=self.var, state=state, validate='all', validatecommand=((frame.main.register(self.callback)), '%P'))
        
        self.label.grid(row=row, column=col,   sticky='w', padx=(left, 0))  # entry and label will always touch, regardless of other parameters
        self.entry.grid(row=row, column=col+1, sticky='w', padx=(0, right)) # entry occupies column adjacent to label
        
        self.default = default
        self.reset_default()
    
    def get_value(self):
        return self.var.get()
    
    def set_value(self, value):
        self.var.set(value)
    
    def reset_default(self):
        self.var.set(self.default)
        
    def callback(self, P): # restricts input to integers
        return (str.isdigit(P) or P == '')
    
    def configure(self, **kwargs):   # allows for disabling in ToggleFrames
        self.label.configure(**kwargs)
        self.entry.configure(**kwargs)
        
    
class Switch: 
    '''An interactive switch button, clicking inverts the boolean state and status display. State can be accessed via the <self>.value attribute'''
    def __init__(self, frame, text, default=False, dep_state='normal', dependents=None, underline=None, width=10, 
                 on_text='Enabled', on_color='green2', off_color='red', off_text='Disabled', row=0, col=0):
        self.label = tk.Label(frame, text=text, underline=underline)
        self.label.grid(row=row, column=col)
        self.switch = tk.Button(frame, width=width, command=self.toggle)
        self.switch.grid(row=row, column=col+1)
        
        self.on_text = on_text
        self.on_color = on_color
        self.off_text = off_text
        self.off_color = off_color
    
        self.dependents = dependents
        self.dep_state = dep_state
        self.value = None
        self.default = default
        self.reset_default()
    
    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value
        state, text, color = self.value and ('normal', self.on_text, self.on_color) or ('disabled', self.off_text, self.off_color)
        
        self.switch.configure(text=text, bg=color)
        self.dep_state = state
        if self.dependents:
            for widget in self.dependents:
                widget.configure(state=self.dep_state)
                
    def reset_default(self):
        self.set_value(self.default)
                
    def enable(self):
        self.set_value(True)
     
    def disable(self):
        self.set_value(False)
    
    def toggle(self):
        if self.value:
            self.disable()
        else:
            self.enable()  
           
        
class GroupableCheck:
    '''A checkbutton which will add to or remove its value from an output list
    (passed as an argument when creating an instance) based on its check status'''
    def __init__(self, frame, value, output, state='normal', row=0, col=0):
        self.var = tk.StringVar()
        self.value = value
        self.output = output
        self.state = state
        
        self.cb = tk.Checkbutton(frame, text=value, variable=self.var, onvalue=self.value, offvalue=None, state=self.state, command=self.edit_output) 
        self.cb.grid(row=row, column=col, sticky='w')
        self.cb.deselect()
        
    def edit_output(self):
        if self.var.get() == self.value:
            self.output.append(self.value)
        else:
            self.output.remove(self.value)
            
    def configure(self, **kwargs):
        self.cb.configure(**kwargs)

ceildiv = lambda a, b : -(a // -b) # ceiling analogue of floor division, useful for scaling grids by span

class CheckPanel:
    '''A panel of GroupableChecks, allows for simple selectivity of the contents of some list. 
    Behaves like RadioButtons, except selection of multiple (or even all) buttons is allowedd'''
    def __init__(self, frame, data, output, default_state='normal', ncols=4, row_start=0, col_start=0):
        self.output   = output
        self.state    = default_state
        self.row_span = ceildiv(len(data), ncols)
        
        self.panel = [GroupableCheck(frame, val, output, state=self.state, row=row_start + i//ncols, col=col_start + i%ncols) for i, val in enumerate(data)]
        
    def wipe_output(self):
        self.output.clear()
        
    def apply_state(self, new_state):
        self.state = new_state
        for gc in self.panel:
            gc.configure(state=self.state)
    
    def enable(self):
        self.apply_state('normal')
     
    def disable(self):
        self.apply_state('disabled')
    
    def toggle(self):
        if self.state == 'normal':
            self.disable()
        else:
            self.enable()        
        
class SelectionWindow:
    '''The window used in -IUMS programs to select species for evaluation'''
    def __init__(self, main, parent_frame, selections, output, window_title='Select Members to Include', ncols=1):
        self.window = tk.Toplevel(main)
        self.window.title(window_title)
        
        self.parent = parent_frame     
        self.panel  = CheckPanel(self.window, selections, output, ncols=ncols)
        
        self.confirm_button    = tk.Button(self.window, text='Confirm Selection', command=self.confirm, bg='deepskyblue2', underline=0, padx=5)
        self.choose_all_button = tk.Button(self.window, text='Toggle All', command=self.toggle_all, underline=7, padx=15)
        
        self.confirm_button.grid(   row=self.panel.row_span, column=ncols-1, sticky='nesw', padx=2, pady=2)
        self.choose_all_button.grid(row=self.panel.row_span, column=ncols-2, sticky='nesw', pady=2)
        
        self.window.bind('c', lambda event : self.confirm())    # bind the confirmation command to the 'c' key
        self.window.bind('a', lambda event : self.toggle_all()) # bind the select all option to the 'a' key
        
        self.parent.disable() # disable the parent to ensure no cross-binding occurs

    def toggle_all(self):
        '''Invert state of all checks in panel, making appropriate list changes in the process'''
        for option in self.panel.panel:
            option.cb.invoke()
        
    def confirm(self):
        '''Confirm selections and exit selection window'''
        self.parent.enable()
        self.window.destroy()
