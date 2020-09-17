'''Some Custom widget classes I've defined to make working with tkinter a bit more palatable.
Widgets, in the order they appear, are ConfirmButton, StatusBox, DynOptionMenu, ToggleFrame, 
NumberedProgBar, LabelledEntry, Switch, GroupableCheck, CheckPanel, and SelectionWindow'''
import tkinter as tk
import tkinter.ttk as ttk
import math # needed for ceiling function


class ConfirmButton: 
    '''A basic confirmation button, will execute whatever function is passed to it
    when pressed. Be sure to exclude parenthesis when passing the bound functions'''
    def __init__(self, frame, funct, padx=5, row=0, col=0, cs=1, sticky=None):
        self.button =tk.Button(frame, text='Confirm Selection', command=funct, padx=padx)
        self.button.grid(row=row, column=col, columnspan=cs, sticky=sticky)
        
        
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
        elif status:
            self.status_box.configure(bg='green2', text=self.on_message)
        else:
            self.status_box.configure(bg='light gray', text=self.off_message)
            
            
class DynOptionMenu:
    '''My addon to the TKinter OptionMenu, adds methods to conveniently update menu contents'''
    def __init__(self, frame, var, option_method, opargs=None, default=None, width=10, row=0, col=0, colspan=1):
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
    '''Progress bar which displays the numerical proportion complete (out of the set total) in the middle of the bar'''
    def __init__(self, frame, total, default=0, style_num=1, length=240, row=0, col=0, cs=1):
        self.curr_val = None
        self.default = default
        self.total = total
        self.style = ttk.Style(frame)
        
        self.style_name = f'NumberedProgBar{style_num}'
        self.style.layout(self.style_name, 
             [('Horizontal.Progressbar.trough', {'children': [('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})],
                                                 'sticky': 'nswe'}),
              ('Horizontal.Progressbar.label', {'sticky': ''})]) 
        
        self.prog_bar = ttk.Progressbar(frame, style=self.style_name, orient='horizontal', length=length, maximum=total)
        self.prog_bar.grid(row=row, column=col, columnspan=cs)
        self.reset()
        
    def set_progress(self, val):
        if val > self.total:
            raise ValueError # ensure that the progressbar is not set betond the total
        else:
            self.curr_val = val
            self.prog_bar.configure(value=self.curr_val)
            self.style.configure(self.style_name, text=f'{self.curr_val}/{self.total}')
        
    def increment(self):
        if self.curr_val == self.total:
            return # don't increment when full
        else:
            self.set_progress(self.curr_val+1) 
            
    def reset(self):
        self.set_progress(self.default)


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
    '''An interactive switch button, clicking inverts the boolean state and status display. State can be accessed
    via the <self>.state() method or with the <self>.var.get() attribute to use dynamically with tkinter'''
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
    '''A checkbutton which will add to or remove its value from an output list
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
    '''A panel of GroupableChecks, allows for simple selectivity of the contents of some list. 
    Behaves like RadioButtons, except selection of multiple (or even all) buttons is allowedd'''
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
    '''The window used in -IUMS programs to select species for evaluation'''
    def __init__(self, main, parent_frame, size, selections, output, window_title='Select Members to Include', ncols=1):
        self.window = tk.Toplevel(main)
        self.window.title(window_title)
        self.window.geometry(size)
        self.parent = parent_frame
        self.parent.disable()
        
        self.panel = CheckPanel(self.window, selections, output, ncols=ncols)
        self.confirm = ConfirmButton(self.window, self.confirm, row=self.panel.row_span, col=ncols-1)

    def confirm(self):
        self.parent.enable()
        self.window.destroy()
