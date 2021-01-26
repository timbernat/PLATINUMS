import cmath
import numpy as np
from iumsutils import *
import matplotlib.pyplot as plt


class Multiplot:
    '''Base class for creating easily referenceable objects to subplot into. Effectively a wrapper for plt.subplots'''
    def __init__(self, nrows=None, ncols=None, span=None, figsize=5):
        if not (nrows or ncols):
            raise ValueError('At least one dimension is needed to specify a multiplot')
        elif bool(nrows) ^ bool(ncols): # if only one dimension is passed
            if not span:
                raise ValueError('Span is required when providing only one dimension')
            elif not nrows:
                nrows = ceildiv(span, ncols) # deduce required number of rows from number of columns
            elif not ncols:
                ncols = ceildiv(span, nrows) # deduce required number of columns from number of rows
        self.nrows, self.ncols = nrows, ncols
                
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(figsize*ncols, figsize*nrows)) # dimensions must be backwards to scale properly
        self.axes = np.array(self.axes).reshape(nrows, ncols) # ensure that axes object has two dimensions, even in scalar/vector cases
        
    def draw(self, plot, index=(0,0)): 
        '''Wrapper for drawing plots in a more OOP-friendly? fashion. Relies upon the draw method for objects in this module being defined approriately'''
        if type(index) == int:
            index = divmod(index, self.ncols) # allow for linear indexing based on size        
        plot.draw(self.axes, index) #!!CRITICAL!! - prereq for all objects which follow is that they have a draw method which accepts and Axes object and an index
        
    def draw_series(self, plot_set):
        '''Used to plot an iterable of plot objects; leverages linear indexing capability'''
        for i, plot in enumerate(plot_set):
            self.draw(plot, index=i)
        
    def save(self, file_name):
        '''Wrapper for saving Multiplots'''
        self.fig.savefig(file_name)

        
# Radar Chart classes
class Mapped_Unit_Circle:
    '''Base background unit circle class used to create Radar Charts'''
    def __init__(self, mapping):
        theta = np.linspace(0, cmath.tau, 100)
        self.circle = (np.cos(theta), np.sin(theta))  # generic black unit circle, preset for all SD objects
        
        self.N = len(mapping) 
        self.labels = tuple(mapping.keys())
        self.poles  = [cmath.exp(1j* i*cmath.tau/self.N) for i in range(self.N)] # poles at the Nth roots of unity
        
    def draw(self, ax):
        ax.plot(*self.circle, 'k-')
        for i, (label, pole) in enumerate(zip(self.labels, self.poles)): # pass the poles and axes to the internal unit circle
            x, y = pole.real, pole.imag # unpack components f values
            ax.plot(x, y, 'ro')        # plot the roots of unity
            ax.plot([0, x], [0, y], 'y--')  # plot radial lines to each root
            ax.annotate(label, (x, y), ha='center') # label each root with the associated family, center horizontally

class Base_RC:
    '''Base Radar Chart class. Builds unit circle based on species mapping, can plot set of point along with unreduced centroid'''     
    unit_circle = None # !!CRITICAL!! - mapping for class must be set prior to use
    
    def __init__(self, title, points, point_marker='gx', centroid_marker='b*'):
        self.title = title
        self.points = points
        
        self.point_marker = point_marker
        self.centroid = sum(self.points)/len(points)
        self.centroid_marker = centroid_marker
            
    def plot_point(self, coords, ax, marker='ro'):
        if type(coords) != tuple:
            coords = (coords.real, coords.imag)
        ax.plot(*coords, marker) 
        
    def draw(self, axes, index=(0,0)):
        ax = axes[index] # index subplots within the passed plt.Axes object
        ax.set_title(self.title)
        
        if not ax.lines: # only draw a circle if one isn't already there; for overlay purposes. NOTE TO SELF: check if better param than "lines" exists
            self.unit_circle.draw(ax) # plot the unit circle background
        
        for point in self.points:
            self.plot_point(point, ax, marker=self.point_marker)
            
        if self.centroid_marker: # only plot the centroid if it is called for
            self.plot_point(self.centroid, ax, marker=self.centroid_marker)
            
class Instance_RC(Base_RC):
    '''0-order Radar Chart class for plotting the axial components and single centroid of a single instance'''
    def __init__(self, dataset, inst_name, point_marker='gx', centroid_marker='b1'):
        aavs = dataset[get_family(inst_name)][isolate_species(inst_name)][inst_name] # perform the appropriate lookup for the species
        axial_points = [aav*pole for (aav, pole) in zip(aavs, Base_RC.unit_circle.poles)] # multiply aavs by axial conponents to obtain set of points
        
        super().__init__(inst_name, axial_points, point_marker, centroid_marker)
        self.centroid *= Base_RC.unit_circle.N # scale centroid by number of points in this case to better adhere to unit circle
        
class Species_RC(Base_RC):
    '''1-order Radar Chart class for plotting centroid of all instances of a species'''
    def __init__(self, dataset, species, point_marker='b1', centroid_marker='m*'):
        inst_centroids = [Instance_RC(dataset, instance).centroid for instance in dataset[get_family(species)][species]]
        super().__init__(species, inst_centroids, point_marker, centroid_marker)
        
class Family_RC(Base_RC):
    '''2-order Radar Chart class for plotting centroid of all instances of a family'''
    def __init__(self, dataset, family, point_marker='m*', centroid_marker='cs'):
        spec_centroids = [Species_RC(dataset, species).centroid for species in dataset[family]]
        super().__init__(family, spec_centroids, point_marker, centroid_marker)
        
class Overlaid_Family_RC(Base_RC):
    '''2.5-order Radar Chart class for plotting all families on a single diagram, color-coded '''
    marker_map  = {
        'Acetates': 'g^',
        'Alcohols': 'r^',
        'Aldehydes': 'y^',
        'Ethers': 'b^',
        'Ketones': 'm^'
    }
    
    def __init__(self, dataset):
        self.famsds = [Family_RC(dataset, family, point_marker=self.marker_map[family]) for family in dataset]
    
    def draw(self, axes, index=(0,0)):
        ax = axes[index]
        for fsd in self.famsds:
            fsd.draw(axes, index) # draw over one another
            
        markers = [ax.scatter(np.nan, np.nan, color=color, marker=marker) for (color, marker) in self.marker_map.values()] # plot fake points for legend
        ax.legend(markers, self.marker_map.keys(), loc='lower right')

class Macro_RC(Base_RC):
    '''3-order Radar Chart from plotting trends across all data'''
    def __init__(self, dataset, point_marker='cs', centroid_marker='gp'):
        fam_centroids = [Family_RC(dataset, family).centroid for family in dataset]
        super().__init__('All Families', fam_centroids, point_marker, centroid_marker) 
       
    
# Line Plot classes
class Line_Plot:
    '''Basic class of line plot, allows for multiple lines and moveable legends, within the confines of the Mutiplot Framework'''
    def __init__(self, *args, title=None, legend_pos=None, colormap={}):
        self.lines = args
        self.legend_pos = legend_pos
        self.colormap = colormap
        self.title = title
        
    def draw(self, axes, index=(0,0)):
        ax = axes[index]
        ax.set_title(self.title)
        
        for line, (label, color) in zip(self.lines, self.colormap.items()):
            ax.plot(line, color, label=label)
            
        if self.legend_pos:
            ax.legend(loc=self.legend_pos)
            
class Metric_Plot(Line_Plot):
    '''Subclass of Line_Plot for plotting training metrics'''
    colormap = {'Loss' : 'r', 'Accuracy' : 'g'}
    
    def __init__(self, losses, accuracies, evals):
        self.title = f'Loss, Acc={[round(i, 3) for i in evals]}'
        super().__init__(losses, accuracies, title=self.title, legend_pos='center right', colormap=self.colormap)
        
class PWA_Plot(Line_Plot):
    '''Point-Wise Aggregate plot generation class'''
    colormap={
        'Maxima' : 'b',
        'Averages' : 'r',
        'Minima' : 'g'
    }
    
    def __init__(self, spectra, species):
        self.spectra  = np.array(spectra)
        self.maxima   = np.amax(self.spectra, axis=0)
        self.averages = np.average(self.spectra, axis=0)
        self.minima   = np.amin(self.spectra, axis=0)
        super().__init__(self.maxima, self.averages, self.minima, title=species, legend_pos='upper right', colormap=self.colormap)
        
        
# Bar Chart classes
class Multibar: # this will serve as the base bar class, as ordinary bar graphs can be viewed as a special case of Multibars
    '''Class for plotting one or more bar graphs along the same main group labels. Takes the names of the group labels,
    the names of the sub-bar plots within each group, and the datasets (arbitrarily many), along with other parameters'''
    colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k']
    bar_group_width = 0.8 # this should be less than 1 to prevent barset overlap
    
    def __init__(self, group_labels, sub_labels, *datasets, title=None, ylim=None, legend_pos=None):
        if len(datasets) != len(sub_labels):
            raise ValueError('Number of datasets must match number of sub labels')
         
        self.datasets     = datasets # unpacked, to allow for arbitrarily many (or few) inputs
        self.group_labels = group_labels
        self.sub_labels   = sub_labels
        self.n_bar_groups = len(group_labels)
        self.n_per_group  = len(sub_labels)
        
        self.bar_width = self.bar_group_width/self.n_per_group # width of an individual bar within a group
        self.x = np.arange(self.n_bar_groups)
        
        self.title = title
        self.ylim  = ylim
        self.legend_pos = legend_pos

    def draw(self, axes, index=(0,0)):
        ax = axes[index]

        ax.set_title(self.title)
        ax.set_xticks(self.x)
        ax.set_xticklabels(self.group_labels)

        for i, dataset in enumerate(self.datasets):
            offset = self.bar_width*(i - (self.n_per_group-1)/2) # amount to offset each set of bars from unit ticks
            ax.bar(self.x + offset, dataset, width=self.bar_width, color=self.colors[i], align='center')

        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.legend_pos:
            ax.legend(self.sub_labels, loc=self.legend_pos)
            
class AAV_Bars(Multibar):
    '''class used for plotting bar charts of the AAVS of a particular species'''
    mapping = None # !!CRITICAL!! - mapping for class must be set prior to use
    
    def __init__(self, inst_name, aavs): # NOTE TO SELF: the "inst_name" sub_label is really just a placeholder, consider inplementing more cleanly going forward
        super().__init__(self.mapping.keys(), [inst_name], aavs, title=inst_name, ylim=(0,1))
