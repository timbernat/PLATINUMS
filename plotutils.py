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
                
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(figsize*ncols, figsize*nrows)) # dimensions must be backwards to scale properly
        self.axs = np.array(self.axs).reshape(nrows, ncols) # ensure that axes object has two dimensions, even in scalar/vector cases
        
    def draw(self, plot, index=(0,0)): 
        '''Wrapper for drawing plots in a more OOP-friendly? fashion. Relies upon the draw method for objects in this module being defined approriately'''
        if type(index) == int:
            index = divmod(index, self.ncols) # allow for linear indexing based on size        
        plot.draw(self.axs, index) #!!CRITICAL!! - prereq for all objects which follow is that they have a draw method which accepts and Axes object and an index
        
    def draw_series(self, plot_set):
        '''Used to plot an iterable of plot objects; leverages linear indexing capability'''
        for i, plot in enumerate(plot_set):
            self.draw(plot, index=i)
        
    def save(self, file_name):
        '''Wrapper for saving Multiplots'''
        self.fig.savefig(file_name)

        
# Spoke Diagram classes
class Mapped_Unit_Circle:
    '''Base background unit circle class used to create spoke diagrams'''
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

class Base_SD:
    '''Base Spoke Diagram class. Builds unit circle based on species mapping, can plot set of point along with unreduced centroid'''     
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
            
class Instance_SD(Base_SD):
    '''0-order Spoke Diagram class for plotting the axial components and single centroid of a single instance'''
    def __init__(self, dataset, inst_name, point_marker='gx', centroid_marker='b1'):
        aavs = dataset[get_family(inst_name)][isolate_species(inst_name)][inst_name] # perform the appropriate lookup for the species
        axial_points = [aav*pole for (aav, pole) in zip(aavs, Base_SD.unit_circle.poles)] # multiply aavs by axial conponents to obtain set of points
        
        super().__init__(inst_name, axial_points, point_marker, centroid_marker)
        self.centroid *= Base_SD.unit_circle.N # scale centroid by number of points in this case to better adhere to unit circle
        
class Species_SD(Base_SD):
    '''1-order Spoke Diagram class for plotting centroid of all instances of a species'''
    def __init__(self, dataset, species, point_marker='b1', centroid_marker='m*'):
        inst_centroids = [Instance_SD(dataset, instance).centroid for instance in dataset[get_family(species)][species]]
        super().__init__(species, inst_centroids, point_marker, centroid_marker)
        
class Family_SD(Base_SD):
    '''2-order Spoke Diagram class for plotting centroid of all instances of a family'''
    def __init__(self, dataset, family, point_marker='m*', centroid_marker='cs'):
        spec_centroids = [Species_SD(dataset, species).centroid for species in dataset[family]]
        super().__init__(family, spec_centroids, point_marker, centroid_marker)
        
class Overlaid_Family_SD(Base_SD):
    '''2.5-order Spoke Diagram class for plotting all families on a single diagram, color-coded '''
    marker_map  = {
        'Acetates': 'g^',
        'Alcohols': 'r^',
        'Aldehydes': 'y^',
        'Ethers': 'b^',
        'Ketones': 'm^'
    }
    
    def __init__(self, dataset):
        self.famsds = [Family_SD(dataset, family, point_marker=self.marker_map[family]) for family in dataset]
    
    def draw(self, axes, index=(0,0)):
        ax = axes[index]
        for fsd in self.famsds:
            fsd.draw(axes, index) # draw over one another
            
        markers = [ax.scatter(np.nan, np.nan, color=color, marker=marker) for (color, marker) in self.marker_map.values()] # plot fake points for legend
        ax.legend(markers, self.marker_map.keys(), loc='lower right')

class Macro_SD(Base_SD):
    '''3-order Spoke Diagram from plotting trends across all data'''
    def __init__(self, dataset, point_marker='cs', centroid_marker='gp'):
        fam_centroids = [Family_SD(dataset, family).centroid for family in dataset]
        super().__init__('All Families', fam_centroids, point_marker, centroid_marker) 