import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.font_manager import FontProperties

def _group_particles(dataframe, features):
    particles = dataframe.particle.unique()

    dict = {}

    for i in particles:
        dict[i] = dataframe.loc[dataframe['particle']==i][features]
    return dict

def view_stack(stack, **kwargs):
    fig, ax = plt.subplots()

    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', stack.max())
    cmap = kwargs.get('cmap', 'gray')
    
    if len(stack.shape)==3:
        fig.index = stack.shape[0]//2
        im = ax.imshow(stack[fig.index, :, :], cmap=cmap, vmin=vmin, vmax=vmax)

        # Make a slider
        sld_axframe = plt.axes([0.20, 0.02, 0.65, 0.03])
        slider = Slider(sld_axframe, 'Frame', 0, stack.shape[0]-1, valinit=fig.index, valfmt='%i')
        
        def _onscroll(event):
            fig = event.canvas.figure
            if event.button == 'up':
                fig.index = (fig.index-1) % stack.shape[0]
            else:
                fig.index = (fig.index+1) % stack.shape[0]
            slider.set_val(fig.index)
            _update(fig.index)

        def _update(frame):
            fig.index = int(frame)
            im.set_data(stack[fig.index, :, :])
            ax.set_ylabel('slice %s' % fig.index)
            im.axes.figure.canvas.draw()

        slider.on_changed(_update)
        fig.canvas.mpl_connect('scroll_event', _onscroll)
        plt.show()
    
    elif len(stack.shape)==2:
        ax.imshow(stack, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.show()

# diameter is displayed in points, not in pixels. Roughly +3 to get to pixels
def view_located_features(stack, localisation_dataframe, diameter, **kwargs):
    _plot_style = kwargs.get('plot_style' , dict(markersize=diameter+3, markeredgewidth=0.5,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none'))

    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', stack.max())
    cmap = kwargs.get('cmap', 'gray')

    fig, ax = plt.subplots()

    # Have a figure attribute that allows us to track the frame
    fig.index = stack.shape[0]//2
    
    # Find localised features in frame
    features = localisation_dataframe.loc[localisation_dataframe['frame']==fig.index][['x', 'y']]

    im = ax.imshow(stack[fig.index, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    centroids = ax.plot(features['x'], features['y'], **_plot_style)[0]

    # Make a slider
    sld_axframe = plt.axes([0.20, 0.02, 0.65, 0.03])
    slider = Slider(sld_axframe, 'Frame', 0, stack.shape[0]-1, valinit=fig.index, valfmt='%i')
    
    def _onscroll(event):
        fig = event.canvas.figure
        if event.button == 'up':
            fig.index = (fig.index-1) % stack.shape[0]
        else:
            fig.index = (fig.index+1) % stack.shape[0]
        slider.set_val(fig.index)
        _update(fig.index)

    def _update(frame):
        fig.index = int(frame)
        features = localisation_dataframe[localisation_dataframe['frame']==fig.index][['x', 'y']]
        im.set_data(stack[int(frame), :, :])
        centroids.set_data(features['x'], features['y'])
        ax.set_ylabel('slice %s' % int(frame))
        im.axes.figure.canvas.draw()
    
    
    slider.on_changed(_update)
    fig.canvas.mpl_connect('scroll_event', _onscroll)
    plt.show()
    
def view_tracks(stack, tracking_dataframe, localisation_dataframe, diameter, **kwargs):
    _plot_style_1 = dict(linewidth=1, alpha=0.7)
    _plot_style_2 = dict(markersize=diameter+3, markeredgewidth=0.5,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none')
    fontP = FontProperties()
    fontP.set_size('xx-small')

    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', stack.max())
    cmap = kwargs.get('cmap', 'gray')

    fig, ax = plt.subplots()

    # Have a figure attribute that allows us to track the frame
    fig.index = stack.shape[0]//2

    features = localisation_dataframe.loc[localisation_dataframe['frame']==fig.index][['x', 'y']]
    
    # Group tracks by particle and plot them
    particle_dict = _group_particles(tracking_dataframe, ['x', 'y'])
    for particle in particle_dict:
        ax.plot(particle_dict[particle].iloc[:,0], particle_dict[particle].iloc[:,1], **_plot_style_1, label = particle)
        
    
    im = ax.imshow(stack[fig.index, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    centroids = ax.plot(features['x'], features['y'], **_plot_style_2)[0]
    ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), prop=fontP)

    # Make a slider
    sld_axframe = plt.axes([0.20, 0.02, 0.65, 0.03])
    slider = Slider(sld_axframe, 'Frame', 0, stack.shape[0]-1, valinit=fig.index, valfmt='%i')
    
    def _onscroll(event):
        fig = event.canvas.figure
        if event.button == 'up':
            fig.index = (fig.index-1) % stack.shape[0]
        else:
            fig.index = (fig.index+1) % stack.shape[0]
        slider.set_val(fig.index)
        _update(fig.index)

    def _update(frame):
        fig.index = int(frame)
        features = localisation_dataframe[localisation_dataframe['frame']==fig.index][['x', 'y']]
        im.set_data(stack[int(frame), :, :])
        centroids.set_data(features['x'], features['y'])
        ax.set_ylabel('slice %s' % int(frame))
        im.axes.figure.canvas.draw()
    
    
    slider.on_changed(_update)
    fig.canvas.mpl_connect('scroll_event', _onscroll)
    plt.show()


