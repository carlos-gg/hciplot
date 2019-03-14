import holoviews as hv
from holoviews import opts

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['plot2d',
           'plot3d']

import os
import shutil
import numpy as np
from subprocess import Popen
from matplotlib.pyplot import figure, subplot, show, Circle, savefig, close
from matplotlib.pyplot import colorbar as mplcbar
import matplotlib.colors as colors
import matplotlib.cm as mplcm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import register_cmap


# Registering heat and cool colormaps from DS9
# taken from: https://gist.github.com/adonath/c9a97d2f2d964ae7b9eb
ds9cool = {'red': lambda v: 2 * v - 1,
           'green': lambda v: 2 * v - 0.5,
           'blue': lambda v: 2 * v}
ds9heat = {'red': lambda v: np.interp(v, [0, 0.34, 1], [0, 1, 1]),
           'green': lambda v: np.interp(v, [0, 1], [0, 1]),
           'blue': lambda v: np.interp(v, [0, 0.65, 0.98, 1], [0, 0, 1, 1])}
register_cmap('ds9cool', data=ds9cool)
register_cmap('ds9heat', data=ds9heat)
hv.extension('bokeh', 'matplotlib')
hciplot_cmap = 'viridis'


def cube_plot(cube, backend='bokeh', dpi=80, figtype='png', vmin=None,
              vmax=None, size=145, width=350, height=300, cmap='viridis',
              colorbar=True, dynamic=True):
    """ Wrapper of HoloViews library for the visualization of multi-dimensional
    high-contrast imaging datacubes (in-memory numpy arrays) on Jupyterlab.
def plot3d(cube, mode='slider', backend='matplotlib', dpi=80, figtype='png',
           vmin=None, vmax=None, size=145, width=350, height=300,
           cmap=None, colorbar=True, dynamic=True, anim_path=None,
           data_step_range=None, label=None, label_step_range=None, delay=50,
           anim_format='gif', **kwargs):
    """ Plot multi-dimensional high-contrast imaging datacubes (3d and 4d numpy
    arrays). It allows to visualize in-memory numpy arrays on Jupyterlab by
    leveraging the HoloViews library. It can also generate matplotlib animations
    from a 3d numpy array.

    Parameters
    ----------
    cube : np.ndarray
        Input cube.
    dpi : int, optional
        [backend='matplotlib'] The rendered dpi of the figure.
    figtype : {'png', 'svg'}, str optional
        [backend='matplotlib'] Type of output.
    vmin : float, optional
        Min value.
    vmax : float, optional
        Max value.
    size :
        [backend='matplotlib']
    width :
        [backend='bokeh']
    height :
        [backend='bokeh']
    cmap : str, optional
        Colormap.
    dynamic : bool, optional
        When False, a HoloMap is created (slower and will take up a lot of RAM
        for large datasets). If True, a DynamicMap is created instead.
        
    Notes
    -----
    http://holoviews.org/getting_started/Gridded_Datasets.html
    http://holoviews.org/user_guide/Gridded_Datasets.html
    http://holoviews.org/user_guide/Applying_Customizations.html

    # Colorbar and aspect ratio:
    https://github.com/pyviz/holoviews/issues/236
    """
    if cmap is None:
        cmap = hciplot_cmap

    if mode == 'slider':
        if cube.ndim == 3:
            # Dataset((X, Y, Z), Data), where
            # X is a 1D array of shape M ,
            # Y is a 1D array of shape N and
            # Z is a 1D array of shape O
            # Data is a ND array of shape NxMxO
            ds = hv.Dataset((range(cube.shape[2]), range(cube.shape[1]),
                             range(cube.shape[0]), cube), ['x', 'y', 'time'],
                            'flux')
            max_frames = cube.shape[0]
        elif cube.ndim == 4:
            # adding a lambda dimension
            ds = hv.Dataset((range(cube.shape[3]), range(cube.shape[2]),
                             range(cube.shape[1]), range(cube.shape[0]), cube),
                            ['x', 'y', 'time', 'lambda'], 'flux')
            max_frames = cube.shape[0] * cube.shape[1]
        else:
            raise TypeError('Only 3d and 4d numpy arrays are accepted when '
                            '`mode`=`slider`')

        # Matplotlib takes None but not Bokeh. We take global min & max instead
        if vmin is None:
            vmin = cube.min()
        if vmax is None:
            vmax = cube.max()

        print(ds)
        print(":Cube_shape\t{}".format(list(cube.shape[::-1])))

        image_stack = ds.to(hv.Image, kdims=['x', 'y'], dynamic=dynamic)
        hv.output(backend=backend, size=size, dpi=dpi, fig=figtype,
                  max_frames=max_frames)

        if backend == 'matplotlib':
            # keywords in the currently active 'matplotlib' renderer are:
            # 'alpha', 'clims', 'cmap', 'filterrad', 'interpolation', 'norm',
            # 'visible'
            options = "Image (cmap='" + cmap + "', interpolation='nearest',"
            options += " clims=("+str(vmin)+','+str(vmax)+")"+")"
            opts(options, image_stack)
            return image_stack.opts(opts.Image(colorbar=colorbar))
        elif backend == 'bokeh':
            options = "Image (cmap='" + cmap + "')"
            opts(options, image_stack)
            return image_stack.opts(opts.Image(colorbar=colorbar,
                                               colorbar_opts={'width': 15},
                                               width=width, height=height,
                                               clim=(vmin, vmax),
                                               tools=['hover']))

    elif mode == 'animation':
        if not (isinstance(cube, np.ndarray) and cube.ndim == 3):
            raise TypeError('Only 3d numpy arrays are accepted when '
                            '`mode`=`animation`')

        dir_path = './animation_temp/'
        if anim_path is None:
            anim_path = './animation'

        if data_step_range is None:
            data_step_range = range(0, cube.shape[0], 1)
        else:
            if not isinstance(data_step_range, tuple):
                msg = '`data_step_range` must be a tuple with 1, 2 or 3 values'
                raise ValueError(msg)
            if len(data_step_range) == 1:
                data_step_range = range(data_step_range)
            elif len(data_step_range) == 2:
                data_step_range = range(data_step_range[0], data_step_range[1])
            elif len(data_step_range) == 3:
                data_step_range = range(data_step_range[0],
                                        data_step_range[1],
                                        data_step_range[2])

        if label_step_range is None:
            label_step_range = data_step_range
        else:
            if not isinstance(label_step_range, tuple):
                msg = '`label_step_range` must be a tuple with 1, 2 or 3 values'
                raise ValueError(msg)
            if len(label_step_range) == 1:
                label_step_range = range(label_step_range)
            elif len(label_step_range) == 2:
                label_step_range = range(label_step_range[0],
                                         label_step_range[1])
            elif len(label_step_range) == 3:
                label_step_range = range(label_step_range[0],
                                         label_step_range[1],
                                         label_step_range[2])

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print('Replacing ' + dir_path)
        os.mkdir(dir_path)

        for i, labstep in zip(data_step_range, list(label_step_range)):
            if label is None:
                label = 'frame '
            savelabel = dir_path + label + str(i + 100)
            plot2d(cube[i], save=savelabel, label=[label + str(labstep + 1)],
                   **kwargs)
        try:
            filename = anim_path + '.' + anim_format
            Popen(['convert', '-delay', str(delay), dir_path + '*.png',
                   filename])
            print('Animation successfully saved to disk as ' + filename)
        except FileNotFoundError:
            print('ImageMagick `convert` command could not be found')

    else:
        raise ValueError("`mode` is not recognized")
    else:
        raise TypeError('This function is intended for 3d and 4d HCI datacubes')

    # Matplotlib handles None but not Bokeh. Instead take the global min & max
    if vmin is None:
        vmin = cube.min()
    if vmax is None:
        vmax = cube.max()

    print(ds)
    print(":Cube_shape\t{}".format(list(cube.shape[::-1])))

    image_stack = ds.to(hv.Image, kdims=['x', 'y'], dynamic=dynamic)
    hv.output(backend=backend, size=size, dpi=dpi, fig=figtype,
              max_frames=max_frames)

    if backend == 'matplotlib':
        # keywords in the currently active 'matplotlib' renderer are:
        # 'alpha', 'clims', 'cmap', 'filterrad', 'interpolation', 'norm',
        # 'visible'
        options = "Image (cmap='" + cmap + "', interpolation='nearest',"
        options += " clims=("+str(vmin)+','+str(vmax)+")"+")"
        opts(options, image_stack)
        return image_stack.opts(opts.Image(colorbar=colorbar))
    elif backend == 'bokeh':
        options = "Image (cmap='" + cmap + "')"
        opts(options, image_stack)
        # Colorbar_opts: 'padding': 2, 'major_tick_out': 4, 'label_standoff': 6
        return image_stack.opts(opts.Image(colorbar=colorbar,
                                           colorbar_opts={'width': 15},
                                           width=width, height=height,
                                           clim=(vmin, vmax), tools=['hover']))



