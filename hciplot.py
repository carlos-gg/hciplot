__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['plot_frames',
           'plot_cubes']

import os
import shutil
import numpy as np
import holoviews as hv
from holoviews import opts
from subprocess import call
from matplotlib.pyplot import figure, subplot, show, Circle, savefig, close
from matplotlib.pyplot import colorbar as mplcbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import register_cmap
import matplotlib.colors as colors
import matplotlib.cm as mplcm


# Registering heat and cool colormaps from SaoImage DS9
# borrowed from: https://gist.github.com/adonath/c9a97d2f2d964ae7b9eb
ds9cool = {'red': lambda v: 2 * v - 1,
           'green': lambda v: 2 * v - 0.5,
           'blue': lambda v: 2 * v}
ds9heat = {'red': lambda v: np.interp(v, [0, 0.34, 1], [0, 1, 1]),
           'green': lambda v: np.interp(v, [0, 1], [0, 1]),
           'blue': lambda v: np.interp(v, [0, 0.65, 0.98, 1], [0, 0, 1, 1])}
register_cmap('ds9cool', data=ds9cool)
register_cmap('ds9heat', data=ds9heat)
default_cmap = 'viridis'
hv.extension('bokeh', 'matplotlib')


def plot_cubes(cube, mode='slider', backend='matplotlib', dpi=80, figtype='png',
               vmin=None, vmax=None, size=145, width=350, height=300,
               cmap=None, colorbar=True, dynamic=True, anim_path=None,
               data_step_range=None, label=None, label_step_range=None,
               delay=50, anim_format='gif', delete_anim_cache=True, **kwargs):
    """ Plot multi-dimensional high-contrast imaging datacubes (3d and 4d numpy
    arrays). It allows to visualize in-memory numpy arrays on Jupyterlab by
    leveraging the HoloViews library. It can also generate matplotlib animations
    from a 3d numpy array.

    Parameters
    ----------
    cube : np.ndarray
        Input cube.
    mode : {'slider', 'animation'}, str optional
        Whether to plot the 3d array as a widget with a slider or to save an
        animation of the 3d array. The animation is saved to disk using
        ImageMagick's convert command (it must be installed otherwise a
         ``FileNotFoundError`` will be raised)
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
    anim_fname : str, optional
        The animation path/filename. If None then the animation will be called
        ``animation``.``anim_format`` and will be saved in the current
        directory.
    data_step_range : tuple, optional
        Tuple of 1, 2 or 3 values that creates a range for slicing the ``data``
        cube.
    label : str, optional
        Label to be overlaid on top of each frame of the animation. If None,
        then 'frame #' will be used.
    labelpad : int, optional
        Padding of the label from the left bottom corner. 10 by default.
    label_step_range : tuple, optional
        Tuple of 1, 2 or 3 values that creates a range for customizing the label
        overlaid on top of the image.
    delay : int, optional
        Delay for displaying the frames in the animation sequence.
    anim_format : str, optional
        Format of the saved animation. By default 'gif' is used. Other formats
        supported by ImageMagick are valid, such as 'mp4'.
    **kwargs : dictionary, optional
        Arguments to be passed to ``plot_2d`` to customize the plot.


    Notes
    -----
    http://holoviews.org/getting_started/Gridded_Datasets.html
    http://holoviews.org/user_guide/Gridded_Datasets.html
    http://holoviews.org/user_guide/Applying_Customizations.html
    """
    if cmap is None:
        cmap = default_cmap

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
            # hv.save(image_stack, 'holomap.gif', fps=5)

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
        if backend == 'bokeh':
            print('Creating animations works with the matplotlib backend')

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

        print('Producing each animation frame...')
        for i, labstep in zip(data_step_range, list(label_step_range)):
            if label is None:
                label = 'frame '
            savelabel = dir_path + label + str(i + 100)
            plot_frames(cube[i], save=savelabel, dpi=dpi, vmin=vmin, vmax=vmax,
                        colorbar=colorbar, cmap=cmap,
                        label=[label + str(labstep + 1)], **kwargs)
        try:
            filename = anim_path + '.' + anim_format
            call(['convert', '-delay', str(delay), dir_path + '*.png',
                  filename])
            if os.path.exists(filename):
                print('Animation successfully saved to disk as ' + filename)
                if delete_anim_cache:
                    shutil.rmtree(dir_path)
                    print('Temp directory deleted ' + dir_path)

        except FileNotFoundError:
            print('ImageMagick `convert` command could not be found')

    else:
        raise ValueError("`mode` is not recognized")


def plot_frames(data, mode='mosaic', rows=1, vmax=None, vmin=None, circle=None,
                circle_alpha=0.8, circle_color='white', circle_radius=6,
                circle_label=False, arrow=None, arrow_alpha=0.8,
                arrow_length=20, arrow_shiftx=5, label=None, label_pad=5,
                label_size=12, grid=False, grid_alpha=0.4, grid_color='#f7f7f7',
                grid_spacing=None, cross=None, cross_alpha=0.4, ang_scale=False,
                ang_ticksep=50, pxscale=0.01, axis=True, show_center=False,
                cmap=None, log=False, colorbar=True, dpi=80, spsize=6,
                horsp=0.4, versp=0.2, title=None, sampling=1, save=False):
    """ Wrapper for easy creation of pyplot subplots. It is convenient for
    displaying VIP images in jupyter notebooks.

    Backend 'matplotlib'.

    Parameters
    ----------
    data : list
        List of 2d arrays or a single 3d array to be plotted.
    angscale : bool
        If True, the axes are displayed in angular scale (arcsecs).
    angticksep : int
        Separation for the ticks when using axis in angular scale.
    arrow : bool
        To show an arrow pointing to input px coordinates.
    arrowalpha : float
        Alpha transparency for the arrow.
    arrowlength : int
        Length of the arrow, 20 px by default.
    arrowshiftx : int
        Shift in x of the arrow pointing position, 5 px by default.
    axis : bool
        Show the axis, on by default.
    circle : tuple or list of tuples
        To show a circle at given px coordinates. The circles are shown on all
        subplots.
    circlealpha : float or list of floats
        Alpha transparencey for each circle.
    circlecolor : str
        Color or circle(s). White by default.
    circlelabel : bool
        Whether to show the coordinates of each circle.
    circlerad : int
        Radius of the circle, 6 px by default.
    cmap : str
        Colormap to be used, 'viridis' by default.
    colorb : bool
        To attach a colorbar, on by default.
    cross : tuple of float
        If provided, a crosshair is displayed at given px coordinates.
    crossalpha : float
        Alpha transparency of thr crosshair.
    dpi : int
        Dots per inch, for plot quality.
    grid : bool
        If True, a grid is displayed over the image, off by default.
    gridalpha : float
        Alpha transparency of the grid.
    gridcolor : str
        Color of the grid lines.
    gridspacing : int
        Separation of the grid lines in pixels.
    horsp : float
        Horizontal gap between subplots.
    label : str or list of str
        Text for annotating on subplots.
    labelpad : int
        Padding of the label from the left bottom corner. 5 by default.
    labelsize : int
        Size of the labels.
    log : bool
        Log colorscale.
    maxplots : int
        When the input (``*args``) is a 3d array, maxplots sets the number of
        cube slices to be displayed.
    pxscale : float
        Pixel scale in arcseconds/px. Default 0.01 for Keck/NIRC2.
    rows : int
        How many rows (subplots in a grid).
    save : str
        If a string is provided the plot is saved using this as the path.
    showcent : bool
        To show a big crosshair at the center of the frame.
    spsize : int
        Determines the size of the plot. Figsize=(spsize*ncols, spsize*nrows).
    title : str
        Title of the plot(s), None by default.
    vmax : int
        For stretching the displayed pixels values.
    vmin : int
        For stretching the displayed pixels values.
    versp : float
        Vertical gap between subplots.
    sampling : int, optional
        [mode='surface'] Sets the stride used to sample the input data to
        generate the surface graph.

    """
    # GEOM ---------------------------------------------------------------------
    # Chekcing inputs, we take a frame (1 or 3 channels) or tuple of them
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = [data]
        elif data.ndim == 3:
            raise TypeError("`data` must be a frame or tuple of frames")
    elif isinstance(data, tuple):
        for i in range(len(data)):
            # checking the elements are 2d (excepting the case of 3 channels)
            if not data[i].ndim == 2 and data[i].shape[2] != 3:
                raise ValueError("`data` must be a frame or tuple of frames")
    else:
        raise ValueError("`data` must be a frame or tuple of frames")

    num_plots = len(data)

    if num_plots % rows == 0:
        cols = num_plots / rows
    else:
        cols = (num_plots / rows) + 1

    # CIRCLE -------------------------------------------------------------------
    if circle is not None:
        if isinstance(circle, tuple):
            show_circle = True
            if isinstance(circle[0], tuple):
                n_circ = len(circle)
                coor_circle = circle
            elif isinstance(circle[0], (float, int)):
                n_circ = 1
                coor_circle = [circle] * n_circ
        else:
            print("`circle` must be a tuple (X,Y) or tuple of tuples (X,Y)")
            show_circle = False
    else:
        show_circle = False

    if show_circle:
        if isinstance(circle_radius, (float, int)):
            # single value is provided, used for all circles
            circle_radius = [circle_radius] * n_circ
        elif isinstance(circle_radius, tuple):
            # a different value for each circle
            if not n_circ == len(circle_radius):
                msg = '`circle_radius` must have the same len as `circle`'
                raise ValueError(msg)
        else:
            raise TypeError("`circle_rad` must be a float or tuple of floats")

    if show_circle:
        if isinstance(circle_alpha, (float, int)):
            circle_alpha = [circle_alpha] * n_circ
        elif isinstance(circle_alpha, tuple):
            # a different value for each circle
            if not num_plots == len(circle_alpha):
                msg = '`circle_alpha` must have the same len as `data`'
                raise ValueError(msg)

    # SHOW_CENTER --------------------------------------------------------------
    if show_center is not None:
        if isinstance(show_center, bool):
            show_center = [show_center] * num_plots

    # ARROW --------------------------------------------------------------------
    if arrow is not None:
        if isinstance(arrow, tuple):
            show_arrow = True
        else:
            raise ValueError("`arrow` must be a tuple (X,Y)")
    else:
        show_arrow = False

    # LABEL --------------------------------------------------------------------
    if label is not None:
        if not num_plots == len(label):
            raise ValueError("`label` does not contain enough items")

    # GRID ---------------------------------------------------------------------
    if grid is not None:
        if isinstance(grid, bool):
            grid = [grid] * num_plots

    if isinstance(grid_alpha, (float, int)):
        grid_alpha = [grid_alpha] * num_plots

    if isinstance(grid_color, str):
        grid_color = [grid_color] * num_plots

    if grid_spacing is not None:
        if isinstance(grid_spacing, int):
            grid_spacing = [grid_spacing] * num_plots
    else:
        grid_spacing = [None] * num_plots

    # VMAX-VMIN ----------------------------------------------------------------
    if isinstance(vmax, tuple):
        if not num_plots == len(vmax):
            raise ValueError("`vmax` does not contain enough items")
    elif isinstance(vmax, (int, float)):
        vmax = [vmax] * num_plots

    if isinstance(vmin, tuple):
        if not num_plots == len(vmin):
            raise ValueError("`vmin` does not contain enough items")
    elif isinstance(vmin, (int, float)):
        vmin = [vmin] * num_plots

    # CROSS --------------------------------------------------------------------
    if cross is not None:
        if not isinstance(cross, tuple):
            raise ValueError("`crosshair` must be a tuple (X,Y)")
        else:
            coor_cross = cross
            show_cross = True
    else:
        show_cross = False

    # ANGSCALE -----------------------------------------------------------------
    if ang_scale:
        print("`Pixel scale set to {}`".format(pxscale))

    # CMAP ---------------------------------------------------------------------
    if cmap is not None:
        custom_cmap = cmap
        if not isinstance(custom_cmap, tuple):
            custom_cmap = [cmap] * num_plots
        else:
            if not num_plots == len(custom_cmap):
                raise ValueError('`cmap` does not contain enough items')
    else:
        custom_cmap = [hciplot_cmap] * num_plots

    # COLORBAR -----------------------------------------------------------------
    if colorbar is not None:
        if isinstance(colorbar, bool):
            colorbar = [colorbar] * num_plots

    # LOG ----------------------------------------------------------------------
    if log:
        # Showing bad/nan pixels with the darkest color in current colormap
        current_cmap = mplcm.get_cmap()
        current_cmap.set_bad(current_cmap.colors[0])
        logscale = log
        if not isinstance(logscale, tuple):
            logscale = [log] * num_plots
        else:
            if not num_plots == len(logscale):
                raise ValueError('`log` does not contain enough items')
    else:
        logscale = [False] * num_plots

    # VMIN/VMAX ----------------------------------------------------------------
    if vmin is None:
        vmin = [None] * num_plots
    if vmax is None:
        vmax = [vmax] * num_plots

    # --------------------------------------------------------------------------
    if rows == 0:
        raise ValueError('Rows must be a positive integer')
    fig = figure(figsize=(cols * spsize, rows * spsize), dpi=dpi)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    if mode == 'surface':
        plot_mosaic = False
    elif mode == 'mosaic':
        plot_mosaic = True
    else:
        raise ValueError("`mode` value was not recognized")

    for i, v in enumerate(range(num_plots)):
        image = data[i].copy()
        frame_size = image.shape[0]  # assuming square frames
        cy = image.shape[0] / 2 - 0.5
        cx = image.shape[1] / 2 - 0.5
        v += 1
        ax = subplot(rows, cols, v)
        ax.set_aspect('equal')

        if plot_mosaic:
            ax = subplot(rows, cols, v)

            if logscale[i]:
                image += np.abs(image.min())
                if vmin[i] is None:
                    linthresh = 1e-2
                else:
                    linthresh = vmin[i]
                norm = colors.SymLogNorm(linthresh)
            else:
                norm = None

            if image.dtype == bool:
                image = image.astype(int)

            im = ax.imshow(image, cmap=custom_cmap[i], interpolation='nearest',
                           origin='lower', vmin=vmin[i], vmax=vmax[i],
                           norm=norm)

        else:
            x = np.outer(np.arange(0, frame_size, 1), np.ones(frame_size))
            y = x.copy().T
            ax = subplot(rows, cols, v, projection='3d')
            ax.plot_surface(x, y, image, rstride=sampling, cstride=sampling,
                            linewidth=2, cmap=custom_cmap[i], antialiased=True,
                            vmin=vmin[i], vmax=vmax[i])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('flux')
            ax.dist = 10
            if title is not None:
                ax.set_title(title)

        if show_circle and plot_mosaic:
            for j in range(n_circ):
                circ = Circle(coor_circle[j], radius=circle_radius[j],
                              fill=False, color=circle_color,
                              alpha=circle_alpha[j])
                ax.add_artist(circ)
                if circle_label:
                    x = coor_circle[j][0]
                    y = coor_circle[j][1]
                    cirlabel = str(int(x))+','+str(int(y))
                    ax.text(x, y + 1.8 * circle_radius[j], cirlabel, fontsize=8,
                            color='white', family='monospace', ha='center',
                            va='top', weight='bold', alpha=circle_alpha[j])

        if show_cross and plot_mosaic:
            ax.scatter([coor_cross[0]], [coor_cross[1]], marker='+',
                       color='white', alpha=cross_alpha)

        if show_center[i] and plot_mosaic:
            ax.axhline(cx, xmin=0, xmax=frame_size, alpha=0.3, lw=0.6,
                       linestyle='dashed', color='white')
            ax.axvline(cy, ymin=0, ymax=frame_size, alpha=0.3, lw=0.6,
                       linestyle='dashed', color='white')

        if show_arrow and plot_mosaic:
            ax.arrow(arrow[0] + arrow_length + arrow_shiftx, arrow[1],
                     -arrow_length, 0, color='white', head_width=10,
                     head_length=8, width=3, length_includes_head=True,
                     alpha=arrow_alpha)

        if label is not None and plot_mosaic:
            ax.annotate(label[i], xy=(label_pad, label_pad), color='white',
                        xycoords='axes pixels', weight='bold', size=label_size)

        if colorbar[i] and plot_mosaic:
            # create an axes to the right ax. The width of cax is 5% of ax
            # and the padding between cax and ax wis fixed at 0.05 inch
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = mplcbar(im, ax=ax, cax=cax, drawedges=False)
            cb.outline.set_linewidth(0.1)
            cb.ax.tick_params(labelsize=8)

        if grid[i] and plot_mosaic:
            if grid_spacing[i] is None:
                if cy < 10:
                    gridspa = 1
                elif cy >= 10:
                    if cy % 2 == 0:
                        gridspa = 4
                    else:
                        gridspa = 5
            else:
                gridspa = grid_spacing[i]

            ax.tick_params(axis='both', which='minor')
            minor_ticks = np.arange(0, data[i].shape[0], gridspa)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(True, which='minor', color=grid_color[i], linewidth=0.5,
                    alpha=grid_alpha[i], linestyle='dashed')
        else:
            ax.grid(False)

        if ang_scale and plot_mosaic:
            # Converting axes from pixels to arcseconds
            half_num_ticks = int(np.round(cy // ang_ticksep))

            # Calculate the pixel locations at which to put ticks
            ticks = []
            for t in range(half_num_ticks, -half_num_ticks-1, -1):
                # Avoid ticks not showing on the last pixel
                if not cy - t * ang_ticksep == frame_size:
                    ticks.append(cy - t * ang_ticksep)
                else:
                    ticks.append((cy - t * ang_ticksep) - 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # Corresponding distance in arcseconds, measured from the center
            labels = []
            for t in range(half_num_ticks, -half_num_ticks-1, -1):
                labels.append(-t * (ang_ticksep * pxscale))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("arcseconds", fontsize=12)
            ax.set_ylabel("arcseconds", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)

        if not axis:
            ax.set_axis_off()

    fig.subplots_adjust(wspace=horsp, hspace=versp)
    if save:
        savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0,
                transparent=True)
        close()
    else:
        show()


