__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['plot_frames',
           'plot_cubes']

import os
import shutil
import numpy as np
import holoviews as hv
from holoviews import opts
from subprocess import call
from matplotlib.pyplot import (figure, subplot, show, Circle, savefig, close,
                               hlines, annotate)
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


def plot_frames(data, backend='matplotlib', mode='mosaic', rows=1, vmax=None,
                vmin=None, circle=None, circle_alpha=0.8, circle_color='white',
                circle_radius=6, circle_label=False, arrow=None,
                arrow_alpha=0.8, arrow_length=10, arrow_shiftx=5, label=None,
                label_pad=5, label_size=12, grid=False, grid_alpha=0.4,
                grid_color='#f7f7f7', grid_spacing=None, cross=None,
                cross_alpha=0.4, ang_scale=False, ang_ticksep=50, pxscale=0.01,
                ang_legend=False, axis=True, show_center=False, cmap=None,
                log=False, colorbar=True, dpi=100, size_factor=6, horsp=0.4,
                versp=0.2, width=400, height=400, title=None, sampling=1,
                save=None, transparent=False):
    """ Plot a 2d array or a tuple of 2d arrays. Supports the ``matplotlib`` and
    ``bokeh`` backends. When having a tuple of 2d arrays, the plot turns into a
    mosaic. For ``matplotlib``, instead of a mosaic of images, we can create a
    mosaic of surface plots. Also, when using the ``matplotlib`` backend, this
    function allows to annotate and customize the plot and produce publication
    quality figures.

    Parameters
    ----------
    data : numpy.ndarray or tuple
        A single 2d array or a tuple of 2d arrays.
    backend : {'matplotlib', 'bokeh'}, str optional
        Selects the backend used to display the plots. ``Matplotlib`` plots
        are static and allow customization (leading to publication quality
        figures). ``Bokeh`` plots are interactive, allowing the used to zoom,
        pan, inspect pixel values, etc.
    mode : {'mosaic', 'surface'}, str optional
        [backend='matplotlib'] Controls whether to turn the images into surface
        plots.
    rows : int, optional
        How many rows (subplots in a grid) in the case ``data`` is a tuple of
        2d arrays.
    vmax : None, float or int, optional
        For defining the data range that the colormap covers. When set to None,
        the colormap covers the complete value range of the supplied data.
    vmin : None, float or int, optional
        For stretching the displayed pixels values. When set to None,
        the colormap covers the complete value range of the supplied data.
    circle : None, tuple or tuple of tuples, optional
        [backend='matplotlib'] To show a circle at the given px coordinates. The
        circles are shown on all subplots.
    circle_alpha : float or tuple of floats, optional
        [backend='matplotlib'] Alpha transparency for each circle.
    circle_color : str, optional
        [backend='matplotlib'] Color of the circles. White by default.
    circle_radius : int, optional
        [backend='matplotlib'] Radius of the circles, 6 px by default.
    circle_label : bool, optional
        [backend='matplotlib'] Whether to show the coordinates next to each
        circle.
    arrow : None or tuple of floats, optional
        [backend='matplotlib'] To show an arrow pointing to the given pixel
        coordinates.
    arrow_alpha : float, optional
        [backend='matplotlib'] Alpha transparency for the arrow.
    arrow_length : int, optional
        [backend='matplotlib'] Length of the arrow, 10 px by default.
    arrow_shiftx : int, optional
        [backend='matplotlib'] Shift in x of the arrow pointing position, 5 px
        by default.
    label : None, str or list of str, optional
        [backend='matplotlib'] Text for labeling each subplot. The label is
        shown at the bottom-left corner if each subplot.
    label_pad : int, optional
        [backend='matplotlib'] Padding of the label from the left bottom corner.
        5 by default.
    label_size : int, optional
        [backend='matplotlib'] Size of the labels font.
    grid : bool, optional
        [backend='matplotlib'] If True, a grid is displayed over the image, off
        by default.
    grid_alpha : float, optional
        [backend='matplotlib'] Alpha transparency of the grid.
    grid_color : str, optional
        [backend='matplotlib'] Color of the grid lines.
    grid_spacing : int, optional
        [backend='matplotlib'] Separation of the grid lines in pixels.
    cross : None or tuple of floats, optional
        [backend='matplotlib'] If provided, a crosshair is displayed at given
        pixel coordinates.
    cross_alpha : float, optional
        [backend='matplotlib'] Alpha transparency of the crosshair.
    ang_scale : bool, optional
        [backend='matplotlib'] If True, the axes are displayed in angular scale
        (arcsecs).
    ang_ticksep : int, optional
        [backend='matplotlib'] Separation for the ticks when using axis in
        angular scale.
    pxscale : float
        [backend='matplotlib'] Pixel scale in arcseconds/px. Default 0.01
        (Keck/NIRC2, SPHERE-IRDIS).
    ang_legend : bool, optional
        [backend='matplotlib'] If True a scaling bar (1 arcsec or 500 mas) will
        be added on the bottom-right corner of the subplots.
    axis : bool, optional
        [backend='matplotlib'] Show the axis, on by default.
    show_center : bool, optional
        [backend='matplotlib'] To show a crosshair at the center of the frame.
    cmap : None, str or tuple of str, optional
        Colormap to be used. When None, the value of the global variable
        ``default_cmap`` will be used.
    log : bool or tuple of bool, optional
        [backend='matplotlib'] Log colorscale.
    colorbar : bool or tuple of bool, optional
        To attach a colorbar, on by default.
    dpi : int, optional
        [backend='matplotlib'] Dots per inch, determines how many pixels the
        figure comprises (which affects the plot quality).
    size_factor : int, optional
        [backend='matplotlib'] Determines the size of the plot by setting the
        figsize parameter (width x height [inches]) as size_factor * ncols,
        size_factor * nrows.
    horsp : float, optional
        [backend='matplotlib'] Horizontal gap between subplots.
    versp : float, optional
        [backend='matplotlib'] Vertical gap between subplots.
    width : int, optional
        [backend='bokeh'] Controls the width of each subplot.
    height : int, optional
        [backend='bokeh'] Controls the height of each subplot.
    title : None or str, optional
        [backend='matplotlib'] Title of the whole figure, None by default.
    sampling : int, optional
        [mode='surface'] Sets the stride used to sample the input data to
        generate the surface graph.
    save : None or str, optional
        If a string is provided the plot is saved using ``save`` as the
        path/filename.
    transparent : bool, optional
        [save=True] Whether to have a transparent background between subplots.
        If False, then a white background is shown.

    """
    # Checking inputs: a frame (1 or 3 channels) or tuple of them
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

    if backend == 'bokeh':
        if mode == 'surface':
            raise ValueError('Surface plotting only supported with matplotlib '
                             'backend')
        if save is not None:
            raise ValueError('Saving is only supported with matplotlib backend')

    num_plots = len(data)

    if rows == 0:
        raise ValueError('Rows must be a positive integer')
    if num_plots % rows == 0:
        cols = int(num_plots / rows)
    else:
        cols = int((num_plots / rows) + 1)

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
            if not n_circ == len(circle_alpha):
                msg = '`circle_alpha` must have the same len as `circle`'
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

    # AXIS ---------------------------------------------------------------------
    if isinstance(axis, bool):
        axis = [axis] * num_plots

    # ANGSCALE -----------------------------------------------------------------
    if ang_scale and save is not None:
        print("`Pixel scale set to {}`".format(pxscale))

    if isinstance(ang_scale, bool):
        ang_scale = [ang_scale] * num_plots

    # CMAP ---------------------------------------------------------------------
    if cmap is not None:
        custom_cmap = cmap
        if not isinstance(custom_cmap, tuple):
            custom_cmap = [cmap] * num_plots
        else:
            if not num_plots == len(custom_cmap):
                raise ValueError('`cmap` does not contain enough items')
    else:
        custom_cmap = [default_cmap] * num_plots

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
    if backend == 'matplotlib':
        # Creating the figure --------------------------------------------------
        fig = figure(figsize=(cols * size_factor, rows * size_factor), dpi=dpi)

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

                im = ax.imshow(image, cmap=custom_cmap[i], origin='lower',
                               interpolation='nearest', vmin=vmin[i],
                               vmax=vmax[i], norm=norm)

            else:
                x = np.outer(np.arange(0, frame_size, 1), np.ones(frame_size))
                y = x.copy().T
                ax = subplot(rows, cols, v, projection='3d')
                ax.plot_surface(x, y, image, rstride=sampling, cstride=sampling,
                                linewidth=2, cmap=custom_cmap[i],
                                antialiased=True, vmin=vmin[i], vmax=vmax[i])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('flux')
                ax.dist = 10
                if title is not None:
                    ax.set_title(title)

            if ang_legend and plot_mosaic:
                scaleng = 1. / pxscale
                scalab = '1 arcsec'
                scalabloc = scaleng / 2. - 8
                if scaleng > frame_size / 2.:
                    scaleng /= 2.
                    scalab = '500 mas'
                    scalabloc = scaleng / 2. - 8
                scapad = 4
                xma = frame_size - scapad
                xmi = xma - scaleng
                hlines(y=scapad, xmin=xmi, xmax=xma, colors='white', lw=1.,
                       linestyles='solid')
                annotate(scalab, (xmi + scalabloc, scapad + 2), color='white')

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
                        ax.text(x, y + 1.8 * circle_radius[j], cirlabel,
                                fontsize=8, color='white', family='monospace',
                                ha='center', va='top', weight='bold',
                                alpha=circle_alpha[j])

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
                         -arrow_length, 0, color='white', head_width=6,
                         head_length=4, width=2, length_includes_head=True,
                         alpha=arrow_alpha)

            if label is not None and plot_mosaic:
                ax.annotate(label[i], xy=(label_pad, label_pad), color='white',
                            xycoords='axes pixels', weight='bold',
                            size=label_size)

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

            if ang_scale[i] and plot_mosaic:
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
            else:
                ax.set_xlabel("x", fontsize=12)
                ax.set_ylabel("y", fontsize=12)

            if not axis[i]:
                ax.set_axis_off()

        fig.subplots_adjust(wspace=horsp, hspace=versp)

        if save is not None and isinstance(save, str):
            savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0,
                    transparent=transparent)
            close()
        else:
            show()

    elif backend == 'bokeh':
        subplots = []
        options = "Image (cmap='" + custom_cmap[0] + "')"  # taking first item
        hv.opts(options)

        for i, v in enumerate(range(num_plots)):
            image = data[i].copy()
            if vmin[i] is None:
                vmin_i = image.min()
            if vmax[i] is None:
                vmax_i = image.max()
            im = hv.Image((range(image.shape[1]), range(image.shape[0]), image))
            subplots.append(im.opts(tools=['hover'], colorbar=colorbar[i],
                                    colorbar_opts={'width': 15},
                                    width=width, height=height,
                                    clim=(vmin_i, vmax_i)))

        return hv.Layout(subplots).cols(cols)

    else:
        raise ValueError('`backend` not supported')


def plot_cubes(cube, mode='slider', backend='matplotlib', dpi=100,
               figtype='png', vmin=None, vmax=None, size=120, width=400,
               height=400, cmap=None, colorbar=True, dynamic=True,
               anim_path=None, data_step_range=None, label=None,
               label_step_range=None, delay=50, anim_format='gif',
               delete_anim_cache=True, **kwargs):
    """ Plot multi-dimensional high-contrast imaging datacubes (3d and 4d
    ``numpy`` arrays). It allows to visualize in-memory ``numpy`` arrays on
    ``Jupyterlab`` by leveraging the ``HoloViews`` library. It can also generate
    and save animations from a 3d ``numpy`` array with ``matplotlib``.

    Parameters
    ----------
    cube : np.ndarray
        Input 3d or 4d cube.
    mode : {'slider', 'animation'}, str optional
        Whether to plot the 3d array as a widget with a slider or to save an
        animation of the 3d array. The animation is saved to disk using
        ImageMagick's convert command (it must be installed otherwise a
        ``FileNotFoundError`` will be raised).
    backend : {'matplotlib', 'bokeh'}, str optional
        Selects the backend used to display the plots. ``Bokeh`` plots are
        interactive, allowing the used to zoom, pan, inspect pixel values, etc.
        ``Matplotlib`` can lead to some flickering when using the slider and
        ``dynamic`` is True.
    dpi : int, optional
        [backend='matplotlib'] The rendered dpi of the figure.
    figtype : {'png', 'svg'}, str optional
        [backend='matplotlib'] Type of output.
    vmin : None, float or int, optional
        For defining the data range that the colormap covers. When set to None,
        the colormap covers the complete value range of the supplied data.
    vmax : None, float or int, optional
        For defining the data range that the colormap covers. When set to None,
        the colormap covers the complete value range of the supplied data.
    size : int, optional
        [backend='matplotlib'] Sets the size of the plot.
    width : int, optional
        [backend='bokeh'] Sets the width of the plot.
    height : int, optional
        [backend='bokeh'] Sets the height of the plot.
    cmap : None or str, optional
        Colormap. When None, the value of the global variable ``default_cmap``
        will be used.
    colorbar : bool, optional
        If True, a colorbar is shown.
    dynamic : bool, optional
        [mode='slider'] When False, a ``HoloViews.HoloMap`` is created (slower
        and will take up a lot of RAM for large datasets). If True, a
        ``HoloViews.DynamicMap`` is created instead.
    anim_path : str, optional
        [mode='animation'] The animation path/filename. If None then the
        animation will be called ``animation``.``anim_format`` and will be saved
        in the current directory.
    data_step_range : tuple, optional
        [mode='animation'] Tuple of 1, 2 or 3 values that creates a range for
        slicing the ``data`` cube.
    label : str, optional
        [mode='animation'] Label to be overlaid on top of each frame of the
        animation. If None, then ``frame <#>`` will be used.
    label_step_range : tuple, optional
        [mode='animation'] Tuple of 1, 2 or 3 values that creates a range for
        customizing the label overlaid on top of each frame of the animation.
    delay : int, optional
        [mode='animation'] Delay for displaying the frames in the animation
        sequence.
    anim_format : str, optional
        [mode='animation'] Format of the saved animation. By default 'gif' is
        used. Other formats supported by ImageMagick are valid, such as 'mp4'.
    delete_anim_cache : str, optional
        [mode='animation'] If True, the cache folder is deleted once the
        animation file is saved to disk.
    **kwargs : dictionary, optional
        [mode='animation'] Arguments to be passed to ``plot_frames`` to
        customize each frame of the animation (adding markers, using a log
        scale, etc).

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

        # not working for bokeh: size, dpi
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
            plot_frames(cube[i], backend='matplotlib', mode='mosaic',
                        save=savelabel, dpi=dpi, vmin=vmin, vmax=vmax,
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


