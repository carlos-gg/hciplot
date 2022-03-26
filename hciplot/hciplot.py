__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['plot_frames',
           'plot_cubes']

from decimal import *
import os
import shutil
import numpy as np
import holoviews as hv
from holoviews import opts
from subprocess import call
from matplotlib.pyplot import (figure, subplot, show, Circle, savefig, close,
                               hlines, annotate)
from matplotlib.pyplot import colorbar as plt_colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import register_cmap
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as mplcm
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

# Registering heat and cool colormaps from SaoImage DS9
# borrowed from: https://gist.github.com/adonath/c9a97d2f2d964ae7b9eb
ds9cool = {'red': lambda v: 2 * v - 1,
           'green': lambda v: 2 * v - 0.5,
           'blue': lambda v: 2 * v}
ds9heat = {'red': lambda v: np.interp(v, [0, 0.34, 1], [0, 1, 1]),
           'green': lambda v: np.interp(v, [0, 1], [0, 1]),
           'blue': lambda v: np.interp(v, [0, 0.65, 0.98, 1], [0, 0, 1, 1])}
register_cmap(cmap=LinearSegmentedColormap('ds9cool', ds9cool))
register_cmap(cmap=LinearSegmentedColormap('ds9heat', ds9heat))
cmap_binary = colors.ListedColormap(['black', 'white'])
default_cmap = 'viridis'


def plot_frames(data, backend='matplotlib', mode='mosaic', rows=1, vmax=None,
                vmin=None, circle=None, circle_alpha=0.8, circle_color='white',
                circle_linestyle='-', circle_radius=6, circle_label=False, circle_label_color='white',
                arrow=None, arrow_alpha=0.8, arrow_length=10, arrow_shiftx=5, 
                arrow_label=None, label=None, label_pad=5, label_size=12, 
                label_color='white',grid=False, grid_alpha=0.4,  grid_color='#f7f7f7', 
                grid_spacing=None, cross=None, cross_alpha=0.4, lab_fontsize=8,
                cross_color='white', ang_scale=False, ang_ticksep=50, ndec=1, 
                pxscale=0.01, auscale=1., ang_legend=False, au_legend=False, 
                axis=True, show_center=False, cmap=None, log=False, 
                colorbar=True, colorbar_ticks=None, dpi=100, size_factor=6, 
                horsp=0.4, versp=0.2, width=400, height=400, title=None, 
                tit_size=16, sampling=1, save=None, transparent=False):
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
    vmax : None, float/int or tuple of float/int, optional
        For defining the data range that the colormap covers. When set to None,
        the colormap covers the complete value range of the supplied data.
    vmin : None, float/int or tuple of float/int, optional
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
    circle_label : bool or string, optional
        [backend='matplotlib'] Whether to show the coordinates next to each
        circle. If a string: the string to be printed. If a tuple, should be 
        a tuple of strings with same length as 'circle'.
    circle_label_color : str, optional
        [backend='matplotlib'] Default 'white'. Sets the color of the circle
        label
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
    arrow_label : bool or string, optional
        [backend='matplotlib'] Label to be printed next to the arrow.
    label : None, str or list of str/None, optional
        [backend='matplotlib'] Text for labeling each subplot. The label is
        shown at the bottom-left corner if each subplot.
    label_pad : int or tuple of int, optional
        [backend='matplotlib'] Padding of the label from the left bottom corner.
        5 by default. If a tuple, sets the padding in x and y.
    label_size : int, optional
        [backend='matplotlib'] Size of the labels font.
    grid : bool or tuple of bools, optional
        [backend='matplotlib'] If True, a grid is displayed over the image, off
        by default.
    grid_alpha : None, float/int or tuple of None/float/int, optional
        [backend='matplotlib'] Alpha transparency of the grid.
    grid_color : str, optional
        [backend='matplotlib'] Color of the grid lines.
    grid_spacing : None, float/int or tuple of None/float/int, optional
        [backend='matplotlib'] Separation of the grid lines in pixels.
    cross : None or tuple of floats, optional
        [backend='matplotlib'] If provided, a crosshair is displayed at given
        pixel coordinates.
    cross_alpha : float, optional
        [backend='matplotlib'] Alpha transparency of the crosshair.
    cross_color : string, optional
        [backend='matplotlib'] Color of the crosshair.
    ang_scale : bool or tuple of bools, optional
        [backend='matplotlib'] If True, the axes are displayed in angular scale
        (arcsecs).
    ang_ticksep : int, optional
        [backend='matplotlib'] Separation for the ticks when using axis in
        angular scale.
    ndec : int, optional
        [backend='matplotlib'] Number of decimals for axes labels.
    pxscale : float, optional
        [backend='matplotlib'] Pixel scale in arcseconds/px. Default 0.01
        (Keck/NIRC2, SPHERE-IRDIS).
    auscale : float, optional
        [backend='matplotlib'] Pixel scale in au/px. Default 1.
    ang_legend : bool or tuple of bools, optional
        [backend='matplotlib'] If True a scaling bar (1 arcsec or 500 mas) will
        be added on the bottom-right corner of the subplots.
    au_legend : bool or tuple of bools, optional
        [backend='matplotlib'] If True (and ang_legend is False) a scaling bar 
        (10 au, 20 au or 50 au) will be added on the top-right corner of the 
        subplots.
    axis : bool, optional
        [backend='matplotlib'] Show the axis, on by default.
    show_center : bool or tuple of bools, optional
        [backend='matplotlib'] To show a cross at the center of the frame.
    cmap : None, str or tuple of str, optional
        Colormap to be used. When None, the value of the global variable
        ``default_cmap`` will be used. Any string corresponding to a valid
        ``matplotlib`` colormap can be used. Additionally, 'ds9cool', 'ds9heat'
        and 'binary' (for binary maps) are valid colormaps for this function.
    log : bool or tuple of bool, optional
        [backend='matplotlib'] Log color scale.
    colorbar : bool or tuple of bool, optional
        To attach a colorbar, on by default.
    colorbar_ticks : None, tuple or tuple of tuples, optional
        [backend='matplotlib'] Custom ticks for the colorbar of each plot.
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
    tit_size: int, optional
        Size of the title font.
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
    def check_bool_param(param, name):
        msg_type = '`' + name + '` must be a bool or tuple of bools'
        if isinstance(param, bool):
            param = [param] * num_plots
        elif isinstance(param, tuple):
            if not num_plots == len(param):
                msg = 'The len of `' + name + '` ({}) does not match the ' + \
                      'number of plots ({})'
                raise ValueError(msg.format(len(param), num_plots))
            else:
                for elem in param:
                    if not isinstance(elem, bool):
                        raise TypeError(msg_type)
        else:
            raise TypeError(msg_type)
        return param

    def check_numeric_param(param, name):
        msg_type = '`' + name + '` must be a None, float/int or tuple of ' + \
                   'None/float/ints'
        if param is None:
            param = [None] * num_plots
        elif isinstance(param, (int, float)):
            param = [param] * num_plots
        elif isinstance(param, tuple):
            if not num_plots == len(param):
                msg = 'The len of `' + name + '` ({}) does not match the ' + \
                      'number of plots ({})'
                raise ValueError(msg.format(len(param), num_plots))
            else:
                for elem in param:
                    if elem and not isinstance(elem, (float, int)):
                        raise TypeError(msg_type)
        else:
            raise TypeError(msg_type)
        return param

    def check_str_param(param, name, default_value=None):
        msg_type = '`' + name + '` must be a None, str or tuple of ' + \
                   'None/str'
        if param is None:
            param = [default_value] * num_plots
        elif isinstance(param, str):
            param = [param] * num_plots
        elif isinstance(param, tuple):
            if not num_plots == len(param):
                msg = 'The len of `' + name + '` ({}) does not match the ' + \
                      'number of plots ({})'
                raise ValueError(msg.format(len(param), num_plots))
            else:
                for elem in param:
                    if elem and not isinstance(elem, str):
                        raise TypeError(msg_type)
        else:
            raise TypeError(msg_type)
        return param
    # --------------------------------------------------------------------------

    # Checking inputs: a frame (1 or 3 channels) or tuple of them
    msg_data_type = "`data` must be a frame (2d array) or tuple of frames"
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = [data]
        elif data.ndim == 3:
            raise TypeError(msg_data_type)
    elif isinstance(data, tuple):
        for i in range(len(data)):
            # checking the elements are 2d (excepting the case of 3 channels)
            if not data[i].ndim == 2:# and data[i].shape[2] != 3:
                raise ValueError(msg_data_type)
    else:
        raise ValueError(msg_data_type)

    if not isinstance(backend, str):
        raise TypeError('`backend` must be a string. ' + msg_data_type)

    if backend == 'bokeh':
        if mode == 'surface':
            raise ValueError('Surface plotting only supported with matplotlib '
                             'backend')
        if save is not None:
            raise ValueError('Saving is only supported with matplotlib backend')

    if isinstance(label_pad, tuple):
        label_pad_x, label_pad_y = label_pad
    else:
        label_pad_x = label_pad
        label_pad_y = label_pad

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

    # ARROW --------------------------------------------------------------------
    if arrow is not None:
        if isinstance(arrow, tuple):
            show_arrow = True
        else:
            raise ValueError("`arrow` must be a tuple (X,Y)")
    else:
        show_arrow = False

    # VMAX-VMIN ----------------------------------------------------------------
    vmin = check_numeric_param(vmin, 'vmin')
    vmax = check_numeric_param(vmax, 'vmax')

    # CROSS --------------------------------------------------------------------
    if cross is not None:
        if not isinstance(cross, tuple):
            raise ValueError("`crosshair` must be a tuple (X,Y)")
        else:
            coor_cross = cross
            show_cross = True
    else:
        show_cross = False

    # AXIS, GRID, ANG_SCALE ----------------------------------------------------
    axis = check_bool_param(axis, 'axis')
    grid = check_bool_param(grid, 'grid')
    grid_alpha = check_numeric_param(grid_alpha, 'grid_alpha')
    grid_spacing = check_numeric_param(grid_spacing, 'grid_spacing')
    show_center = check_bool_param(show_center, 'show_center')
    ang_scale = check_bool_param(ang_scale, 'ang_scale')
    ang_legend = check_bool_param(ang_legend, 'ang_legend')
    au_legend = check_bool_param(au_legend, 'au_legend')

    if isinstance(grid_color, str):
        grid_color = [grid_color] * num_plots

    if any(ang_scale) and save is not None:
        print("`Pixel scale set to {}`".format(pxscale))

    # LABEL --------------------------------------------------------------------
    label = check_str_param(label, 'label')

    # CMAP ---------------------------------------------------------------------
    custom_cmap = check_str_param(cmap, 'cmap', default_cmap)

    # COLORBAR -----------------------------------------------------------------
    colorbar = check_bool_param(colorbar, 'colorbar')

    if colorbar_ticks is not None:
        cbar_ticks = colorbar_ticks
        # must be a tuple
        if isinstance(cbar_ticks, tuple):
            # tuple of tuples
            if isinstance(cbar_ticks[0], tuple):
                if not num_plots == len(cbar_ticks):
                    raise ValueError('`colorbar_ticks` does not contain enough '
                                     'items')
            # single tuple
            elif isinstance(cbar_ticks[0], (float, int)):
                cbar_ticks = [colorbar_ticks] * num_plots
        else:
            raise TypeError('`colorbar_ticks` must be a tuple or tuple of '
                            'tuples')
    else:
        cbar_ticks = [None] * num_plots

    # LOG ----------------------------------------------------------------------
    logscale = check_bool_param(log, 'log')
#    if any(logscale):
#        # Showing bad/nan pixels with the darkest color in current colormap
#        current_cmap = mplcm.get_cmap()
#        current_cmap.set_bad(current_cmap.colors[0])

    # --------------------------------------------------------------------------
    if backend == 'matplotlib':
        # Creating the figure --------------------------------------------------
        fig = figure(figsize=(cols * size_factor, rows * size_factor), dpi=dpi)

        if title is not None:
            fig.suptitle(title, fontsize=tit_size, va='center', x=0.51, 
                         #y=1-0.08*(28/tit_size)**(0.5))
                         y=1-0.1*(16/tit_size))

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

            if plot_mosaic:
                ax = subplot(rows, cols, v)
                ax.set_aspect('auto')

                if logscale[i]:
                    image += np.abs(np.nanmin(image))
                    if vmin[i] is None:
                        linthresh = 1e-2
                    else:
                        linthresh = vmin[i]
                    norm = colors.SymLogNorm(linthresh, base=10)
                else:
                    norm = None

                if image.dtype == bool:
                    image = image.astype(int)

                if custom_cmap[i] == 'binary' and image.max() == 1 and \
                   image.min() == 0:
                    cucmap = cmap_binary
                    cbticks = (0, 0.5, 1)

                else:
                    cucmap = custom_cmap[i]
                    cbticks = cbar_ticks[i]

                im = ax.imshow(image, cmap=cucmap, origin='lower', norm=norm,
                               interpolation='nearest', vmin=vmin[i],
                               vmax=vmax[i])

                if colorbar[i]:
                    divider = make_axes_locatable(ax)
                    # the width of cax is 5% of ax and the padding between cax
                    # and ax wis fixed at 0.05 inch
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cb = plt_colorbar(im, ax=ax, cax=cax, drawedges=False,
                                      ticks=cbticks)
                    cb.outline.set_linewidth(0.1)
                    cb.ax.tick_params(labelsize=lab_fontsize)

            else:
                # Leave the import to make porjection='3d' work
                #from mpl_toolkits.mplot3d import Axes3D
                x = np.outer(np.arange(0, frame_size, 1), np.ones(frame_size))
                y = x.copy().T
                ax = subplot(rows, cols, v, projection='3d')
                ax.set_aspect('auto')
                surf = ax.plot_surface(x, y, image, rstride=sampling,
                                       cstride=sampling, linewidth=2,
                                       cmap=custom_cmap[i], antialiased=True,
                                       vmin=vmin[i], vmax=vmax[i])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.dist = 10
                if title is not None:
                    ax.set_title(title)

                if colorbar[i]:
                    fig.colorbar(surf, aspect=10, pad=0.05, fraction=0.04)

            if ang_legend[i] and plot_mosaic:
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
                annotate(scalab, (xmi + scalabloc, scapad + 2), color='white', 
                         size=label_size)
            elif au_legend[i] and plot_mosaic:
                pxsc_fac = (0.012265/pxscale)
                labsz_fac = (label_size/12)
                scaleng = 50. / auscale
                scalab = '50 au'
                scalabloc = scaleng / 2. - 6.4*pxsc_fac*labsz_fac
                if scaleng > frame_size / 3.:
                    scaleng = 20. / auscale
                    scalab = '20 au'
                    scalabloc = scaleng / 2. - 6.4*pxsc_fac*labsz_fac
                    if scaleng > frame_size / 3.:
                        scaleng = 10. / auscale
                        scalab = '10 au'
                        scalabloc = scaleng / 2. - 6.4*pxsc_fac*labsz_fac
                scapad = 5*pxsc_fac*labsz_fac
                xma = frame_size - scapad
                xmi = xma - scaleng
                hlines(y=xma-scapad, xmin=xmi, xmax=xma, colors='white', lw=1.,
                       linestyles='solid')
                annotate(scalab, (xmi + scalabloc, xma-0.5*scapad), 
                         color='white', size=label_size)
                
            if show_circle and plot_mosaic:
                if isinstance(circle_linestyle,tuple):
                    c_offset = circle_linestyle[0]
                    circle_linestyle = circle_linestyle[1]
                else:
                    c_offset = lab_fontsize+1  # vertical offset is equal to the font size + 1, was 2
                for j in range(n_circ):
                    if isinstance(circle_color, (list, tuple)):
                        circle_color_tmp = circle_color[j]
                    else:
                        circle_color_tmp = circle_color
                    if isinstance(circle_linestyle, (list, tuple)):
                        circle_linestyle_tmp = circle_linestyle[j]
                    else:
                        circle_linestyle_tmp = circle_linestyle
                    circ = Circle(coor_circle[j], radius=circle_radius[j],
                                  fill=False, color=circle_color_tmp,
                                  alpha=circle_alpha[j], ls=circle_linestyle_tmp)
                    ax.add_artist(circ)
                    if circle_label:                  
                        x = coor_circle[j][0]
                        y = coor_circle[j][1]
                        if isinstance(circle_label,str):
                            cirlabel = circle_label
                        elif isinstance(circle_label,tuple):
                            cirlabel = circle_label[j]
                        else:
                            cirlabel = str(int(x))+','+str(int(y))
                        ax.text(x, y + circle_radius[j] + c_offset, cirlabel,
                                fontsize=lab_fontsize, color=circle_label_color, family='monospace',
                                ha='center', va='center', weight='bold',
                                alpha=circle_alpha[j])

            if show_cross and plot_mosaic:
                ax.axhline(coor_cross[0], xmin=0, xmax=frame_size, alpha=cross_alpha, lw=0.6,
                           linestyle='dashed', color='white')
                ax.axvline(coor_cross[1], ymin=0, ymax=frame_size, alpha=cross_alpha, lw=0.6,
                           linestyle='dashed', color='white')

            if show_center[i] and plot_mosaic:
                ax.scatter([cy], [cx], marker='+',
                           color=cross_color, alpha=cross_alpha)

            if show_arrow and plot_mosaic:
                ax.arrow(arrow[0] + arrow_length + arrow_shiftx, arrow[1],
                         -arrow_length, 0, color='white', head_width=6,
                         head_length=4, width=2, length_includes_head=True,
                         alpha=arrow_alpha)
                if arrow_label:                  
                    x = arrow[0]
                    y = arrow[1]
                    if isinstance(arrow_label,str):
                        arrlabel = arrow_label
                    else:
                        arrlabel = str(int(x))+','+str(int(y))
                    if len(arrlabel) < 5:
                        arr_fontsize=14
                    else:
                        arr_fontsize=lab_fontsize
                    ax.text(x + arrow_length + 1.3*arrow_shiftx, y, arrlabel,
                            fontsize=arr_fontsize, color='white', family='monospace',
                            ha='left', va='center', weight='bold',
                            alpha=arrow_alpha)
                                
            if label[i] is not None and plot_mosaic:
                ax.annotate(label[i], xy=(label_pad_x, label_pad_y), color=label_color,
                            xycoords='axes pixels', weight='bold',
                            size=label_size)

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
                        ticks.append((cy - t * ang_ticksep)-1)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

                # Corresponding distance in arcseconds, measured from the center
                labels_y = []
                labels_x = []
                for t in range(half_num_ticks, -half_num_ticks-1, -1):
                    labels_y.append(round(Decimal(-t * (ang_ticksep * pxscale)),ndec))
                    labels_x.append(round(Decimal(t * (ang_ticksep * pxscale)),ndec))
                ax.set_xticklabels(labels_x)
                ax.set_yticklabels(labels_y)
                ax.set_xlabel('\u0394RA["]', fontsize=label_size)
                ax.set_ylabel('\u0394Dec["]', fontsize=label_size)
                ax.tick_params(axis='both', which='major', labelsize=label_size)
            else:
                ax.set_xlabel("x", fontsize=label_size)
                ax.set_ylabel("y", fontsize=label_size)

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
        hv.extension(backend)
        subplots = []
        # options = "Image (cmap='" + custom_cmap[0] + "')"  # taking first item
        # hv.opts(options)

        for i, v in enumerate(range(num_plots)):
            image = data[i].copy()
            if vmin[i] is None:
                vmin[i] = image.min()
            if vmax[i] is None:
                vmax[i] = image.max()
            im = hv.Image((range(image.shape[1]), range(image.shape[0]), image))
            subplots.append(im.opts(tools=['hover'], colorbar=colorbar[i],
                                    colorbar_opts={'width': 15},
                                    width=width, height=height,
                                    clim=(vmin[i], vmax[i]),
                                    cmap=custom_cmap[0]))

        return hv.Layout(subplots).cols(cols)

    else:
        raise ValueError('`backend` not supported')


def plot_cubes(cube, mode='slider', backend='matplotlib', dpi=100,
               figtype='png', vmin=None, vmax=None, size=100, width=360,
               height=360, cmap=None, colorbar=True, dynamic=True,
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
    hv.extension(backend)

    if not isinstance(cube, np.ndarray):
        raise TypeError('`cube` must be a numpy.ndarray')

    if cmap is None:
        cmap = default_cmap

    if mode == 'slider':
        if cube.ndim not in (3, 4):
            raise ValueError('`cube` must be a 3 or 4 array when `mode` set to '
                             'slider')

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

        # Matplotlib takes None but not Bokeh. We take global min & max instead
        if vmin is None:
            vmin = cube.min()
        if vmax is None:
            vmax = cube.max()

        print(ds)
        print(":Cube_shape\t{}".format(list(cube.shape[::-1])))

        # not working for bokeh: dpi
        image_stack = ds.to(hv.Image, kdims=['x', 'y'], dynamic=dynamic)
        hv.output(backend=backend, size=size, dpi=dpi, fig=figtype,
                  max_frames=max_frames)

        if backend == 'matplotlib':
            # keywords in the currently active 'matplotlib' renderer are:
            # 'alpha', 'clims', 'cmap', 'filterrad', 'interpolation', 'norm',
            # 'visible'
            #options = "Image (cmap='" + cmap + "', interpolation='nearest',"
            #options += " clims=("+str(vmin)+','+str(vmax)+")"+")"
            #opts(options, image_stack)
            return image_stack.opts(opts.Image(colorbar=colorbar,
                                               cmap=cmap,
                                               clim=(vmin, vmax)))
            # hv.save(image_stack, 'holomap.gif', fps=5)

        elif backend == 'bokeh':
            #options = "Image (cmap='" + cmap + "')"
            #opts(options, image_stack)
            # Compensating the width to accommodate the colorbar
            if colorbar:
                cb_wid = 15
                cb_pad = 3
                tick_len = len(str(int(cube.max())))
                if tick_len < 4:
                    cb_tick = 25
                elif tick_len == 4:
                    cb_tick = 35
                elif tick_len > 4:
                    cb_tick = 45
                width_ = width + cb_pad + cb_wid + cb_tick
            else:
                width_ = width

            return image_stack.opts(opts.Image(colorbar=colorbar,
                                               colorbar_opts={'width': 15,
                                                              'padding': 3},
                                               width=width_, height=height,
                                               clim=(vmin, vmax),
                                               cmap=cmap,
                                               tools=['hover']))

    elif mode == 'animation':
        if cube.ndim != 3:
            raise ValueError('`cube` must be a 3 array when `mode` set to '
                             'animation')

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


