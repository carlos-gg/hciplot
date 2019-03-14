import holoviews as hv
from holoviews import opts

hv.extension('bokeh', 'matplotlib')


def cube_plot(cube, backend='bokeh', dpi=80, figtype='png', vmin=None,
              vmax=None, size=145, width=350, height=300, cmap='viridis',
              colorbar=True, dynamic=True):
    """ Wrapper HoloViews for the visualization of multi-dimensional
    high-contrast imaging datacubes on Jupyterlab.
    
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



