import holoviews as hv
from holoviews import opts

hv.extension('matplotlib')

def cube_plot(cube, fig_size=100, fig_dpi=80, fig_type='png', fig_vmin=None, fig_vmax=None, 
              fig_cmap='viridis', dynamic=True):
    """ Wrapper HoloViews for plotting multi-dimensional high-contrast imaging datacubes on 
    Jupyterlab.
    
    Parameters
    ----------
    cube : np.ndarray
        Input cube.
    fig_size : int, optional
        The percentage size of displayed output.
    fig_dpi : int, optional 
        The rendered dpi of the figure.
    fig_type : {'png', 'svg'}, str optional
        Type of output.
    fig_vmin : float, optional
        Min value.
    fig_vmax : float, optional
        Max value.
    fig_cmap : str, optional
        Colormap.
    dynamic : bool, optional
        When False, a HoloMap is created (slower and will take up a lot of RAM for large datasets). 
        If True, a DynamicMap is created instead.
        
    Notes
    -----
    http://holoviews.org/getting_started/Gridded_Datasets.html
    http://holoviews.org/user_guide/Gridded_Datasets.html
    http://holoviews.org/user_guide/Applying_Customizations.html
    """   
    if cube.ndim not in [3, 4]:
        raise TypeError('This function is intended for 3d and 4d HCI datacubes')

    if cube.ndim == 3:
        # Dataset((X, Y, Z), Data), where
        # X is a 1D array of shape M , 
        # Y is a 1D array of shape N and 
        # Z is a 1D array of shape O 
        # Data is a ND array of shape NxMxO 
        ds = hv.Dataset((range(cube.shape[2]), range(cube.shape[1]), range(cube.shape[0]), 
                         cube), ['x', 'y', 'time'], 'flux')
    elif cube.ndim == 4:
        # adding a lambda dimension
        ds = hv.Dataset((range(cube.shape[3]), range(cube.shape[2]), range(cube.shape[1]), 
                         range(cube.shape[0]), cube), ['x', 'y', 'time', 'lambda'], 'flux')

    print(ds)
    print(":Cube_shape\t{}".format(list(cube.shape[::-1])))
        
    image_stack = ds.to(hv.Image, kdims=['x', 'y'], dynamic=dynamic)
    hv.output(size=fig_size, dpi=fig_dpi, fig=fig_type, max_frames=100000)
    
    # keywords in the currently active 'matplotlib' renderer are: 
    # ['alpha', 'clims', 'cmap', 'filterrad', 'interpolation', 'norm', 'visible']
    options = "Image (cmap='"+fig_cmap+"', interpolation='nearest',"
    options += " alpha=1.0, clims=("+str(fig_vmin)+','+str(fig_vmax)+")"+")"
    opts(options, image_stack)
 
    return image_stack

