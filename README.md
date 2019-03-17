# hciplot

``HCIplot`` -- High-contrast Imaging Plotting library. The goal of this
library is to be the "Swiss army" solution for plotting and visualizing 
multi-dimensional high-contrast imaging datacubes on ``Jupyter lab``. 
While visualizing FITS files is straightforward with SaoImage DS9 or any
other FITS viewer, exploring the content of an HCI datacube as an 
in-memory ``numpy`` array (for example when running your Jupyter session
on a remote machine) is far from easy. 

``HCIplot`` contains two functions, ``plot_frames`` and ``plot_cubes``,
and relies on the ``matplotlib`` and ``HoloViews`` libraries and 
``ImageMagick``. With ``HCIplot`` you can:

* plot a single 2d array or create a mosaic of several 2d arrays,  
* annotate save publication ready images,
* visualize 2d arrays as surface plots,
* create interactive plots when handling 3d or 4d arrays (thanks to 
``HoloViews``,
* save to disk a 3d array as an animation (gif or mp4).