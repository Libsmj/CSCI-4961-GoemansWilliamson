#===============================================================================
#	 Contact us:
#	
#    Andrea Casini <andreacasini88@gmail.com>
#    Nicola Rebagliati <nicola.rebagliati@gmail.com>
#
#===============================================================================

Installation Notes

This project is written in Python language. In order to make this work you may 
need some extra libraries:

- Numpy, the fundamental package for scientific computing with Python;
  http://numpy.scipy.org/

- SciPy (pronounced "Sigh Pie") is open-source software for mathematics, 
  science, and engineering
  http://www.scipy.org/

- matplotlib is a python 2D plotting library which produces publication quality 
  figures in a variety of hardcopy formats and interactive environments 
  across platforms.
  http://matplotlib.sourceforge.net/

- NetworkX is a Python language software package for the creation, manipulation, 
  and study of the structure, dynamics, and functions of complex networks.
  http://networkx.lanl.gov/
 
To install these libraries in your Ubuntu system follow these steps.

Add to your sources.list file the following entry:
deb http://debs.astraw.com/ dapper/

~$ sudo gedit /etc/apt/sources.list

Then, type on your terminal:

~$ sudo apt-get install python-setuptools python-numpy python-matplotlib python-scipy
~$ sudo easy_install networkx

Finally, locate the project src folder and type:
~$ python main.py

If your're experiencing some trouble you should install the libraries from
the source package.
