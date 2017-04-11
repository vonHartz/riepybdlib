# RiePybDlib
This repository contains a Programming-by-Demonstration library for Riemannian Manifolds.

- [Installation](#installation)
- [Reference](#reference)
- [Compatibility](#compatibility)
- [Licence](#licence)

A quick tutorial on how to use the library can be found via:
https://www.martijnzeestraten.nl/media/html/riepybdlib_tutorial.html

# Installation

## Python2.7:

Install the required dependencies:
```
sudo apt-get update
sudo apt-get install git
sudo apt-get install python-numpy python-scipy python-matplotlib ipython 
```

Clone and install riepybdlib:
```
git clone https://gitlab.martijnzeestraten.nl/martijn/riepybdlib.git
cd riepybdlib
sudo python setup.py install
```

## Python3.x: 
Install the required dependencies:
```
sudo apt-get update
sudo apt-get install git
sudo apt-get install python3-numpy python3-scipy python3-matplotlib ipython3 
```
Clone and install riepybdlib:
```
git clone https://gitlab.martijnzeestraten.nl/martijn/riepybdlib.git
cd riepybdlib
sudo python3 setup.py install
```

# Reference
Did you find RiePybDlib usefull for your research? Please acknowledge the authors in any acedemic publication that used parts of these codes.

```
@article{Zeestraten2017,
	title = "An Approach for Imitation Learning on {R}iemannian Manifolds",
	author = "Zeestraten, M.J.A. and Havoutis, I. and Silv\'erio, J. and Calinon, S. and Caldwell, D. G.",
	journal = "{IEEE} Robotics and Automation Letters ({RA-L})",
	year = 2017,
	volume = 2,
	number = 3,
	pages = "1240--1247",
	month = "June",
}
```

# Compatibility
The code is developed to work with with both Python 2.7 and Python 3.x. Please report any compatibility issues.

# Licence
RiePybDlib is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.

RiePybDlib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RiePybDlib. If not, see http://www.gnu.org/licenses/.
