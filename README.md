# Unwrapped Icosahedral Maps 
Create unwrapped icosahedral maps from equirectangular images.

![image](./assets/output.png)
## Installation
This code was developed using ```python 3.8```, however it should run on anything that has  >= ```python 3.6```.

To install the requirements, one can simply run:

```bash
$ pip install -r requiremets.py
```

## Usage
The base class that one can use is the ```IcosahedralSampler``` class.
Sample usage:
```python
from ico_sampler import IcosahedralSampler

eq_image = imread('./assets/0.png')
ico_sampler = IcosahedralSampler(resolution = 600)

# generate unwrapped maps (as presented above)
unwrapped_image = ico_sampler.unwrap(eq_image, face_offset=0)

# create the image of the triangular face
face_image = ico_sampler.get_face_image(face_no=0, eq_image=eq_image)

# sample face colors from an eq image
face_colors = ico_sampler.get_face_rgb(face_no=0, eq_image=eq_image)

```
One can run the provided [sample notebook](./examples.ipynb) to see exactly how the code works.

## Command line
This repository also contains a command line utility program that can convert an equirectangular image 
to an incosahedral projection map:

```bash
$ python unwrap.py --input=<path to input> \ 
                   --output=<path to output> \
                   --face_resolution=600 \ 
                   --face_offset=0
```

# Notes:
- the image may be grainy due to the sampling method, using a higher resolution(>600px / face) should diminish this effect

## TODOs
A list of TODOs that might be implemented in the future:
- [ ] add interpolation when asmpling the colors (current method: nearest)
- [ ] add a tutorial like notebook to go over spherical projections

## References
During the creation of this repository I hhave found the following articles to be useful:

- [http://www.paulbourke.net/panorama/icosahedral/](http://www.paulbourke.net/panorama/icosahedral/)
- [https://www.songho.ca/opengl/gl_sphere.html](https://www.songho.ca/opengl/gl_sphere.html)
- [https://en.wikipedia.org/wiki/Regular_icosahedron](https://en.wikipedia.org/wiki/Regular_icosahedron)
- [https://mathworld.wolfram.com/RegularIcosahedron.html](https://mathworld.wolfram.com/RegularIcosahedron.html)
