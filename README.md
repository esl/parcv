parcv
=====

The demo is comprised of 2 subsystems:

* imgproc, which is a C binding to OpenCL transforms, along with the kernels we
  were experimenting with. This is the bulk of the code and what runs on the 
  Parallella machine.
* opencv_client is the code which runs on camera nodes. It captures video frames
  grayscales them and sends them to the Parallella.
* ocl is a lightweight binding to opencl which isn't particularly featureful. I
  left it here in case someone was interested in reading NIF code.

