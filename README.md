parcv
=====

This repository contains a collection of experiments we've conducted on the Parallella prototype,
details of which can be found [here](http://www.parallella.org/2013/05/25/explorations-in-erlang-with-the-parallela-a-prelude/)

The image processing demo mentioned in the blog post is comprised of 2 subsystems:

* `imgproc`, which is a C binding to OpenCL transforms, along with the kernels we
  were experimenting with. This is the bulk of the code and what runs on the Parallella machine.
* `opencv_client` is the code which runs on camera nodes. It captures video frames, performs
* grayscale conversion and sends them to the Parallella.

We've also included `ocl`, a lightweight binding to OpenCL. While it isn't particularly featureful,
it might serve as a useful example of writing NIF code.

# License

  parcv is licensed under the Apache License, Version 2.0 (the "License"); You may not use this
library except in compliance with the License.  You may obtain a copy of the License at

	  http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software distributed under the
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied.  See the License for the specific language governing permissions and limitations
under the License.
