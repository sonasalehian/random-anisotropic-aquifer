#!/bin/bash
sudo docker run -v $(pwd):/shared -w /shared -ti dolfinx/dolfinx:v0.7.1
