#!/bin/bash
docker run -v $(pwd):/shared -w /shared -ti dolfinx/dolfinx:nightly
