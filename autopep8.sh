#!/bin/bash

PROJECT_ROOT_PATH=`git rev-parse --show-toplevel`

autopep8 --in-place --jobs 32 ${PROJECT_ROOT_PATH}/*.py
