#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# use PYTHON_EXECUTABLE from CMake to get the right python to execute
/usr/bin/python3 -u "$DIR"/protoc-gen-mypy.py
