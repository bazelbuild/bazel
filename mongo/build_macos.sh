#!/bin/bash

set -o errexit
set -o verbose

bash mongo/generic_build.sh "$1" "$2"
