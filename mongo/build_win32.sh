#!/bin/bash

set -o errexit
set -o verbose

pacman-key --init
pacman-key --populate
pacman --noconfirm -S zip unzip patch

# build
bash mongo/generic_build.sh "$1" "$2" ".exe"
