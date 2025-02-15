#!/bin/bash

set -o errexit
set -o verbose

# setup msys
rm -rf msys
mkdir msys
cd msys
curl -L -O https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20241208.tar.xz
tar xf msys2-base-x86_64-20241208.tar.xz
chmod -R 755 msys64
