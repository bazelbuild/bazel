#!/bin/bash

set -o errexit
set -o verbose

if [[ "$OSTYPE" == "linux"* ]]; then
  bash mongo/build_linux.sh "$1" "$2" "$3"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  bash mongo/build_macos.sh "$1" "$2" "$3"
else
  rm -rf msys
  mkdir msys
  curl -L -O https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20241208.tar.xz
  tar -C msys -xf msys2-base-x86_64-20241208.tar.xz
  chmod -R 755 msys

  export PATH=$PWD/msys/msys64/usr/bin:$PATH
  msys/msys64/usr/bin/bash.exe mongo/build_windows.sh "$1" "$2" "$3"
fi
