#!/bin/bash -x

set -euo pipefail

bazeljar=$1
outputjar=$2
proguardjar=$3
proguardconf=$4
embeddedjdk=$5

UNAME=$(uname -s | tr 'A-Z' 'a-z')

if [[ "$UNAME" =~ msys_nt* ]]; then
  # FIXME
  mv "$bazeljar" "$outputjar"
  exit 0
fi

mv "$bazeljar" src/BazelServer_deploy.jar
tar xf "$embeddedjdk" -C src --strip-components=1
java -Dallow.incomplete.class.hierarchy -jar "$proguardjar" @"$proguardconf" -verbose || mv src/BazelServer_deploy.jar src/BazelServer_proguarded.jar
mv src/BazelServer_proguarded.jar "$outputjar"
