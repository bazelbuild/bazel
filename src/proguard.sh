#!/bin/bash -x

set -euo pipefail

bazeljar=$1
outputjar=$2
proguardjar=$3
proguardconf=$4
embeddedjdk=$5

mv "$bazeljar" src/BazelServer_deploy.jar
tar xf "$embeddedjdk" -C src --strip-components=1
java -Dallow.incomplete.class.hierarchy -jar "$proguardjar" @"$proguardconf" -verbose
mv src/BazelServer_proguarded.jar "$outputjar"
