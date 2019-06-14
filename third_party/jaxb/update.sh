#!/bin/bash

set -euxo pipefail

mkdir srcs
cd srcs
jar xvf ../jaxb-api-2.3.1-sources.jar
patch -p1 < ../remove-java.desktop.patch
rm -rf META-INF
find . -type f -name \*.java -print0 | xargs -0 javac
find . -type f -name \*.class -print0 | xargs -0 jar -cvf ../jaxb-api-2.3.1-patched.jar
