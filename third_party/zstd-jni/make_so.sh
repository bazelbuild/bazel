#!/bin/bash

compile () {
    ARCH=$1
    OS=$2
    CC=${3:-gcc}
    HOST="gcc-$ARCH-$OS"
    echo "Compiling for $ARCH/$OS on $HOST"
    INSTALL=target/classes/$OS/$ARCH
    rsync -r --delete jni $HOST:
    rsync -r --delete src/main/native $HOST:
    ssh $HOST 'export PATH=$HOME/bin:$PATH; '$CC' -shared -DDYNAMIC_BMI2=0 -fPIC -O3 -DZSTD_LEGACY_SUPPORT=4 -DZSTD_MULTITHREAD=1 -I/usr/include -I./jni -I./native -I./native/common -I./native/legacy -std=c99 -lpthread -o libzstd-jni.so native/*.c native/legacy/*.c native/common/*.c native/compress/*.c native/decompress/*.c native/dictBuilder/*.c'
    mkdir -p $INSTALL
    scp $HOST:libzstd-jni.so $INSTALL
}

compile amd64   linux
compile i386    linux
compile ppc64   linux
compile ppc64le linux
compile ppc64   aix
compile aarch64 linux
compile mips64  linux
compile amd64   freebsd cc
