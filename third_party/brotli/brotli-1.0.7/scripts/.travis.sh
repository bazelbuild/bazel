#!/bin/bash

case "$1" in
    "before_install")
	;;
    "install")
	case "${TRAVIS_OS_NAME}" in
	    "osx")
		brew update
		brew install binutils

		case "${CC}" in
		    "gcc-"*)
			which ${CC} || brew install $(echo "${CC}" | sed 's/\-/@/') || brew link --overwrite $(echo "${CC}" | sed 's/\-/@/')
			;;
		esac

		case "${BUILD_SYSTEM}" in
		    "bazel")
			brew install bazel
			;;
		esac
		;;
	    "linux")
		case "${CC}" in
		    "pgcc")
			wget 'https://raw.githubusercontent.com/nemequ/pgi-travis/de6212d94fd0e7d07a6ef730c23548c337c436a7/install-pgi.sh'
			echo 'acd3ef995ad93cfb87d26f65147395dcbedd4c3c844ee6ec39616f1a347c8df5  install-pgi.sh' | sha256sum -c --strict || exit 1
			/bin/sh install-pgi.sh
			;;
		esac
		;;
	esac
	;;
    "script")
	case "${BUILD_SYSTEM}" in
	    "cmake")
		mkdir builddir && cd builddir
		CMAKE_FLAGS=
		if [ "${CROSS_COMPILE}" = "yes" ]; then
		    CMAKE_FLAGS="-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_RC_COMPILER=${RC_COMPILER}"
		fi
		cmake ${CMAKE_FLAGS} -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DENABLE_SANITIZER="${SANITIZER}" -DCMAKE_C_FLAGS="${CFLAGS}" .. || exit 1
		make VERBOSE=1 || exit 1
		ctest -V || exit 1
		;;
	    "python")
		python setup.py test
		;;
	    "maven")
		cd java/org/brotli
		mvn install && cd integration && mvn verify
		;;
	    "autotools")
		./bootstrap && ./configure && make
		;;
	    "fuzz")
		./c/fuzz/test_fuzzer.sh
		;;
	    "bazel")
		bazel build -c opt ...:all &&
		cd go && bazel test -c opt ...:all && cd .. &&
		cd java && bazel test -c opt ...:all && cd .. &&
		cd js && bazel test -c opt ...:all && cd .. &&
		cd research && bazel build -c opt ...:all && cd ..
		;;
	esac
	;;
    "after_success")
	;;
    "before_deploy")
	case "${BUILD_SYSTEM}" in
	    "bazel")
		export RELEASE_DATE=`date +%Y-%m-%d`
		perl -p -i -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' scripts/.bintray.json
		zip -j9 brotli.zip bazel-bin/libbrotli*.a bazel-bin/libbrotli*.so bazel-bin/brotli
		;;
	esac
	;;
esac
