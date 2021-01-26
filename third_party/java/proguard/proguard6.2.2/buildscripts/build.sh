#!/bin/bash
#
# GNU/Linux build script for ProGuard.

cd $(dirname "$0")

# Standard modules.
../core/build.sh        && \
../retrace/build.sh     && \
../gui/build.sh         && \
../annotations/build.sh || exit 1

# Optional modules.
../gradle/build.sh
../ant/build.sh
../wtk/build.sh
