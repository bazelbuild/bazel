#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eux

if [ -z "${TRAVIS_OS_NAME+x}" ]; then
    echo "TRAVIS_OS_NAME not set, set it to 'linux' or 'osx' to run locally."
    exit 1
fi

if [[ $TRAVIS_OS_NAME = 'osx' ]]; then
    export JAVA_VERSION=1.7
    sed -i.bak 's/_version = "8",/_version = "7",/' tools/jdk/BUILD
    cat .travis/jdk7.WORKSPACE >WORKSPACE
    # Ignore zip tests as they requires to much space and jdk8 stuff
    cat <<'EOF' >.bazelrc
build --test_tag_filters -zip,-jdk8,-skyframe
EOF
    export BAZELRC=$PWD/.bazelrc
    ./compile.sh all
else
    sudo apt-get update -qq
    sudo apt-get install -y netcat-traditional
    sudo update-alternatives --set nc /bin/nc.traditional
    export JAVA_HOME=/usr/lib/jvm/java-8-oracle
    export JAVA_OPTS="-Xmx3000m"
    cat > .bazelrc <<EOF
test --ram_utilization_factor=10
EOF
    export BAZELRC="$(pwd)/.bazelrc"
    ./compile.sh all
fi
