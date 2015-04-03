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
    sudo brew install protobuf libarchive
else
    sudo apt-get update -qq
    sudo apt-get install -y protobuf-compiler libarchive-dev
fi
