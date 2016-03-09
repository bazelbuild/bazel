#!/bin/bash
# Copyright 2015 The Bazel Authors. All rights reserved.
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
#
# environment_plist generates a plist file that contains some
# environment variables of the host machine (like DTPlatformBuild
# or BuildMachineOSBuild) given a target platform.
#
# This script only runs on darwin and you must have Xcode installed.
#
# --output    - the path to place the output plist file.
# --platform  - the target platform, e.g. 'iphoneos' or 'iphonesimulator8.3'
#

set -u

while [[ $# > 1 ]]
do
key="$1"

case $key in
  --platform)
    PLATFORM="$2"
    shift
    ;;
  --output)
    OUTPUT="$2"
    shift
    ;;
  *)
    # unknown option
    ;;
esac
shift
done

PLATFORM_DIR=$(/usr/bin/xcrun --sdk "${PLATFORM}" --show-sdk-platform-path)
PLATFORM_PLIST="${PLATFORM_DIR}"/Info.plist
PLIST=$(mktemp -d "${TMPDIR:-/tmp}/bazel_environment.XXXXXX")/env.plist
trap 'rm -rf "${PLIST}"' ERR EXIT

os_build=$(/usr/bin/sw_vers -buildVersion)
compiler=$(/usr/libexec/PlistBuddy -c "print :DefaultProperties:DEFAULT_COMPILER" "${PLATFORM_PLIST}")
platform_version=$(/usr/bin/xcodebuild -version -sdk "${PLATFORM}" PlatformVersion)
sdk_build=$(/usr/bin/xcodebuild -version -sdk "${PLATFORM}" ProductBuildVersion)
platform_build=$"${sdk_build}"
xcode_build=$(/usr/bin/xcodebuild -version | grep Build | cut -d ' ' -f3)
xcode_version_string=$(/usr/bin/xcodebuild -version | grep Xcode | cut -d ' ' -f2)
xcode_version=$(/usr/bin/printf '%02d%d%d\n' $(echo "${xcode_version_string//./ }"))


/usr/bin/defaults write "${PLIST}" DTPlatformBuild -string ${platform_build:-""}
/usr/bin/defaults write "${PLIST}" DTSDKBuild -string ${sdk_build:-""}
/usr/bin/defaults write "${PLIST}" DTPlatformVersion -string ${platform_version:-""}
/usr/bin/defaults write "${PLIST}" DTXcode -string ${xcode_version:-""}
/usr/bin/defaults write "${PLIST}" DTXcodeBuild -string ${xcode_build:-""}
/usr/bin/defaults write "${PLIST}" DTCompiler -string ${compiler:-""}
/usr/bin/defaults write "${PLIST}" BuildMachineOSBuild -string ${os_build:-""}
cat "${PLIST}" > "${OUTPUT}"
