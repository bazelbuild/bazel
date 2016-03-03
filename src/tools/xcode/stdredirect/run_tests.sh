#!/bin/sh

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

# Runs a series of unit tests for an app bundle.
# This is an example script to show how to use the StdRedirect library.

# $1 path to app bundle
# $2 path to xctest bundle
# $3 path to stderr
# $4 path to StdRedirect.dylib

set -e

if [ $# -ne 4 ]; then
  echo "Usage RunTests <app bundle path> <xctest bundle path> <stderr path> <redirect library path>"
  exit 1
fi

SIMULATOR_RUNNING=$(osascript -e "tell application \"System Events\" to (name of processes) contains \"iOS Simulator\"")

TEST_DEVICE_ID=$(xcrun simctl create TestDevice com.apple.CoreSimulator.SimDeviceType.iPhone-6 com.apple.CoreSimulator.SimRuntime.iOS-8-3)

# Instruments will return an error because we are calling it without a template arg.
# It's the only way I know of to launch the simulator safely using xcrun.
# This will launch the simulator with a given device. If the simulator is already running
# it will switch to the given device.
# Radar 21392428 xcrun should allow me to specifiy "iOS Simulator" in some manner
xcrun instruments -w $TEST_DEVICE_ID &>/dev/null || true

xcrun simctl install $TEST_DEVICE_ID $1

PLATFORM_PATH="$(xcrun --sdk iphonesimulator --show-sdk-platform-path)"
export SIMCTL_CHILD_DYLD_INSERT_LIBRARIES="$PLATFORM_PATH/Developer/Library/PrivateFrameworks/IDEBundleInjection.framework/IDEBundleInjection:$4"
export SIMCTL_CHILD_GSTDERR="$3"
export SIMCTL_CHILD_XCInjectBundle="$2"
BUNDLE_BASE=$(basename $1)
BUNDLE_INFO_PLIST="$1/Info.plist"
EXECUTABLE_NAME=$(xcrun PlistBuddy -c "Print :CFBundleExecutable" "$BUNDLE_INFO_PLIST")
BUNDLE_ID=$(xcrun PlistBuddy -c "Print :CFBundleIdentifier" "$BUNDLE_INFO_PLIST")

# The "*" is unfortunate, but there is no way to get back the UUID of the installed application.
# Since we created the simulator from scratch, there should only be one app installed on it.
# Radar 21392479 simctl install should return the UUID of the installed app.
# Radar 21392325 simctl getenv never appears to function
export SIMCTL_CHILD_XCInjectBundleInto="$HOME/Library/Developer/CoreSimulator/Devices/$TEST_DEVICE_ID/data/Containers/Bundle/Application/*/$BUNDLE_BASE/$EXECUTABLE_NAME"
export SIMCTL_CHILD_DYLD_FALLBACK_FRAMEWORK_PATH="$PLATFORM_PATH/Developer/Library/Frameworks"
IOS_PID=$(xcrun simctl launch $TEST_DEVICE_ID "$BUNDLE_ID" -XCTest All)
IOS_PID=$(echo $IOS_PID | awk '{ print $2 }')

# The simulator is not a subprocess of the script, so we cannot wait on it and must poll instead.
while kill -0 "$IOS_PID" &>/dev/null; do
  sleep 0.5
done

# If the simulator wasn't running when we started, then we should clean it up.
if [ "${SIMULATOR_RUNNING}" = false ]; then
  osascript -e "tell application \"iOS Simulator\" to quit"
fi

# Radar 21392585 simctl delete should allows me to delete multiple devices in one call
xcrun simctl delete $TEST_DEVICE_ID
