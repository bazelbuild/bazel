// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
//  StdRedirect.c
//
//  Used for controlling stdin, stdout, stderr of iOS Applications launched in the simulator
//  using simctl. stdin, stdout, and stderr are set using environment variables.
//
//  To use:
//
//  export GSTDIN="PATH_TO_STD_IN"
//  export GSTDOUT="PATH_TO_STD_OUT"
//  export GSTDERR="PATH_TO_STD_ERR"
//  PLATFORM_PATH="$($(xcrun --sdk iphonesimulator --show-sdk-platform-path)"
//  export SIMCTL_CHILD_DYLD_FALLBACK_FRAMEWORK_PATH="$PLATFORM_PATH/Developer/Library/Frameworks"
//  export SIMCTL_CHILD_DYLD_INSERT_LIBRARIES="$PLATFORM_PATH/Developer/Library/PrivateFrameworks" \
//      "/IDEBundleInjection.framework/IDEBundleInjection:<Full path to StdRedirect.dylib>"
//  export SIMCTL_CHILD_XCInjectBundle="Full path to your *.xctest Bundle"
//  export SIMCTL_CHILD_XCInjectBundleInto="Full path to your app binary inside of " \
//      "~/Library/Developer/CoreSimulator/Devices"
//  <Launch the simulator in some fashion>
//  xcrun simctl launch booted <device> <app binary bundle ID> -XCTest All

//  Note that all of GSTDIN/GSTDOUT/GSTDERR are optional. Xcode dumps test results to GSTDERR.

//  For a practical example of using it see run_tests.sh

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <sys/stat.h>
#include <unistd.h>

static void SetUpStdFileDescriptor(const char *env_name, int fileNo) {
  const char *path = getenv(env_name);
  if (path) {
    int fd = open(path, O_RDWR | O_CREAT | O_APPEND);
    if (fd == -1) {
      syslog(LOG_ERR, "Could not open %s for %s - %s", env_name, path, strerror(errno));
    } else {
      if (fchmod(fd, 0666) == -1) {
        syslog(LOG_ERR, "Could not chmod %s for %s - %s", env_name, path, strerror(errno));
      }
      if (dup2(fd, fileNo) == -1) {
        syslog(LOG_ERR, "Could not dup %s for %s - %s", env_name, path, strerror(errno));
      }
    }
  }
}

__attribute__((constructor)) static void SetUpStdFileDescriptors() {
  SetUpStdFileDescriptor("GSTDIN", STDIN_FILENO);
  SetUpStdFileDescriptor("GSTDOUT", STDOUT_FILENO);
  SetUpStdFileDescriptor("GSTDERR", STDERR_FILENO);
}
