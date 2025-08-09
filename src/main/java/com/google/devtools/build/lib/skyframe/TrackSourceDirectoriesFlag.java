// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

/**
 * A flag to enable / disable tracking of source directories. Uses a system property which can be
 * set via a startup flag. The intention is for this code to be temporary, so I didn't want to add a
 * permanent flag to startup options (and there's already --host_jvm_args, which we can use to roll
 * this out). The flag affects Skyframe dependencies, so it needs to clear the Skyframe graph - the
 * easiest way to do that is to restart the server, which is done when --host_jvm_args changes.
 */
public class TrackSourceDirectoriesFlag {
  private static final boolean TRACK_SOURCE_DIRECTORIES;

  static {
    TRACK_SOURCE_DIRECTORIES = !"0".equals(System.getProperty("BAZEL_TRACK_SOURCE_DIRECTORIES"));
  }

  public static boolean trackSourceDirectories() {
    return TRACK_SOURCE_DIRECTORIES;
  }

  // Private constructor to prevent instantiation.
  private TrackSourceDirectoriesFlag() {}
}
