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
 * set via a startup flag. This ensures that toggling the flag causes a server restart and discards
 * Skyframe state.
 *
 * <p>Note that Bazel and Blaze have different defaults. (We hope to one day make them the same.)
 */
public class TrackSourceDirectoriesFlag {
  // The default value, which differs between Bazel and Blaze (rewritten by Copybara).
  private static final boolean TRACK_SOURCE_DIRECTORIES_DEFAULT = true;

  // The effective value.
  private static final boolean TRACK_SOURCE_DIRECTORIES =
      switch (System.getProperty("BAZEL_TRACK_SOURCE_DIRECTORIES", "")) {
        case "0" -> false;
        case "1" -> true;
        default -> TRACK_SOURCE_DIRECTORIES_DEFAULT;
      };

  public static boolean trackSourceDirectories() {
    return TRACK_SOURCE_DIRECTORIES;
  }

  // Private constructor to prevent instantiation.
  private TrackSourceDirectoriesFlag() {}
}
