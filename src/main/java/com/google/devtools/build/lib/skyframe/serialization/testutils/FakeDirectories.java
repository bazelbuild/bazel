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
package com.google.devtools.build.lib.skyframe.serialization.testutils;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.vfs.Path;

/** Helper providing an {@link FsUtils#TEST_FILESYSTEM} backed {@link BlazeDirectories} instance. */
public class FakeDirectories {
  public static final Path OUTPUT_USER_BASE =
      FsUtils.TEST_FILESYSTEM.getPath("/output_root/_bazel_testuser");

  public static final Path OUTPUT_BASE =
      OUTPUT_USER_BASE.getRelative("ba5eba11ba5eba11ba5eba11ba5eba11");

  public static final ServerDirectories SERVER_DIRECTORIES =
      new ServerDirectories(/*installBase=*/ null, OUTPUT_BASE, OUTPUT_USER_BASE);

  public static final BlazeDirectories BLAZE_DIRECTORIES =
      new BlazeDirectories(
          SERVER_DIRECTORIES,
          OUTPUT_BASE.getRelative("execroot/io_bazel"),
          /* defaultSystemJavabase= */ null,
          "bazel");

  private FakeDirectories() {}
}
