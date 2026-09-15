// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.bazel;

import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Runfiles utilities for the black box tests initialization */
public class RunfilesUtil {

  /**
   * Find a runtime location of a path
   *
   * @param path input path
   * @return runtime location as a {@link Path}
   * @throws IOException in case Runfiles can not access manifest or file system
   */
  static Path find(String path) throws IOException {
    return Paths.get(Runfiles.create().rlocation(path));
  }
}
