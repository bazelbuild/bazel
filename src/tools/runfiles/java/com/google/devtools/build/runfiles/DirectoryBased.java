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

package com.google.devtools.build.runfiles;

/** {@link Runfiles} implementation that appends runfiles paths to the runfiles root. */
class DirectoryBased extends Runfiles {
  private final String runfilesRoot;

  DirectoryBased(String runfilesDir) {
    Util.checkArgument(runfilesDir != null);
    Util.checkArgument(!runfilesDir.isEmpty());
    this.runfilesRoot = runfilesDir;
  }

  @Override
  String rlocationChecked(String path) {
    return runfilesRoot + "/" + path;
  }
}
