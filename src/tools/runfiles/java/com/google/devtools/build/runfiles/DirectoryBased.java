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

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** {@link Runfiles} implementation that appends runfiles paths to the runfiles root. */
final class DirectoryBased extends Runfiles {
  private final String runfilesRoot;

  DirectoryBased(String runfilesDir) throws IOException {
    Util.checkArgument(!Util.isNullOrEmpty(runfilesDir));
    Util.checkArgument(new File(runfilesDir).isDirectory());
    this.runfilesRoot = runfilesDir;
  }

  @Override
  String rlocationChecked(String path) {
    return runfilesRoot + "/" + path;
  }

  @Override
  public Map<String, String> getEnvVars() {
    HashMap<String, String> result = new HashMap<>(2);
    result.put("RUNFILES_DIR", runfilesRoot);
    // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can pick up RUNFILES_DIR.
    result.put("JAVA_RUNFILES", runfilesRoot);
    return result;
  }
}
