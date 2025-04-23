// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.cache;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/** A cache directory that stores the contents of fetched repos across different workspaces. */
public class RepoContentsCache {
  @Nullable private Path path;

  public void setPath(@Nullable Path path) {
    this.path = path;
  }

  public boolean isEnabled() {
    return path != null;
  }

  public Path recordedInputsFile(String predeclaredInputHash) {
    Preconditions.checkState(path != null);
    return path.getRelative(predeclaredInputHash + ".recorded_inputs");
  }

  public Path contentsDir(String predeclaredInputHash) {
    Preconditions.checkState(path != null);
    return path.getRelative(predeclaredInputHash);
  }
}
