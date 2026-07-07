// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.vfs;

import javax.annotation.Nullable;

/**
 * Implemented by {@link FileSystem}s that serve the contents of external repositories from a
 * remote cache and support recovering from the remote cache losing the contents of individual
 * files by refetching the repository.
 */
public interface RewindableRepoFileSystem {
  /**
   * Marks the file at the given path as lost, i.e., its contents are no longer available in the
   * remote cache.
   *
   * <p>If the file can be recovered by rewinding (in the sense of {@code --rewind_lost_inputs})
   * the Skyframe node that fetches the repository containing it, returns the canonical name of
   * that repository. Otherwise, returns null.
   */
  @Nullable
  String markLostRepoFile(PathFragment path);
}
