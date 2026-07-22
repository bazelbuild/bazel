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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;

/**
 * A file system that may keep the contents of certain directory trees in memory, backed by remote
 * storage, and that supports materializing them to the local file system on demand.
 */
public interface SubtreeMaterializer {
  /**
   * Materializes the subtree rooted at the given path to the local file system if its contents are
   * currently only available in memory, together with the targets of any symlinks below it.
   *
   * <p>Does nothing if the subtree is already backed by the local file system.
   */
  void ensureSubtreeMaterialized(PathFragment path) throws IOException, InterruptedException;
}
