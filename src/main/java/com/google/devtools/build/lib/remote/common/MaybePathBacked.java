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
package com.google.devtools.build.lib.remote.common;

import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * An interface to mark {@link java.io.OutputStream}s that may be known to write to an associated
 * {@link Path}.
 */
public interface MaybePathBacked {
  /** If this stream is backed by a Path, returns that Path. Otherwise, returns null. */
  @Nullable
  Path maybeGetPath();

  /**
   * Returns the path where the content written to this stream is expected to remain after the
   * enclosing operation completes, or {@code null} if unknown.
   *
   * <p>This differs from {@link #maybeGetPath} when the stream writes to a temporary staging
   * location that is later moved into place, e.g. when downloading an action output.
   */
  @Nullable
  default Path maybeGetFinalPath() {
    return maybeGetPath();
  }
}
