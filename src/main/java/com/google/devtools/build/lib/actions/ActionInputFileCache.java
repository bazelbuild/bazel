// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * The interface for Action inputs metadata (Digest and size).
 *
 * NOTE: Implementations must be thread safe.
 */
@ThreadSafe
public interface ActionInputFileCache extends MetadataProvider {
  /**
   * Checks if the file is available locally, based on the assumption that previous operations on
   * the ActionInputFileCache would have created a cache entry for it.
   *
   * @param digest the digest to lookup (as lowercase hex).
   * @return true if the specified digest is backed by a locally-readable file, false otherwise
   */
  boolean contentsAvailableLocally(ByteString digest);

  /**
   * Concrete subclasses must implement this to provide a mapping from digest to file path,
   * based on files previously seen as inputs.
   *
   * @param digest the digest (as lowercase hex).
   * @return an ActionInput corresponding to the given digest.
   */
  @Nullable
  ActionInput getInputFromDigest(ByteString digest);

  /**
   * The absolute path that this input is located at. The usual {@link ActionInput} implementation
   * is {@link Artifact}, which currently embeds its full path, so implementations should just
   * return this path if {@code input} is an {@link Artifact}. Otherwise, implementations should
   * resolve the relative path into an absolute one and return that.
   */
  Path getInputPath(ActionInput input);
}
