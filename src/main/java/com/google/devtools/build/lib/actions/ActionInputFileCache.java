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

import java.io.IOException;

import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * The interface for Action inputs metadata (Digest and size).
 *
 * NOTE: Implementations must be thread safe.
 */
@ThreadSafe
public interface ActionInputFileCache {
  /**
   * Returns digest for the given artifact. This digest is current as of some time t >= the start of
   * the present build. If the artifact is an output of an action that already executed at time p,
   * then t >= p. Aside from these properties, t can be any value and may vary arbitrarily across
   * calls.
   *
   * The return value is owned by the cache and must not be modified.
   *
   * @param input the input to retrieve the digest for
   * @return the artifact's digest or null if digest cannot be obtained (due to artifact
   *         non-existence, lookup errors, or any other reason)
   *
   * @throws DigestOfDirectoryException in case {@code input} is a directory.
   * @throws IOException If the file cannot be digested.
   *
   */
  @Nullable
  byte[] getDigest(ActionInput input) throws IOException;

  /**
   * Retrieves whether or not the input Artifact is a file or symlink to an existing file.
   *
   * @param input the input
   * @return true if input is a file or symlink to an existing file, otherwise false
   */
  boolean isFile(Artifact input);

  /**
   * Retrieve the size of the file at the given path. Will usually return 0 on failure instead of
   * throwing an IOException. Returns 0 for files inaccessible to user, but available to the
   * execution environment.
   *
   * @param input the input.
   * @return the file size in bytes.
   * @throws IOException on failure.
   */
  long getSizeInBytes(ActionInput input) throws IOException;

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
