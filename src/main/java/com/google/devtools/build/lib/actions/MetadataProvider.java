// Copyright 2017 The Bazel Authors. All rights reserved.
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

import java.io.IOException;
import javax.annotation.Nullable;

/**
 * The interface for Action inputs metadata (Digest and size).
 *
 * NOTE: Implementations must be thread safe.
 */
public interface MetadataProvider {
  /**
   * Returns digest for the given artifact. This digest is current as of some time t >= the start of
   * the present build. If the artifact is an output of an action that already executed at time p,
   * then t >= p. Aside from these properties, t can be any value and may vary arbitrarily across
   * calls.
   *
   * <p>The return value is owned by the cache and must not be modified.
   *
   * @param input the input to retrieve the digest for
   * @return the artifact's digest or null if digest cannot be obtained (due to artifact
   *     non-existence, lookup errors, or any other reason)
   * @throws DigestOfDirectoryException in case {@code input} is a directory, or a symlink to a
   * directory.
   * @throws IOException If the file cannot be digested.
   */
  @Nullable
  FileArtifactValue getMetadata(ActionInput input) throws IOException;

  /** Looks up an input from its exec path. */
  @Nullable
  ActionInput getInput(String execPath);
}
