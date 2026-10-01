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

package com.google.devtools.build.lib.skyframe.rewinding;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.server.FailureDetails.Filesystem;
import com.google.devtools.build.lib.vfs.DetailedIOException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;

/**
 * Thrown when the contents of a file in an external repository served from the remote repo contents
 * cache are no longer available in the remote cache.
 *
 * <p>The file is recovered by rewinding the fetch of {@link #getRepo}, which runs the repo rule
 * again and uploads the repo's contents anew, so this is only thrown while rewinding is enabled. A
 * reader that can't rewind fails with the same transient {@link
 * Filesystem.Code#REMOTE_FILE_EVICTED} detail as when rewinding is disabled, so that readers
 * preserving the detail have the command as a whole retried.
 */
public final class LostRemoteRepoFileException extends DetailedIOException {

  private final RepositoryName repo;
  private final String digest;

  public LostRemoteRepoFileException(
      String message, IOException cause, RepositoryName repo, String digest) {
    super(message, cause, Filesystem.Code.REMOTE_FILE_EVICTED, Transience.TRANSIENT);
    this.repo = repo;
    this.digest = digest;
  }

  /** The canonical name of the repository whose refetch recovers the lost file. */
  public RepositoryName getRepo() {
    return repo;
  }

  /**
   * The digest of the lost file in the {@code hash/size} form that identifies lost inputs, so that
   * a lost file discovered while materializing an action input can be reported as such.
   */
  public String getDigest() {
    return digest;
  }
}
