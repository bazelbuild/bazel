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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.server.FailureDetails.Filesystem;
import com.google.devtools.build.lib.vfs.DetailedIOException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Thrown when the contents of a file in an external repository served from the remote repo contents
 * cache are no longer available in the remote cache.
 *
 * <p>Nodes that fail because of such a file recover by returning the {@link
 * com.google.devtools.build.skyframe.SkyFunction.Reset} constructed by {@link RepoRewinding}. A
 * reader that can't rewind fails with the transient {@link Filesystem.Code#REMOTE_FILE_EVICTED}
 * detail, which has the command as a whole retried if it is preserved.
 */
public final class LostRemoteRepoFileException extends DetailedIOException {

  private final RepositoryName repo;
  private final String digest;
  @Nullable private final Label label;

  public LostRemoteRepoFileException(
      String message, IOException cause, RepositoryName repo, String digest) {
    this(message, cause, repo, digest, /* label= */ null);
  }

  private LostRemoteRepoFileException(
      String message,
      IOException cause,
      RepositoryName repo,
      String digest,
      @Nullable Label label) {
    super(message, cause, Filesystem.Code.REMOTE_FILE_EVICTED, Transience.TRANSIENT);
    this.repo = repo;
    this.digest = digest;
    this.label = label;
  }

  /**
   * Returns a copy of this exception that records the label the lost file was read through.
   *
   * <p>Resolving a label requests the fetch of its repo, which is what allows the failing node to
   * rewind that fetch.
   */
  public LostRemoteRepoFileException withLabel(Label label) {
    return new LostRemoteRepoFileException(getMessage(), this, repo, digest, label);
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

  /** The label the lost file was read through, if any. */
  @Nullable
  public Label getLabel() {
    return label;
  }
}
