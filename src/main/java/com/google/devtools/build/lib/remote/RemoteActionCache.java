// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import java.io.IOException;
import java.util.Collection;
import javax.annotation.Nullable;

/** A cache for storing artifacts (input and output) as well as the output of running an action. */
@ThreadCompatible
interface RemoteActionCache {
  // CAS API

  // TODO(buchgr): consider removing the CacheNotFoundException, and replacing it with other
  // ways to signal a cache miss.

  /**
   * Ensures that the tree structure of the inputs, the input files themselves, and the command are
   * available in the remote cache, such that the tree can be reassembled and executed on another
   * machine given the root digest.
   *
   * <p>The cache may check whether files or parts of the tree structure are already present, and do
   * not need to be uploaded again.
   *
   * <p>Note that this method is only required for remote execution, not for caching itself.
   * However, remote execution uses a cache to store input files, and that may be a separate
   * end-point from the executor itself, so the functionality lives here. A pure remote caching
   * implementation that does not support remote execution may choose not to implement this
   * function, and throw {@link UnsupportedOperationException} instead. If so, it should be clearly
   * documented that it cannot be used for remote execution.
   */
  void ensureInputsPresent(
      TreeNodeRepository repository, Path execRoot, TreeNode root, Command command)
          throws IOException, InterruptedException;

  /**
   * Download the output files and directory trees of a remotely executed action to the local
   * machine, as well stdin / stdout to the given files.
   */
  // TODO(olaola): will need to amend to include the TreeNodeRepository for updating.
  void download(ActionResult result, Path execRoot, FileOutErr outErr)
      throws IOException, InterruptedException, CacheNotFoundException;
  /**
   * Attempts to look up the given action in the remote cache and return its result, if present.
   * Returns {@code null} if there is no such entry. Note that a successful result from this method
   * does not guarantee the availability of the corresponding output files in the remote cache.
   */
  @Nullable
  ActionResult getCachedActionResult(ActionKey actionKey) throws IOException, InterruptedException;

  /**
   * Upload the result of a locally executed action to the cache by uploading any necessary files,
   * stdin / stdout, as well as adding an entry for the given action key to the cache.
   */
  void upload(ActionKey actionKey, Path execRoot, Collection<Path> files, FileOutErr outErr)
      throws IOException, InterruptedException;

  /** Release resources associated with the cache. The cache may not be used after calling this. */
  void close();
}
