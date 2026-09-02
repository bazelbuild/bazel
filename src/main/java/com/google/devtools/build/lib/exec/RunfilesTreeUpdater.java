// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.exec;

import com.google.common.base.Throwables;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Utility used in local execution to create a runfiles tree if {@code --nobuild_runfile_links} has
 * been specified.
 *
 * <p>It is safe to call {@link #updateRunfiles} concurrently.
 */
@ThreadSafe
public class RunfilesTreeUpdater {
  private final Path execRoot;

  /**
   * Deduplicates multiple attempts to update the same runfiles tree.
   *
   * <p>Attempts may occur concurrently, e.g. if multiple local actions have the same input.
   *
   * <p>The presence of an entry in the map signifies that an earlier attempt to update the
   * corresponding runfiles tree was started, and will (have) set the future upon completion.
   */
  private final ConcurrentHashMap<PathFragment, CompletableFuture<Void>> updatedTrees =
      new ConcurrentHashMap<>();

  public RunfilesTreeUpdater(Path execRoot) {
    this.execRoot = execRoot;
  }

  /** Creates or updates input runfiles trees for a spawn. */
  public void updateRunfiles(Iterable<RunfilesTree> runfilesTrees)
      throws ExecException, IOException, InterruptedException {
    for (RunfilesTree tree : runfilesTrees) {
      PathFragment runfilesDir = tree.getExecPath();
      if (tree.isBuildRunfileLinks()) {
        continue;
      }

      var freshFuture = new CompletableFuture<Void>();
      CompletableFuture<Void> priorFuture = updatedTrees.putIfAbsent(runfilesDir, freshFuture);

      if (priorFuture == null) {
        // We are the first attempt; update the runfiles tree and mark the future complete.
        try {
          updateRunfilesTree(tree);
          freshFuture.complete(null);
        } catch (Exception e) {
          freshFuture.completeExceptionally(e);
          throw e;
        }
      } else {
        // There was a previous attempt; wait for it to complete.
        try {
          priorFuture.join();
        } catch (CompletionException e) {
          Throwable cause = e.getCause();
          if (cause != null) {
            Throwables.throwIfInstanceOf(cause, ExecException.class);
            Throwables.throwIfInstanceOf(cause, IOException.class);
            Throwables.throwIfInstanceOf(cause, InterruptedException.class);
            Throwables.throwIfUnchecked(cause);
          }
          throw new AssertionError("Unexpected exception", e);
        }
      }
    }
  }

  private void updateRunfilesTree(RunfilesTree tree) throws IOException, ExecException {
    Path runfilesDir = execRoot.getRelative(tree.getExecPath());
    Path inputManifest =
        execRoot.getRelative(RunfilesSupport.inputManifestExecPath(tree.getExecPath()));
    if (!inputManifest.exists()) {
      return;
    }
    Path outputManifest =
        execRoot.getRelative(RunfilesSupport.outputManifestExecPath(tree.getExecPath()));
    // Note that the runfiles directory is not checked for being up to date here: the only cheap
    // signal available for that is the output manifest matching the input manifest, which merely
    // states that the *set* of runfiles is unchanged. That implies that the tree is up to date only
    // if it consists of symbolic links, whose targets are the authoritative files. It does not on a
    // file system that materializes symlinks as copies (Windows without --windows_enable_symlinks),
    // where the contents of the tree can be stale even though the manifest is unchanged - and since
    // linkManifest() makes the output manifest a symbolic link on every other file system, that is
    // the only situation in which such a check would ever apply. Recreating the tree is cheap if it
    // is already up to date: SymlinkTreeHelper only mutates entries that don't match.

    if (!runfilesDir.exists()) {
      runfilesDir.createDirectoryAndParents();
    }

    SymlinkTreeHelper helper =
        new SymlinkTreeHelper(inputManifest, outputManifest, runfilesDir, tree.getWorkspaceName());

    switch (tree.getSymlinksMode()) {
      case CREATE -> {
        helper.createRunfilesSymlinks(tree.getMapping());
        helper.linkManifest();
      }
      case SKIP -> helper.createMinimalRunfilesDirectory();
    }
  }
}
