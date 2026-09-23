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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Splitter;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.XattrProvider;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Utility used to create a runfiles tree on demand rather than during the build: before a local
 * action that has it as an input is executed, before the {@code run} command executes a target, and
 * for top-level targets if the output service defers runfiles-tree creation.
 *
 * <p>It is safe to call {@link #updateRunfiles} concurrently.
 */
@ThreadSafe
public class RunfilesTreeUpdater {
  private final Path execRoot;
  private final XattrProvider xattrProvider;
  private volatile boolean materializeBuiltRunfilesTrees;

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

  public RunfilesTreeUpdater(Path execRoot, XattrProvider xattrProvider) {
    this.execRoot = execRoot;
    this.xattrProvider = xattrProvider;
  }

  /** Enables materialization of runfiles trees built with {@code --build_runfile_links}. */
  public void setMaterializeBuiltRunfilesTrees() {
    materializeBuiltRunfilesTrees = true;
  }

  /** Creates or updates the given runfiles trees. */
  public void updateRunfiles(Iterable<RunfilesTree> runfilesTrees)
      throws ExecException, IOException, InterruptedException {
    for (RunfilesTree tree : runfilesTrees) {
      PathFragment runfilesDir = tree.getExecPath();
      // Runfiles trees built with --build_runfile_links have already been created by
      // SymlinkTreeAction during the build unless the output service defers that to this class.
      if (tree.isBuildRunfileLinks() && !materializeBuiltRunfilesTrees) {
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

  /**
   * Returns whether the given runfiles tree exists and is up to date with its input manifest, in
   * which case {@link #updateRunfiles} leaves it unchanged.
   *
   * <p>This is only ever the case for a tree that was previously populated by this class, which
   * copies the input manifest into the tree after creating the symlinks. A tree that only contains
   * an output manifest symlinked to the input manifest, as created by {@code SymlinkTreeAction}
   * when it doesn't create the symlinks itself, is never up to date.
   */
  public boolean isUpToDate(RunfilesTree tree) {
    Path runfilesDir = execRoot.getRelative(tree.getExecPath());
    Path inputManifest =
        execRoot.getRelative(RunfilesSupport.inputManifestExecPath(tree.getExecPath()));
    Path outputManifest =
        execRoot.getRelative(RunfilesSupport.outputManifestExecPath(tree.getExecPath()));
    try {
      var inputManifestStat = inputManifest.statIfFound();
      return inputManifestStat != null
          && isUpToDate(tree, runfilesDir, inputManifest, inputManifestStat, outputManifest);
    } catch (IOException e) {
      return false;
    }
  }

  private boolean isUpToDate(
      RunfilesTree tree,
      Path runfilesDir,
      Path inputManifest,
      FileStatus inputManifestStat,
      Path outputManifest)
      throws IOException {
    // The runfiles directory is up to date if the manifest in it matches the input manifest,
    // implying the symlinks exist and are already up to date. If the output manifest is a symbolic
    // link, it is a symbolic link to the input manifest created by SymlinkTreeAction, so we cannot
    // trust it as an up-to-date check.
    // On Windows, where symlinks may be silently replaced by copies, a previous run in SKIP mode
    // could have resulted in an output manifest that is an identical copy of the input manifest,
    // which we must not treat as up to date, but we also don't want to unnecessarily rebuild the
    // runfiles directory all the time. Instead, check for the presence of the first runfile in
    // the manifest. If it is present, we can be certain that the previous mode wasn't SKIP.
    if (tree.getSymlinksMode() != RunfileSymlinksMode.CREATE) {
      return false;
    }
    // Not following symlinks means that the stat describes the output manifest itself, which is
    // only the file we digest below if it isn't a symbolic link - which is checked first.
    var outputManifestStat = outputManifest.statIfFound(Symlinks.NOFOLLOW);
    return outputManifestStat != null
        && !outputManifestStat.isSymbolicLink()
        && Arrays.equals(
            DigestUtils.getDigestWithManualFallback(
                outputManifest, xattrProvider, outputManifestStat),
            DigestUtils.getDigestWithManualFallback(
                inputManifest, xattrProvider, inputManifestStat))
        && (OS.getCurrent() != OS.WINDOWS
            || isRunfilesDirectoryPopulated(runfilesDir, outputManifest));
  }

  private void updateRunfilesTree(RunfilesTree tree) throws IOException, ExecException {
    Path runfilesDir = execRoot.getRelative(tree.getExecPath());
    Path inputManifest =
        execRoot.getRelative(RunfilesSupport.inputManifestExecPath(tree.getExecPath()));
    var inputManifestStat = inputManifest.statIfFound();
    if (inputManifestStat == null) {
      return;
    }
    Path outputManifest =
        execRoot.getRelative(RunfilesSupport.outputManifestExecPath(tree.getExecPath()));
    try {
      if (isUpToDate(tree, runfilesDir, inputManifest, inputManifestStat, outputManifest)) {
        return;
      }
    } catch (IOException e) {
      // Ignore it - we will just try to create runfiles directory.
    }

    if (!runfilesDir.exists()) {
      runfilesDir.createDirectoryAndParents();
    }

    SymlinkTreeHelper helper =
        new SymlinkTreeHelper(inputManifest, outputManifest, runfilesDir, tree.getWorkspaceName());

    switch (tree.getSymlinksMode()) {
      case CREATE -> {
        helper.createRunfilesSymlinks(tree.getMapping());
        // Copy rather than link the manifest so that the up-to-date check above can tell that the
        // symlinks have been created and match the manifest.
        helper.copyManifest();
      }
      case SKIP -> helper.createMinimalRunfilesDirectory();
    }
  }

  private static boolean isRunfilesDirectoryPopulated(Path runfilesDir, Path outputManifest) {
    String relativeRunfilePath;
    try (BufferedReader reader =
        new BufferedReader(new InputStreamReader(outputManifest.getInputStream(), ISO_8859_1))) {
      // If it is created at all, the manifest always contains at least one line.
      relativeRunfilePath = Splitter.on(' ').splitToList(reader.readLine()).get(0);
      // The runfile could be a dangling symlink.
      return runfilesDir.getRelative(relativeRunfilePath).exists(Symlinks.NOFOLLOW);
    } catch (IOException e) {
      // Instead of failing outright, just assume the runfiles directory is not populated.
      return false;
    }
  }
}
