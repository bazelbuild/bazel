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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestUtils;
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
 * Utility used in local execution to create a runfiles tree if {@code --nobuild_runfile_links} has
 * been specified.
 *
 * <p>It is safe to call {@link #updateRunfiles} concurrently.
 */
@ThreadSafe
public class RunfilesTreeUpdater {
  private final Path execRoot;
  private final XattrProvider xattrProvider;
  private final boolean detectStaleRunfiles;

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

  public static RunfilesTreeUpdater forCommandEnvironment(CommandEnvironment env) {
    ExecutionOptions options = env.getOptions().getOptions(ExecutionOptions.class);
    return new RunfilesTreeUpdater(
        env.getExecRoot(),
        env.getXattrProvider(),
        options != null && options.getDetectStaleRunfiles());
  }

  public RunfilesTreeUpdater(
      Path execRoot, XattrProvider xattrProvider, boolean detectStaleRunfiles) {
    this.execRoot = execRoot;
    this.xattrProvider = xattrProvider;
    this.detectStaleRunfiles = detectStaleRunfiles;
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
    try {
      // Avoid rebuilding the runfiles directory if the manifest in it matches the input manifest,
      // implying the symlinks exist and are already up to date. If the output manifest is a
      // symbolic link, it is likely a symbolic link to the input manifest, so we cannot trust it as
      // an up-to-date check.
      if (tree.getSymlinksMode() == RunfileSymlinksMode.CREATE
          && !outputManifest.isSymbolicLink()
          && Arrays.equals(
              DigestUtils.getDigestWithManualFallback(outputManifest, xattrProvider),
              DigestUtils.getDigestWithManualFallback(inputManifest, xattrProvider))
          && runfilesDirIsFresh(runfilesDir, outputManifest)) {
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
        helper.linkManifest();
      }
      case SKIP -> helper.createMinimalRunfilesDirectory();
    }
  }

  private boolean runfilesDirIsFresh(Path runfilesDir, Path outputManifest) {
    if (detectStaleRunfiles) {
      // Manifest equality only proves freshness when entries are real symlinks, not copies.
      return runfilesUseRealSymlinks(runfilesDir, outputManifest);
    }
    // Legacy: only checks that the first runfile exists (doesn't catch copy-mode staleness).
    return OS.getCurrent() != OS.WINDOWS
        || isRunfilesDirectoryPopulated(runfilesDir, outputManifest);
  }

  private static boolean isRunfilesDirectoryPopulated(Path runfilesDir, Path outputManifest) {
    String relativeRunfilePath;
    try (BufferedReader reader =
        new BufferedReader(new InputStreamReader(outputManifest.getInputStream(), ISO_8859_1))) {
      // If it is created at all, the manifest always contains at least one line.
      relativeRunfilePath = reader.readLine().split(" ", -1)[0];
    } catch (IOException e) {
      // Instead of failing outright, just assume the runfiles directory is not populated.
      return false;
    }
    // The runfile could be a dangling symlink.
    return runfilesDir.getRelative(relativeRunfilePath).exists(Symlinks.NOFOLLOW);
  }

  /**
   * Returns true if the runfiles directory is populated and its entries are real symlinks.
   *
   * <p>When the entries are real symlinks (Linux, or Windows with symlink privileges), the
   * manifest-equality short-circuit in {@link #updateRunfilesTree} is sound: the symlinks
   * auto-follow current source content. When they are content copies (Windows without symlink
   * privileges, where {@code WindowsFileSystem.createSymbolicLink} silently falls back to {@code
   * Files.copy}), the short-circuit is unsafe and the caller must re-sync.
   *
   * <p>Returns false when the directory is empty/missing (e.g. the previous run was SKIP) or when
   * the first non-empty entry isn't a symlink, so the caller falls through to the create path.
   *
   * <p>Empty-file manifest entries (lines with only a path and no target) are skipped: they are
   * always materialized as regular files even when the rest of the tree uses symlinks, so they
   * can't be used to distinguish symlink-mode from copy-mode.
   */
  @VisibleForTesting
  static boolean runfilesUseRealSymlinks(Path runfilesDir, Path outputManifest) {
    String firstEntryPath = null;
    try (BufferedReader reader =
        new BufferedReader(new InputStreamReader(outputManifest.getInputStream(), ISO_8859_1))) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split(" ", -1);
        if (parts.length >= 2 && !parts[1].isEmpty()) {
          firstEntryPath = parts[0];
          break;
        }
      }
    } catch (IOException e) {
      return false;
    }
    if (firstEntryPath == null) {
      return false;
    }
    return runfilesDir.getRelative(firstEntryPath).isSymbolicLink();
  }
}
