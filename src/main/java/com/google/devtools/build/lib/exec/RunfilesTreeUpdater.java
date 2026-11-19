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
import com.google.common.hash.HashFunction;
import com.google.common.hash.HashingOutputStream;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.SourceManifestAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.util.DeterministicWriter;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.XattrProvider;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.SortedMap;
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
    // Without an action that writes the input manifest, its path next to the runfiles directory is
    // not reserved, so the manifest is generated here and only written into the runfiles directory.
    SortedMap<PathFragment, Artifact> generatedMapping =
        tree.isBuildRunfileManifests() ? null : tree.getMapping();
    DeterministicWriter generatedManifest =
        generatedMapping == null
            ? null
            : out -> SourceManifestAction.writeSourceSymlinksManifest(out, generatedMapping);
    FileStatus inputManifestStat = null;
    if (generatedManifest == null) {
      inputManifestStat = inputManifest.statIfFound();
      if (inputManifestStat == null) {
        return;
      }
    }
    Path outputManifest =
        execRoot.getRelative(RunfilesSupport.outputManifestExecPath(tree.getExecPath()));
    try {
      // Avoid rebuilding the runfiles directory if the manifest in it matches the input manifest,
      // implying the symlinks exist and are already up to date. If the output manifest is a
      // symbolic link, it is likely a symbolic link to the input manifest, so we cannot trust it as
      // an up-to-date check.
      // A previous run in SKIP mode can leave a regular file with the same content behind: a
      // generated manifest is always written as one, and on Windows symlinks may be silently
      // replaced by copies. We must not treat that as up to date, but we also don't want to
      // unnecessarily rebuild the runfiles directory all the time. Instead, check for the presence
      // of the first runfile. If it is present, we can be certain that the previous mode wasn't
      // SKIP.
      if (tree.getSymlinksMode() == RunfileSymlinksMode.CREATE) {
        // Not following symlinks means that the stat describes the output manifest itself, which is
        // only the file we digest below if it isn't a symbolic link - which is checked first.
        var outputManifestStat = outputManifest.statIfFound(Symlinks.NOFOLLOW);
        if (outputManifestStat != null
            && !outputManifestStat.isSymbolicLink()
            && Arrays.equals(
                DigestUtils.getDigestWithManualFallback(
                    outputManifest, xattrProvider, outputManifestStat),
                generatedManifest != null
                    ? digest(
                        generatedManifest,
                        outputManifest.getFileSystem().getDigestFunction().getHashFunction())
                    : DigestUtils.getDigestWithManualFallback(
                        inputManifest, xattrProvider, inputManifestStat))
            && isRunfilesDirectoryPopulated(runfilesDir, outputManifest, tree)) {
          return;
        }
      }
    } catch (IOException e) {
      // Ignore it - we will just try to create runfiles directory.
    }

    if (!runfilesDir.exists()) {
      runfilesDir.createDirectoryAndParents();
    }

    SymlinkTreeHelper helper = new SymlinkTreeHelper(inputManifest, outputManifest, runfilesDir);

    switch (tree.getSymlinksMode()) {
      case CREATE -> {
        // An interrupted sync must not leave behind a manifest that vouches for a half-updated tree.
        helper.deleteOutputManifest();
        helper.createRunfilesSymlinks(
            generatedMapping != null ? generatedMapping : tree.getMapping());
        if (generatedManifest != null) {
          helper.writeManifest(generatedManifest);
        } else {
          helper.linkManifest();
        }
      }
      case SKIP -> {
        if (generatedManifest != null) {
          helper.createMinimalRunfilesDirectory(generatedManifest);
        } else {
          helper.createMinimalRunfilesDirectory();
        }
      }
    }
  }

  private static byte[] digest(DeterministicWriter manifest, HashFunction hashFunction)
      throws IOException {
    try (HashingOutputStream out =
        new HashingOutputStream(hashFunction, OutputStream.nullOutputStream())) {
      manifest.writeTo(out);
      return out.hash().asBytes();
    }
  }

  private static boolean isRunfilesDirectoryPopulated(
      Path runfilesDir, Path outputManifest, RunfilesTree tree) {
    try (BufferedReader reader =
        new BufferedReader(new InputStreamReader(outputManifest.getInputStream(), ISO_8859_1))) {
      String firstLine = reader.readLine();
      if (firstLine == null) {
        // An empty tree looks the same in both modes.
        return true;
      }
      // An escaped entry starts with a space. The manifest matched, so the mapping has the same
      // first entry, unescaped.
      PathFragment firstRunfile =
          firstLine.startsWith(" ")
              ? tree.getMapping().firstKey()
              : PathFragment.create(Splitter.on(' ').splitToList(firstLine).get(0));
      // The runfile could be a dangling symlink.
      return runfilesDir.getRelative(firstRunfile).exists(Symlinks.NOFOLLOW);
    } catch (IOException e) {
      // Instead of failing outright, just assume the runfiles directory is not populated.
      return false;
    }
  }
}
