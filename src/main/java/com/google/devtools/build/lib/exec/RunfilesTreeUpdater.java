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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.concurrent.GuardedBy;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Utility used in local execution to create a runfiles tree if {@code --nobuild_runfile_links} has
 * been specified.
 *
 * <p>It is safe to call {@link #updateRunfilesDirectory} concurrently.
 */
@ThreadSafe
public class RunfilesTreeUpdater {

  public static final RunfilesTreeUpdater INSTANCE = new RunfilesTreeUpdater();

  private final Object lock = new Object();

  private static final class LockWithRefcnt {
    int refcnt = 1;
  }

  /**
   * Poor man's reference counted object pool.
   *
   * <p>Maintains a mapping of runfiles directory to a monitor. The monitor maintains a counter that
   * tracks how many threads it is acquired by at the moment. If the count drops to zero the mapping
   * is removed.
   */
  @GuardedBy("lock")
  private final Map<PathFragment, LockWithRefcnt> locksWithRefcnt = new HashMap<>();

  private RunfilesTreeUpdater() {}

  private static void updateRunfilesTree(
      Path execRoot,
      PathFragment runfilesDir,
      BinTools binTools,
      ImmutableMap<String, String> env,
      OutErr outErr,
      boolean enableRunfiles)
      throws IOException, ExecException {
    Path runfilesDirPath = execRoot.getRelative(runfilesDir);
    Path inputManifest = RunfilesSupport.inputManifestPath(runfilesDirPath);
    if (!inputManifest.exists()) {
      return;
    }

    Path outputManifest = RunfilesSupport.outputManifestPath(runfilesDirPath);
    try {
      // Avoid rebuilding the runfiles directory if the manifest in it matches the input manifest,
      // implying the symlinks exist and are already up to date. If the output manifest is a
      // symbolic link, it is likely a symbolic link to the input manifest, so we cannot trust it as
      // an up-to-date check.
      if (!outputManifest.isSymbolicLink()
          && Arrays.equals(outputManifest.getDigest(), inputManifest.getDigest())) {
        return;
      }
    } catch (IOException e) {
      // Ignore it - we will just try to create runfiles directory.
    }

    if (!runfilesDirPath.exists()) {
      runfilesDirPath.createDirectoryAndParents();
    }

    SymlinkTreeHelper helper =
        new SymlinkTreeHelper(inputManifest, runfilesDirPath, /* filesetTree= */ false);
    helper.createSymlinks(execRoot, outErr, binTools, env, enableRunfiles);
  }

  private LockWithRefcnt getLockAndIncrementRefcnt(PathFragment runfilesDirectory) {
    synchronized (lock) {
      LockWithRefcnt lock = locksWithRefcnt.get(runfilesDirectory);
      if (lock != null) {
        lock.refcnt++;
        return lock;
      }
      lock = new LockWithRefcnt();
      locksWithRefcnt.put(runfilesDirectory, lock);
      return lock;
    }
  }

  private void decrementRefcnt(PathFragment runfilesDirectory) {
    synchronized (lock) {
      LockWithRefcnt lock = locksWithRefcnt.get(runfilesDirectory);
      lock.refcnt--;
      if (lock.refcnt == 0) {
        if (!locksWithRefcnt.remove(runfilesDirectory, lock)) {
          throw new IllegalStateException(
              String.format(
                  "Failed to remove lock for dir '%s'." + " This is a bug.", runfilesDirectory));
        }
      }
    }
  }

  public void updateRunfilesDirectory(
      Path execRoot,
      RunfilesSupplier runfilesSupplier,
      BinTools binTools,
      ImmutableMap<String, String> env,
      OutErr outErr)
      throws ExecException, IOException {
    for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> runfiles :
        runfilesSupplier.getMappings().entrySet()) {
      PathFragment runfilesDir = runfiles.getKey();
      if (runfilesSupplier.isBuildRunfileLinks(runfilesDir)) {
        continue;
      }

      try {
        long startTime = Profiler.nanoTimeMaybe();
        // Synchronize runfiles tree generation on the runfiles directory in order to prevent
        // concurrent modifications of the runfiles tree. In particular this can happen for sharded
        // tests and --runs_per_test > 1 in which case multiple actions use the same runfiles tree.
        synchronized (getLockAndIncrementRefcnt(runfilesDir)) {
          Profiler.instance()
              .logSimpleTask(startTime, ProfilerTask.WAIT, "Waiting to create runfiles tree");
          updateRunfilesTree(
              execRoot,
              runfilesDir,
              binTools,
              env,
              outErr,
              runfilesSupplier.isRunfileLinksEnabled(runfilesDir));
        }
      } finally {
        decrementRefcnt(runfilesDir);
      }
    }
  }
}
