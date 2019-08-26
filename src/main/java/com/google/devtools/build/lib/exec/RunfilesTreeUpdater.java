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
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Utility used in local execution to create a runfiles tree if {@code --nobuild_runfile_links}
 * has been specified.
 */
@ThreadSafe
public class RunfilesTreeUpdater {

  /**
   * Poor man's reference counted object pool.
   *
   * <p>Maintains a mapping of runfiles directory to monitor. The monitor is an AtomicInteger in
   * order to keep track how often the monitor has been acquired. If the count drops to 0 the
   * mapping will be removed.
   */
  private final ConcurrentMap<PathFragment, AtomicInteger> monitorsWithRefcnt
      = new ConcurrentHashMap<>();
  private final boolean enableRunfiles;


  public RunfilesTreeUpdater(boolean enableRunfiles) {
    this.enableRunfiles = enableRunfiles;
  }

  private static void updateRunfilesTree(Path execRoot, PathFragment runfilesDir,
      BinTools binTools, ImmutableMap<String, String> env, OutErr outErr, boolean enableRunfiles)
      throws IOException, ExecException {
    Path runfilesDirPath = execRoot.getRelative(runfilesDir);
    Path inputManifest = RunfilesSupport.inputManifestPath(runfilesDirPath);
    if (!inputManifest.exists()) {
      // TODO(buchgr): Add logic to create a runfiles tree if --nobuild_runfile_manifests is set
      throw new UserExecException("Can't build local runfiles tree because "
          + "--nobuild_runfile_manifests is set.");
    }

    Path outputManifest = RunfilesSupport.outputManifestPath(runfilesDirPath);
    try {
      // Avoid rebuilding the runfiles directory if the manifest in it matches the input manifest,
      // implying the symlinks exist and are already up to date. If the output manifest is a
      // symbolic link, it is likely a symbolic link to the input manifest, so we cannot trust it as
      // an up-to-date check.
      FileStatus stat = outputManifest.stat();
      if (stat.isFile() && !stat.isSymbolicLink() && Arrays.equals(outputManifest.getDigest(),
          inputManifest.getDigest())) {
        return;
      }
    } catch (IOException e) {
      // Ignore it - we will just try to create runfiles directory.
    }

    if (!runfilesDirPath.exists()) {
      runfilesDirPath.createDirectoryAndParents();
    }

    SymlinkTreeHelper helper = new SymlinkTreeHelper(inputManifest, runfilesDirPath,
        /* filesetTree= */ false);
    helper.createSymlinks(execRoot, outErr, binTools, env, enableRunfiles);
  }
  
  private AtomicInteger tryGetMonitor(PathFragment runfilesDirectory) {
    AtomicInteger newMonitorWithRefcnt = new AtomicInteger(1);
    AtomicInteger monitorWithRefcnt =
        monitorsWithRefcnt.putIfAbsent(runfilesDirectory, newMonitorWithRefcnt);
    if (monitorWithRefcnt == null) {
      // This call created the entry for the runfiles directory as there wasn't one before.
      return newMonitorWithRefcnt;
    }
    while (true) {
      int count = monitorWithRefcnt.get();
      if (count == 0) {
        // Getting the lock failed as the reference count was 0. A value of 0 means that we
        // raced with returnMonitor() and the monitor is about to be removed from the
        // monitorsWithRefcnt map.
        return null;
      }
      if (monitorWithRefcnt.compareAndSet(count, count + 1)) {
        return monitorWithRefcnt;
      }
    }
  }

  private AtomicInteger getMonitor(PathFragment runfilesDirectory) {
    while (true) {
      AtomicInteger monitor = tryGetMonitor(runfilesDirectory);
      if (monitor != null) {
        return monitor;
      }
    }
  }

  private void returnMonitor(AtomicInteger monitor, PathFragment runfilesDirectory) {
    synchronized (monitor) {
      int count = monitor.decrementAndGet();
      if (count == 0) {
        if (!monitorsWithRefcnt.remove(runfilesDirectory, monitor)) {
          throw new IllegalStateException(String.format("Failed to remove monitor for '%s'."
              + " This is a bug.", runfilesDirectory));
        }
      }
    }
  }

  public void updateRunfilesDirectory(Path execRoot, RunfilesSupplier runfilesSupplier,
      ArtifactPathResolver pathResolver, BinTools binTools, ImmutableMap<String, String> env,
      OutErr outErr) throws ExecException, IOException {
    for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> runfiles : runfilesSupplier.getMappings(pathResolver).entrySet()) {
      PathFragment runfilesDir = runfiles.getKey();
      // Synchronize runfiles tree generation on the runfiles directory in order to prevent
      // concurrent modifications of the runfiles tree. In particular this can happen for sharded
      // tests and --runs_per_test > 1 in which case multiple actions use the same runfiles tree.
      long startTime = Profiler.nanoTimeMaybe();
      AtomicInteger monitor = getMonitor(runfilesDir);
      synchronized (monitor) {
        try {
          Profiler.instance()
              .logSimpleTask(startTime, ProfilerTask.WAIT, "Waiting to create runfiles tree");
          updateRunfilesTree(execRoot, runfilesDir, binTools, env, outErr, enableRunfiles);
        } finally {
          returnMonitor(monitor, runfilesDir);
        }
      }
    }
  }
}
