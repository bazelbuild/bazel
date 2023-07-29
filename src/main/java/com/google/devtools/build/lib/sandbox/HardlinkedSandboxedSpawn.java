// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Collection;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Creates an execRoot for a Spawn that contains input files as hardlinks to their original
 * destination.
 */
public class HardlinkedSandboxedSpawn extends AbstractContainerizingSandboxedSpawn {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private boolean sandboxDebug = false;

  public HardlinkedSandboxedSpawn(
      Path sandboxPath,
      Path sandboxExecRoot,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      SandboxInputs inputs,
      SandboxOutputs outputs,
      Set<Path> writableDirs,
      TreeDeleter treeDeleter,
      @Nullable Path sandboxDebugPath,
      @Nullable Path statisticsPath,
      boolean sandboxDebug,
      String mnemonic) {
    super(
        sandboxPath,
        sandboxExecRoot,
        arguments,
        environment,
        inputs,
        outputs,
        writableDirs,
        treeDeleter,
        sandboxDebugPath,
        statisticsPath,
        mnemonic);
    this.sandboxDebug = sandboxDebug;
  }

  @Override
  protected void copyFile(Path source, Path target) throws IOException {
    hardLinkRecursive(source, target);
  }

  /**
   * Recursively creates hardlinks for all files in {@code source} path, in {@code target} path.
   * Symlinks are resolved. If files is located on another disk, hardlink will fail and a copy will
   * be made instead. Throws IllegalArgumentException if source path is a subdirectory of target
   * path.
   */
  private void hardLinkRecursive(Path source, Path target) throws IOException {
    if (source.isSymbolicLink()) {
      source = source.resolveSymbolicLinks();
    }

    if (source.isFile(Symlinks.NOFOLLOW)) {
      try {
        source.createHardLink(target);
      } catch (IOException e) {
        if (sandboxDebug) {
          logger.atInfo().log(
              "File %s could not be hardlinked, file will be copied instead.", source);
        }
        FileSystemUtils.copyFile(source, target);
      }
    } else if (source.isDirectory()) {
      if (source.startsWith(target)) {
        throw new IllegalArgumentException(source + " is a subdirectory of " + target);
      }
      target.createDirectory();
      Collection<Path> entries = source.getDirectoryEntries();
      for (Path entry : entries) {
        Path toPath = target.getChild(entry.getBaseName());
        hardLinkRecursive(entry, toPath);
      }
    }
  }
}
