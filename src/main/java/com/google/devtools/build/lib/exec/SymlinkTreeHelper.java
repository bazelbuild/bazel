// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.CommandUtils;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Helper class responsible for the symlink tree creation. Used to generate runfiles and fileset
 * symlink farms.
 */
public final class SymlinkTreeHelper {
  @VisibleForTesting
  public static final String BUILD_RUNFILES = "build-runfiles" + OsUtils.executableExtension();

  private final Path inputManifest;
  private final Path symlinkTreeRoot;
  private final boolean filesetTree;

  /**
   * Creates SymlinkTreeHelper instance. Can be used independently of SymlinkTreeAction.
   *
   * @param inputManifest exec path to the input runfiles manifest
   * @param symlinkTreeRoot the root of the symlink tree to be created
   * @param filesetTree true if this is fileset symlink tree, false if this is a runfiles symlink
   *     tree.
   */
  public SymlinkTreeHelper(Path inputManifest, Path symlinkTreeRoot, boolean filesetTree) {
    this.inputManifest = inputManifest;
    this.symlinkTreeRoot = symlinkTreeRoot;
    this.filesetTree = filesetTree;
  }

  public Path getOutputManifest() {
    return symlinkTreeRoot;
  }

  /**
   * Creates symlink tree and output manifest using the {@code build-runfiles.cc} tool.
   *
   * @param enableRunfiles If {@code false} only the output manifest is created.
   */
  public void createSymlinks(
      Path execRoot,
      OutErr outErr,
      BinTools binTools,
      Map<String, String> shellEnvironment,
      boolean enableRunfiles)
      throws ExecException {
    if (enableRunfiles) {
      createSymlinksUsingCommand(execRoot, binTools, shellEnvironment, outErr);
    } else {
      copyManifest();
    }
  }

  /** Copies the input manifest to the output manifest. */
  public void copyManifest() throws ExecException {
    // Pretend we created the runfiles tree by copying the manifest
    try {
      symlinkTreeRoot.createDirectoryAndParents();
      FileSystemUtils.copyFile(inputManifest, symlinkTreeRoot.getChild("MANIFEST"));
    } catch (IOException e) {
      throw new EnvironmentalExecException(e);
    }
  }

  /**
   * Creates a symlink tree using a CommandBuilder. This means that the symlink tree will always be
   * present on the developer's workstation. Useful when running commands locally.
   *
   * <p>Warning: this method REALLY executes the command on the box Bazel is running on, without any
   * kind of synchronization, locking, or anything else.
   */
  public void createSymlinksUsingCommand(
      Path execRoot, BinTools binTools, Map<String, String> shellEnvironment, OutErr outErr)
      throws EnvironmentalExecException {
    Command command = createCommand(execRoot, binTools, shellEnvironment);
    try {
      if (outErr != null) {
        command.execute(outErr.getOutputStream(), outErr.getErrorStream());
      } else {
        command.execute();
      }
    } catch (CommandException e) {
      throw new EnvironmentalExecException(CommandUtils.describeCommandFailure(true, e), e);
    }
  }

  @VisibleForTesting
  Command createCommand(Path execRoot, BinTools binTools, Map<String, String> shellEnvironment) {
    Preconditions.checkNotNull(shellEnvironment);
    List<String> args = Lists.newArrayList();
    args.add(binTools.getEmbeddedPath(BUILD_RUNFILES).asFragment().getPathString());
    if (filesetTree) {
      args.add("--allow_relative");
      args.add("--use_metadata");
    }
    args.add(inputManifest.relativeTo(execRoot).getPathString());
    args.add(symlinkTreeRoot.relativeTo(execRoot).getPathString());
    return new CommandBuilder()
        .addArgs(args)
        .setWorkingDir(execRoot)
        .setEnv(shellEnvironment)
        .build();
  }

  static Map<PathFragment, PathFragment> readSymlinksFromFilesetManifest(Path manifest)
      throws IOException {
    Map<PathFragment, PathFragment> result = new HashMap<>();
    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(
                // ISO_8859 is used to write the manifest in {Runfiles,Fileset}ManifestAction.
                manifest.getInputStream(), ISO_8859_1))) {
      String line;
      int lineNumber = 0;
      while ((line = reader.readLine()) != null) {
        // If the input has metadata (for fileset), they appear in every other line.
        if (++lineNumber % 2 == 0) {
          continue;
        }
        int spaceIndex = line.indexOf(' ');
        result.put(
            PathFragment.create(line.substring(0, spaceIndex)),
            PathFragment.create(line.substring(spaceIndex + 1)));
      }
      if (lineNumber % 2 != 0) {
        throw new IOException(
            "Possibly corrupted manifest file '" + manifest.getPathString() + "'");
      }
    }
    return result;
  }
}
