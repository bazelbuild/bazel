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


import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.CommandUtils;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

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

  private Path getOutputManifest() {
    return symlinkTreeRoot.getChild("MANIFEST");
  }

  /** Creates a symlink tree by making VFS calls. */
  public void createSymlinksDirectly(Path symlinkTreeRoot, Map<PathFragment, Artifact> symlinks)
      throws IOException {
    Preconditions.checkState(!filesetTree);
    // Our strategy is to minimize mutating file system operations as much as possible. Ideally, if
    // there is an existing symlink tree with the expected contents, we don't make any changes. Our
    // algorithm goes as follows:
    //
    // 1. Create a tree structure that represents the entire set of paths that we want to exist. The
    //    tree structure contains a node for every intermediate directory. For example, this is the
    //    tree structure corresponding to the symlinks {"a/b/c": "foobar", "a/d/e": null}:
    //
    //      / b - c (symlink to "foobar")
    //    a
    //      \ d - e (empty file)
    //
    //    Note that we need to distinguish directories, symlinks, and empty files. In the Directory
    //    class below, we use two maps for that purpose: one for directories, and one for symlinks
    //    and empty files. This avoids having to create additional classes / objects to distinguish
    //    them.
    //
    // 2. Perform a depth-first traversal over the on-disk file system tree and make each directory
    //    match our expected directory layout. To that end, call readdir, and compare the result
    //    with the contents of the corresponding node in the in-memory tree.
    //
    //    For each Dirent entry in the readdir result:
    //    - If the entry is not in the current node, if the entry has an incompatible type, or if it
    //      is a symlink that points to the wrong location, delete the entry on disk (recursively).
    //    - Otherwise:
    //      - If the entry is a directory, recurse into that directory
    //      - In all cases, delete the entry in the current in-memory node.
    //
    // 3. For every remaining entry in the node, create the corresponding file, symlink, or
    //    directory on disk. If it is a directory, recurse into that directory.
    Directory root = new Directory();
    for (Map.Entry<PathFragment, Artifact> entry : symlinks.entrySet()) {
      // This creates intermediate directory nodes as a side effect.
      Directory parentDir = root.walk(entry.getKey().getParentDirectory());
      parentDir.addSymlink(entry.getKey().getBaseName(), entry.getValue());
    }
    root.syncTreeRecursively(symlinkTreeRoot);
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
      FileSystemUtils.copyFile(inputManifest, getOutputManifest());
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

  static ImmutableMap<PathFragment, PathFragment> processFilesetLinks(
      ImmutableList<FilesetOutputSymlink> links, PathFragment root, PathFragment execRoot) {
    Map<PathFragment, PathFragment> symlinks = new HashMap<>();
    for (FilesetOutputSymlink symlink : links) {
      symlinks.put(root.getRelative(symlink.getName()), symlink.reconstituteTargetPath(execRoot));
    }
    return ImmutableMap.copyOf(symlinks);
  }

  private static final class Directory {
    private final Map<String, Artifact> symlinks = new HashMap<>();
    private final Map<String, Directory> directories = new HashMap<>();

    void addSymlink(String basename, @Nullable Artifact artifact) {
      symlinks.put(basename, artifact);
    }

    Directory walk(PathFragment dir) {
      Directory result = this;
      for (String segment : dir.getSegments()) {
        Directory next = result.directories.get(segment);
        if (next == null) {
          next = new Directory();
          result.directories.put(segment, next);
        }
        result = next;
      }
      return result;
    }

    void syncTreeRecursively(Path at) throws IOException {
      // This is a reimplementation of the C++ code in build-runfiles.cc. This avoids having to ship
      // a separate native tool to create a few runfiles.
      // TODO(ulfjack): Measure performance.
      FileStatus stat = at.statNullable(Symlinks.FOLLOW);
      if (stat == null) {
        at.createDirectoryAndParents();
      } else if (!stat.isDirectory()) {
        at.deleteTree();
        at.createDirectoryAndParents();
      }
      // TODO(ulfjack): provide the mode bits from FileStatus and use that to construct the correct
      //  chmod call here. Note that we do not have any tests for this right now. Something like
      //  this:
      // if (!stat.isExecutable() || !stat.isReadable()) {
      //   at.chmod(stat.getMods() | 0700);
      // }
      for (Dirent dirent : at.readdir(Symlinks.FOLLOW)) {
        String basename = dirent.getName();
        Path next = at.getChild(basename);
        if (symlinks.containsKey(basename)) {
          Artifact value = symlinks.remove(basename);
          if (value == null) {
            if (dirent.getType() != Dirent.Type.FILE) {
              next.deleteTree();
              FileSystemUtils.createEmptyFile(next);
            }
            // For consistency with build-runfiles.cc, we don't truncate the file if one exists.
          } else {
            // TODO(ulfjack): On Windows, this call makes a copy rather than creating a symlink.
            FileSystemUtils.ensureSymbolicLink(next, value.getPath().asFragment());
          }
        } else if (directories.containsKey(basename)) {
          Directory nextDir = directories.remove(basename);
          if (dirent.getType() != Dirent.Type.DIRECTORY) {
            next.deleteTree();
          }
          nextDir.syncTreeRecursively(at.getChild(basename));
        } else {
          at.getChild(basename).deleteTree();
        }
      }

      for (Map.Entry<String, Artifact> entry : symlinks.entrySet()) {
        Path next = at.getChild(entry.getKey());
        if (entry.getValue() == null) {
          FileSystemUtils.createEmptyFile(next);
        } else {
          FileSystemUtils.ensureSymbolicLink(next, entry.getValue().getPath().asFragment());
        }
      }
      for (Map.Entry<String, Directory> entry : directories.entrySet()) {
        entry.getValue().syncTreeRecursively(at.getChild(entry.getKey()));
      }
    }
  }
}
