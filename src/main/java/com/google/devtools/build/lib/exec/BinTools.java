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
import com.google.common.hash.Hasher;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.skyframe.FileArtifactValue;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Initializes the &lt;execRoot>/_bin/ directory that contains auxiliary tools used during action
 * execution (alarm, etc). The main purpose of this is to make sure that those tools are accessible
 * using relative paths from the execution root.
 */
public final class BinTools {
  private final Path embeddedBinariesRoot;
  private final Path execrootParent;
  private final ImmutableList<String> embeddedTools;
  private ImmutableMap<String, ActionInput> actionInputs;

  private Path binDir;  // the working bin directory under execRoot

  private BinTools(BlazeDirectories directories, ImmutableList<String> tools) {
    this(
        directories.getEmbeddedBinariesRoot(),
        directories.getExecRoot().getParentDirectory(),
        tools);
  }

  private BinTools(Path embeddedBinariesRoot, Path execrootParent, ImmutableList<String> tools) {
    this.embeddedBinariesRoot = embeddedBinariesRoot;
    this.execrootParent = execrootParent;
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    // Files under embedded_tools shouldn't be copied to under _bin dir
    // They won't be used during action execution time.
    for (String tool : tools) {
      if (!tool.startsWith("embedded_tools/")) {
        builder.add(tool);
      }
    }
    this.embeddedTools = builder.build();
    this.binDir = null;
  }

  private ImmutableMap<String, ActionInput> populateActionInputMap() {
    ImmutableMap.Builder<String, ActionInput> result = ImmutableMap.builder();
    for (String embeddedPath : embeddedTools) {
      PathFragment execPath = getExecPath(embeddedPath);
      Path path = binDir.getRelative(execPath.getBaseName());
      result.put(embeddedPath, new PathActionInput(path, execPath));
    }
    return result.build();
  }

  /**
   * Creates an instance with the list of embedded tools obtained from scanning the directory
   * into which said binaries were extracted by the launcher.
   */
  public static BinTools forProduction(BlazeDirectories directories) throws IOException {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    scanDirectoryRecursively(builder, directories.getEmbeddedBinariesRoot(), "");
    return new BinTools(directories, builder.build());
  }

  /**
   * Creates an empty instance for testing.
   */
  @VisibleForTesting
  public static BinTools empty(BlazeDirectories directories) {
    return new BinTools(directories, ImmutableList.<String>of())
        .setBinDir(directories.getWorkspace().getBaseName());
  }

  /**
   * Creates an instance for testing without actually symlinking the tools.
   *
   * <p>Used for tests that need a set of embedded tools to be present, but not the actual files.
   */
  @VisibleForTesting
  public static BinTools forUnitTesting(BlazeDirectories directories, Iterable<String> tools) {
    return new BinTools(directories, ImmutableList.copyOf(tools))
        .setBinDir(directories.getWorkspace().getBaseName());
  }

  /**
   * Creates an instance for testing without actually symlinking the tools.
   *
   * <p>Used for tests that need a set of embedded tools to be present, but not the actual files.
   */
  @VisibleForTesting
  public static BinTools forUnitTesting(Path execroot, Iterable<String> tools) {
    return new BinTools(
        execroot.getRelative("/fake/embedded/tools"),
        execroot.getParentDirectory(),
        ImmutableList.copyOf(tools)).setBinDir(execroot.getBaseName());
  }

  /**
   * Returns a BinTools instance. Before calling this method, you have to populate the
   * {@link BlazeDirectories#getEmbeddedBinariesRoot} directory.
   */
  @VisibleForTesting
  public static BinTools forIntegrationTesting(
      BlazeDirectories directories, Iterable<String> tools, String repositoryName) {
    return new BinTools(directories, ImmutableList.copyOf(tools)).setBinDir(repositoryName);
  }

  private static void scanDirectoryRecursively(
      ImmutableList.Builder<String> result, Path root, String relative) throws IOException {
    for (Dirent dirent : root.readdir(Symlinks.NOFOLLOW)) {
      String childRelative = relative.isEmpty()
          ? dirent.getName()
          : relative + "/" + dirent.getName();
      switch (dirent.getType()) {
        case FILE:
          result.add(childRelative);
          break;

        case DIRECTORY:
          scanDirectoryRecursively(result, root.getChild(dirent.getName()), childRelative);
          break;

        default:
          // Nothing to do here -- we ignore symlinks, since they should not be present in the
          // embedded binaries tree.
          break;
      }
    }
  }

  /**
   * Returns an action input for the given embedded tool.
   */
  public ActionInput getActionInput(String embeddedPath) {
    if (actionInputs == null) {
      actionInputs = populateActionInputMap();
    }
    return actionInputs.get(embeddedPath);
  }

  public PathFragment getExecPath(String embedPath) {
    if (!embeddedTools.contains(embedPath)) {
      return null;
    }
    return PathFragment.create("_bin").getRelative(PathFragment.create(embedPath).getBaseName());
  }

  private BinTools setBinDir(String workspaceName) {
    binDir = execrootParent.getRelative(workspaceName).getRelative("_bin");
    return this;
  }

  /**
   * Initializes the build tools not available at absolute paths. Note that
   * these must be constant across all configurations.
   */
  public void setupBuildTools(String workspaceName) throws ExecException {
    setBinDir(workspaceName);
    try {
      binDir.createDirectoryAndParents();
    } catch (IOException e) {
      throw new EnvironmentalExecException("could not create directory '" + binDir  + "'", e);
    }

    for (String embeddedPath : embeddedTools) {
      setupTool(embeddedPath);
    }
  }

  private void setupTool(String embeddedPath) throws ExecException {
    Preconditions.checkNotNull(binDir);
    Path sourcePath = embeddedBinariesRoot.getRelative(embeddedPath);
    Path linkPath = binDir.getRelative(PathFragment.create(embeddedPath).getBaseName());
    linkTool(sourcePath, linkPath);
  }

  private void linkTool(Path sourcePath, Path linkPath) throws ExecException {
    if (linkPath.getFileSystem().supportsSymbolicLinksNatively(linkPath)) {
      try {
        if (!linkPath.isSymbolicLink()) {
          // ensureSymbolicLink() does not handle the case where there is already
          // a file with the same name, so we need to handle it here.
          linkPath.delete();
        }
        FileSystemUtils.ensureSymbolicLink(linkPath, sourcePath);
      } catch (IOException e) {
        throw new EnvironmentalExecException("failed to link '" + sourcePath + "'", e);
      }
    } else {
      // For file systems that do not support linking, copy.
      try {
        FileSystemUtils.copyTool(sourcePath, linkPath);
      } catch (IOException e) {
        throw new EnvironmentalExecException("failed to copy '" + sourcePath + "'" , e);
      }
    }
  }

  /** An ActionInput pointing at an absolute path. */
  public static final class PathActionInput implements VirtualActionInput {
    private final Path path;
    private final PathFragment execPath;
    private Metadata metadata;

    public PathActionInput(Path path, PathFragment execPath) {
      this.path = path;
      this.execPath = execPath;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      try (InputStream in = path.getInputStream()) {
        ByteStreams.copy(in, out);
      }
    }

    @Override
    public ByteString getBytes() throws IOException {
      ByteString.Output out = ByteString.newOutput();
      writeTo(out);
      return out.toByteString();
    }

    @Override
    public synchronized Metadata getMetadata() throws IOException {
      // We intentionally delay hashing until it is necessary.
      if (metadata == null) {
        metadata = hash(path);
      }
      return metadata;
    }

    private static Metadata hash(Path path) throws IOException {
      HashFunction hashFn = path.getFileSystem().getDigestFunction();
      Hasher hasher = hashFn.getHash().newHasher();
      int bytesCopied = 0;
      try (InputStream in = path.getInputStream()) {
        byte[] buffer = new byte[1024];
        int len;
        while ((len = in.read(buffer)) > 0) {
          hasher.putBytes(buffer, 0, len);
          bytesCopied += len;
        }
      }
      return FileArtifactValue.createForVirtualActionInput(
          hasher.hash().asBytes(),
          bytesCopied);
    }

    @Override
    public String getExecPathString() {
      return execPath.getPathString();
    }

    @Override
    public PathFragment getExecPath() {
      return execPath;
    }
  }
}
