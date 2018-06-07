// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.common.hash.Hashing;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * File system for actions.
 *
 * <p>This class is thread-safe except that
 *
 * <ul>
 *   <li>{@link updateContext} must be called exclusively of any other methods.
 *   <li>This class relies on synchronized access to {@link env}. If there are other threads, that
 *       access {@link env}, they must also used synchronized access.
 * </ul>
 */
final class ActionFileSystem extends FileSystem implements ActionInputFileCache, InjectionListener {
  private static final Logger LOGGER = Logger.getLogger(ActionFileSystem.class.getName());

  /** Actual underlying filesystem. */
  private final FileSystem delegate;

  private final PathFragment execRootFragment;
  private final Path execRootPath;
  private final ImmutableList<PathFragment> sourceRoots;

  private final ActionInputMap inputArtifactData;

  /** exec path → artifact and metadata */
  private final HashMap<PathFragment, OptionalInputMetadata> optionalInputs;

  /** digest → artifacts in {@link inputs} */
  private final ConcurrentHashMap<ByteString, Artifact> optionalInputsByDigest;

  /** exec path → artifact and metadata */
  private final ImmutableMap<PathFragment, OutputMetadata> outputs;

  /** Used to lookup metadata for optional inputs. */
  private SkyFunction.Environment env = null;

  /**
   * Called whenever there is new metadata for an output.
   *
   * <p>This is backed by injection into an {@link ActionMetadataHandler} instance so should only be
   * called once per artifact.
   */
  private MetadataConsumer metadataConsumer = null;

  ActionFileSystem(
      FileSystem delegate,
      Path execRoot,
      ImmutableList<Root> sourceRoots,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> allowedInputs,
      Iterable<Artifact> outputArtifacts) {
    try {
      Profiler.instance().startTask(ProfilerTask.ACTION_FS_STAGING, "staging");
      this.delegate = delegate;

      this.execRootFragment = execRoot.asFragment();
      this.execRootPath = getPath(execRootFragment);
      this.sourceRoots =
          sourceRoots
              .stream()
              .map(root -> root.asPath().asFragment())
              .collect(ImmutableList.toImmutableList());

      validateRoots();

      this.inputArtifactData = inputArtifactData;

      this.optionalInputs = new HashMap<>();
      for (Artifact input : allowedInputs) {
        // Skips staging source artifacts as a performance optimization. We may want to stage them
        // if we want stricter enforcement of source sandboxing.
        //
        // TODO(shahan): there are no currently known cases where metadata is requested for an
        // optional source input. If there are any, we may want to stage those.
        if (input.isSourceArtifact() || inputArtifactData.getMetadata(input) != null) {
          continue;
        }
        optionalInputs.computeIfAbsent(
            input.getExecPath(), unused -> new OptionalInputMetadata(input));
      }

      this.optionalInputsByDigest = new ConcurrentHashMap<>();

      this.outputs =
          Streams.stream(outputArtifacts)
              .collect(
                  ImmutableMap.toImmutableMap(a -> a.getExecPath(), a -> new OutputMetadata(a)));
    } finally {
      Profiler.instance().completeTask(ProfilerTask.ACTION_FS_STAGING);
    }
  }

  /**
   * Must be called prior to access and updated as needed.
   *
   * <p>These cannot be passed into the constructor because while {@link ActionFileSystem} is
   * action-scoped, the environment and metadata consumer change multiple times, at well defined
   * points, during the lifetime of an action.
   */
  public void updateContext(SkyFunction.Environment env, MetadataConsumer metadataConsumer) {
    this.env = env;
    this.metadataConsumer = metadataConsumer;
  }

  // -------------------- ActionInputFileCache implementation --------------------

  @Override
  @Nullable
  public FileArtifactValue getMetadata(ActionInput actionInput) throws IOException {
    return getMetadataChecked(actionInput.getExecPath());
  }

  @Override
  @Nullable
  public ActionInput getInput(String execPath) {
    return inputArtifactData.getInput(execPath);
  }

  // -------------------- InjectionListener Implementation --------------------

  @Override
  public void onInsert(ActionInput dest, byte[] digest, long size, int backendIndex)
      throws IOException {
    OutputMetadata output = outputs.get(dest.getExecPath());
    if (output != null) {
      output.set(new RemoteFileArtifactValue(digest, size, backendIndex),
          /*notifyConsumer=*/ false);
    }
  }

  // -------------------- FileSystem implementation --------------------

  @Override
  public boolean supportsModifications(Path path) {
    return isOutput(path);
  }

  @Override
  public boolean supportsSymbolicLinksNatively(Path path) {
    return isOutput(path);
  }

  @Override
  protected boolean supportsHardLinksNatively(Path path) {
    return isOutput(path);
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return true;
  }

  /** ActionFileSystem currently doesn't track directories. */
  @Override
  public boolean createDirectory(Path path) throws IOException {
    return true;
  }

  @Override
  public void createDirectoryAndParents(Path path) throws IOException {}

  @Override
  protected long getFileSize(Path path, boolean followSymlinks) throws IOException {
    return getMetadataOrThrowFileNotFound(path).getSize();
  }

  @Override
  public boolean delete(Path path) throws IOException {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    return getMetadataOrThrowFileNotFound(path).getModifiedTime();
  }

  @Override
  public void setLastModifiedTime(Path path, long newTime) throws IOException {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected byte[] getFastDigest(Path path, HashFunction hash) throws IOException {
    if (hash != HashFunction.MD5) {
      return null;
    }
    return getMetadataOrThrowFileNotFound(path).getDigest();
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
    FileArtifactValue metadata = getMetadataUnchecked(path);
    return metadata == null ? false : metadata.getType() == FileStateType.DIRECTORY;
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
    FileArtifactValue metadata = getMetadataUnchecked(path);
    return metadata == null ? false : metadata.getType() == FileStateType.REGULAR_FILE;
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
    FileArtifactValue metadata = getMetadataUnchecked(path);
    return metadata == null ? false : metadata.getType() == FileStateType.SPECIAL_FILE;
  }

  private static String createSymbolicLinkErrorMessage(
      Path linkPath, PathFragment targetFragment, String message) {
    return "createSymbolicLink(" + linkPath + ", " + targetFragment + "): " + message;
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    PathFragment targetExecPath = asExecPath(targetFragment);
    FileArtifactValue inputMetadata = inputArtifactData.getMetadata(targetExecPath.getPathString());
    if (inputMetadata == null) {
      OptionalInputMetadata metadataHolder = optionalInputs.get(targetExecPath);
      if (metadataHolder != null) {
        inputMetadata = metadataHolder.get();
      }
    }
    if (inputMetadata == null) {
      throw new FileNotFoundException(
          createSymbolicLinkErrorMessage(
              linkPath, targetFragment, targetFragment + " is not an input."));
    }
    OutputMetadata outputHolder = outputs.get(asExecPath(linkPath));
    if (outputHolder == null) {
      throw new FileNotFoundException(
          createSymbolicLinkErrorMessage(
              linkPath, targetFragment, linkPath + " is not an output."));
    }
    outputHolder.set(inputMetadata, /*notifyConsumer=*/ true);
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
    return getMetadataUnchecked(path) != null;
  }

  @Override
  protected Collection<String> getDirectoryEntries(Path path) throws IOException {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected boolean isReadable(Path path) throws IOException {
    return exists(path, true);
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {}

  @Override
  protected boolean isWritable(Path path) throws IOException {
    return isOutput(path);
  }

  @Override
  public void setWritable(Path path, boolean writable) throws IOException {}

  @Override
  protected boolean isExecutable(Path path) throws IOException {
    return true;
  }

  @Override
  protected void setExecutable(Path path, boolean executable) throws IOException {}

  @Override
  protected InputStream getInputStream(Path path) throws IOException {
    FileArtifactValue metadata = getMetadataChecked(asExecPath(path));
    if (metadata instanceof FileArtifactValue.InlineFileArtifactValue) {
      return ((FileArtifactValue.InlineFileArtifactValue) metadata).getInputStream();
    }
    Preconditions.checkArgument(
        !(metadata instanceof FileArtifactValue.RemoteFileArtifactValue),
        "getInputStream called for remote file: %s",
        path);
    return delegate.getPath(path.asFragment()).getInputStream();
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append) throws IOException {
    Preconditions.checkArgument(!append, "ActionFileSystem doesn't support append.");
    return Preconditions.checkNotNull(
            outputs.get(asExecPath(path)), "getOutputStream called for non-output: %s", path)
        .getOutputStream();
  }

  @Override
  public void renameTo(Path sourcePath, Path targetPath) throws IOException {
    throw new UnsupportedOperationException("renameTo(" + sourcePath + ", " + targetPath + ")");
  }

  @Override
  protected void createFSDependentHardLink(Path linkPath, Path originalPath) throws IOException {
    throw new UnsupportedOperationException(
        "createFSDependendHardLink(" + linkPath + ", " + originalPath + ")");
  }

  // -------------------- Implementation Helpers --------------------

  private PathFragment asExecPath(Path path) {
    return asExecPath(path.asFragment());
  }

  private PathFragment asExecPath(PathFragment fragment) {
    if (fragment.startsWith(execRootFragment)) {
      return fragment.relativeTo(execRootFragment);
    }
    for (PathFragment root : sourceRoots) {
      if (fragment.startsWith(root)) {
        return fragment.relativeTo(root);
      }
    }
    throw new IllegalArgumentException(
        fragment + " was not found under any known root: " + execRootFragment + ", " + sourceRoots);
  }

  @Nullable
  private FileArtifactValue getMetadataChecked(PathFragment execPath) throws IOException {
    {
      FileArtifactValue metadata = inputArtifactData.getMetadata(execPath.getPathString());
      if (metadata != null) {
        return metadata;
      }
    }
    {
      OptionalInputMetadata metadataHolder = optionalInputs.get(execPath);
      if (metadataHolder != null) {
        return metadataHolder.get();
      }
    }
    {
      OutputMetadata metadataHolder = outputs.get(execPath);
      if (metadataHolder != null) {
        FileArtifactValue metadata = metadataHolder.get();
        if (metadata != null) {
          return metadata;
        }
      }
    }
    return null;
  }

  private FileArtifactValue getMetadataOrThrowFileNotFound(Path path) throws IOException {
    FileArtifactValue metadata = getMetadataChecked(asExecPath(path));
    if (metadata == null) {
      throw new FileNotFoundException(path.getPathString() + " was not found");
    }
    return metadata;
  }

  @Nullable
  private FileArtifactValue getMetadataUnchecked(Path path) {
    try {
      return getMetadataChecked(asExecPath(path));
    } catch (IOException e) {
      throw new IllegalStateException(
          "Error getting metadata for " + path.getPathString() + ": " + e.getMessage(), e);
    }
  }

  private boolean isOutput(Path path) {
    PathFragment fragment = path.asFragment();
    if (!fragment.startsWith(execRootFragment)) {
      return false;
    }
    return outputs.containsKey(fragment.relativeTo(execRootFragment));
  }

  /**
   * Verifies that no root is the prefix of any other root.
   *
   * <p>TODO(shahan): if this is insufficiently general, we can topologically order on the prefix
   * relation between roots.
   */
  private void validateRoots() {
    for (PathFragment root1 : sourceRoots) {
      Preconditions.checkState(
          !root1.startsWith(execRootFragment), "%s starts with %s", root1, execRootFragment);
      Preconditions.checkState(
          !execRootFragment.startsWith(root1), "%s starts with %s", execRootFragment, root1);
      for (PathFragment root2 : sourceRoots) {
        if (root1 == root2) {
          continue;
        }
        Preconditions.checkState(!root1.startsWith(root2), "%s starts with %s", root1, root2);
      }
    }
  }

  private static ByteString toByteString(byte[] digest) {
    return ByteString.copyFrom(BaseEncoding.base16().lowerCase().encode(digest).getBytes(US_ASCII));
  }

  @FunctionalInterface
  public interface MetadataConsumer {
    void accept(Artifact artifact, FileArtifactValue value) throws IOException;
  }

  private class OptionalInputMetadata {
    private final Artifact artifact;
    private volatile FileArtifactValue metadata = null;

    private OptionalInputMetadata(Artifact artifact) {
      this.artifact = artifact;
    }

    public FileArtifactValue get() throws IOException {
      if (metadata == null) {
        synchronized (this) {
          if (metadata == null) {
            try {
              // TODO(shahan): {@link SkyFunction.Environment} requires single-threaded access so
              // we enforce that here by making these (multithreaded) calls synchronized. It might
              // be better to make the underlying methods synchronized to avoid having another
              // caller unintentionally calling into the environment without locking.
              //
              // This is currently known to be reached from the distributor during remote include
              // scanning which we expect to propagate exceptions up for skyframe restarts.
              synchronized (env) {
                metadata = (FileArtifactValue) env.getValue(ArtifactSkyKey.key(artifact, false));
              }
            } catch (InterruptedException e) {
              throw new InterruptedIOException(e.getMessage());
            }
            if (metadata == null) {
              throw new ActionExecutionFunction.MissingDepException();
            }
            if (metadata.getType().exists() && metadata.getDigest() != null) {
              optionalInputsByDigest.put(toByteString(metadata.getDigest()), artifact);
            }
          }
        }
      }
      return metadata;
    }
  }

  private class OutputMetadata {
    private final Artifact artifact;
    @Nullable private volatile FileArtifactValue metadata = null;

    private OutputMetadata(Artifact artifact) {
      this.artifact = artifact;
    }

    @Nullable
    public FileArtifactValue get() {
      return metadata;
    }

    /**
     * Sets the output metadata, and maybe notify the metadataConsumer.
     *
     * @param metadata the metadata to write
     * @param notifyConsumer whether to notify metadataConsumer. Callers should not notify the
     * metadataConsumer if it will be notified separately at the Spawn level.
     */
    public void set(FileArtifactValue metadata, boolean notifyConsumer) throws IOException {
      if (notifyConsumer) {
        metadataConsumer.accept(artifact, metadata);
      }
      this.metadata = metadata;
    }

    /** Callers are expected to close the returned stream. */
    public ByteArrayOutputStream getOutputStream() {
      Preconditions.checkState(metadata == null, "getOutputStream called twice for: %s", artifact);
      return new ByteArrayOutputStream() {
        @Override
        public void close() throws IOException {
          super.close();
          byte[] data = toByteArray();
          set(
              new FileArtifactValue.InlineFileArtifactValue(
                  data, Hashing.md5().hashBytes(data).asBytes()), /*notifyConsumer=*/ true);
        }
      };
    }
  }
}
