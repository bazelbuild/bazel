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
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.common.hash.Hashing;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.InlineFileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.SourceFileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.AbstractFileSystemWithCustomStat;
import com.google.devtools.build.lib.vfs.FileStatus;
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
final class ActionFileSystem extends AbstractFileSystemWithCustomStat
    implements MetadataProvider, InjectionListener {
  private static final BaseEncoding LOWER_CASE_HEX = BaseEncoding.base16().lowerCase();

  /** Actual underlying filesystem. */
  private final FileSystem delegate;

  private final PathFragment execRootFragment;
  private final PathFragment outputPathFragment;
  private final ImmutableList<PathFragment> sourceRoots;

  private final ActionInputMap inputArtifactData;

  /** exec path → artifact and metadata */
  private final HashMap<PathFragment, OptionalInputMetadata> optionalInputs;

  /** exec path → artifact and metadata */
  private final LoadingCache<PathFragment, OutputMetadata> outputs;

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
      PathFragment execRoot,
      String relativeOutputPath,
      ImmutableList<Root> sourceRoots,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> allowedInputs,
      Iterable<Artifact> outputArtifacts) {
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.ACTION_FS_STAGING, "staging")) {
      this.delegate = delegate;

      this.execRootFragment = execRoot;
      this.outputPathFragment = execRootFragment.getRelative(relativeOutputPath);
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

      ImmutableMap<PathFragment, Artifact> outputsMapping = Streams.stream(outputArtifacts)
          .collect(ImmutableMap.toImmutableMap(Artifact::getExecPath, a -> a));
      this.outputs = CacheBuilder.newBuilder().build(
          CacheLoader.from(path -> new OutputMetadata(outputsMapping.get(path))));
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

  // -------------------- MetadataProvider implementation --------------------

  @Override
  @Nullable
  public FileArtifactValue getMetadata(ActionInput actionInput) throws IOException {
    return getMetadataChecked(actionInput.getExecPath());
  }

  @Override
  @Nullable
  public ActionInput getInput(String execPath) {
    ActionInput input = inputArtifactData.getInput(execPath);
    if (input != null) {
      return input;
    }
    OptionalInputMetadata metadata =
        optionalInputs.get(PathFragment.createAlreadyNormalized(execPath));
    return metadata == null ? null : metadata.getArtifact();
  }

  // -------------------- InjectionListener Implementation --------------------

  @Override
  public void onInsert(ActionInput dest, byte[] digest, long size, int backendIndex)
      throws IOException {
    outputs.getUnchecked(dest.getExecPath()).set(
        new RemoteFileArtifactValue(digest, size, backendIndex),
        /*notifyConsumer=*/ false);
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

  @Override
  protected FileStatus stat(Path path, boolean followSymlinks) throws IOException {
    FileArtifactValue metadata = getMetadataOrThrowFileNotFound(path);
    return new FileStatus() {
      @Override
      public boolean isFile() {
        return metadata.getType() == FileStateType.REGULAR_FILE;
      }

      @Override
      public boolean isDirectory() {
        // TODO(felly): Support directory awareness.
        return false;
      }

      @Override
      public boolean isSymbolicLink() {
        // TODO(felly): We should have minimal support for symlink awareness when looking at
        // output --> src and src --> src symlinks.
        return false;
      }

      @Override
      public boolean isSpecialFile() {
        return metadata.getType() == FileStateType.SPECIAL_FILE;
      }

      @Override
      public long getSize() {
        return metadata.getSize();
      }

      @Override
      public long getLastModifiedTime() {
        return metadata.getModifiedTime();
      }

      @Override
      public long getLastChangeTime() {
        return metadata.getModifiedTime();
      }

      @Override
      public long getNodeId() {
        throw new UnsupportedOperationException();
      }
    };
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
    PathFragment execPath = asExecPath(path);
    OutputMetadata output = outputs.getIfPresent(execPath);
    return output != null && outputs.asMap().remove(execPath, output);
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
  public byte[] getxattr(Path path, String name) throws IOException {
    FileArtifactValue metadata = getMetadataChecked(asExecPath(path));
    if (metadata instanceof RemoteFileArtifactValue) {
      RemoteFileArtifactValue remote = (RemoteFileArtifactValue) metadata;
      // TODO(b/80244718): inject ActionFileSystem from elsewhere and replace with correct metadata
      return ("/CENSORED_BY_LEAKR/"
              + remote.getLocationIndex()
              + "/"
              + LOWER_CASE_HEX.encode(remote.getDigest()))
          .getBytes(US_ASCII);
    }
    if (metadata instanceof SourceFileArtifactValue) {
      return resolveSourcePath((SourceFileArtifactValue) metadata).getxattr(name);
    }
    return getSourcePath(path.asFragment()).getxattr(name);
  }

  @Override
  protected byte[] getFastDigest(Path path, HashFunction hash) throws IOException {
    if (hash != HashFunction.MD5) {
      return null;
    }
    return getMetadataOrThrowFileNotFound(path).getDigest();
  }

  @Override
  protected Collection<String> getDirectoryEntries(Path path) throws IOException {
    // TODO(felly): Support directory traversal.
    return ImmutableList.of();
  }

  private static String createSymbolicLinkErrorMessage(
      Path linkPath, PathFragment targetFragment, String message) {
    return "createSymbolicLink(" + linkPath + ", " + targetFragment + "): " + message;
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    // TODO(shahan): this might need to be loosened, but will require more information
    Preconditions.checkArgument(
        targetFragment.isAbsolute(),
        "ActionFileSystem requires symlink targets to be absolute: %s -> %s",
        linkPath,
        targetFragment);

    // When creating symbolic links, it matters whether target is a source path or not because
    // the metadata needs to be handled differently in that case.
    PathFragment targetExecPath = null;
    int sourceRootIndex = -1; // index into sourceRoots or -1 if not a source
    if (targetFragment.startsWith(execRootFragment)) {
      targetExecPath = targetFragment.relativeTo(execRootFragment);
    } else {
      for (int i = 0; i < sourceRoots.size(); ++i) {
        if (targetFragment.startsWith(sourceRoots.get(i))) {
          targetExecPath = targetFragment.relativeTo(sourceRoots.get(i));
          sourceRootIndex = i;
          break;
        }
      }
      if (sourceRootIndex == -1) {
        throw new IllegalArgumentException(
            linkPath
                + " was not found under any known root: "
                + execRootFragment
                + ", "
                + sourceRoots);
      }
    }

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
    OutputMetadata outputHolder = Preconditions.checkNotNull(
        outputs.getUnchecked(asExecPath(linkPath)),
        "Unexpected null output path: %s", linkPath);
    if (sourceRootIndex >= 0) {
      Preconditions.checkState(!targetExecPath.startsWith(outputPathFragment), "Target exec path "
          + "%s does not start with output path fragment %s", targetExecPath, outputPathFragment);
      outputHolder.set(
          new SourceFileArtifactValue(
              targetExecPath, sourceRootIndex, inputMetadata.getDigest(), inputMetadata.getSize()),
          true);
    } else {
      outputHolder.set(inputMetadata, /*notifyConsumer=*/ true);
    }
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
    if (metadata instanceof InlineFileArtifactValue) {
      return ((InlineFileArtifactValue) metadata).getInputStream();
    }
    if (metadata instanceof SourceFileArtifactValue) {
      return resolveSourcePath((SourceFileArtifactValue) metadata).getInputStream();
    }
    if (metadata instanceof RemoteFileArtifactValue) {
      throw new IOException("ActionFileSystem cannot read remote file: " + path);
    }
    return getSourcePath(path.asFragment()).getInputStream();
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append) {
    Preconditions.checkArgument(!append, "ActionFileSystem doesn't support append.");
    return outputs.getUnchecked(asExecPath(path)).getOutputStream();
  }

  @Override
  public void renameTo(Path sourcePath, Path targetPath) throws IOException {
    PathFragment sourceExecPath = asExecPath(sourcePath);
    OutputMetadata sourceMetadata = outputs.getIfPresent(sourceExecPath);
    if (sourceMetadata == null) {
      throw new IOException("No output file at " + sourcePath + " to move to " + targetPath);
    }
    outputs.put(asExecPath(targetPath), sourceMetadata);
    outputs.invalidate(sourceExecPath);
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
      OutputMetadata metadataHolder = outputs.getIfPresent(execPath);
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
    return path.asFragment().startsWith(outputPathFragment);
  }

  private Path getSourcePath(PathFragment path) throws IOException {
    if (path.startsWith(outputPathFragment)) {
      throw new IOException("ActionFS cannot delegate to underlying output path for " + path);
    }
    return delegate.getPath(path);
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

  /** NB: resolves to the underlying filesytem instead of this one. */
  private Path resolveSourcePath(SourceFileArtifactValue metadata) throws IOException {
    return getSourcePath(sourceRoots.get(metadata.getSourceRootIndex()))
        .getRelative(metadata.getExecPath());
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

    public Artifact getArtifact() {
      return artifact;
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
          }
        }
      }
      return metadata;
    }
  }

  private class OutputMetadata {
    private final @Nullable Artifact artifact;
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
      if (notifyConsumer && artifact != null) {
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
          flush();
          super.close();
        }

        @Override
        public void flush() throws IOException {
          super.flush();
          byte[] data = toByteArray();
          set(new InlineFileArtifactValue(data, Hashing.md5().hashBytes(data).asBytes()),
              /*notifyConsumer=*/ true);
        }
      };
    }
  }
}
