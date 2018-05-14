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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.vfs.AbstractFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.protobuf.ByteString;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * File system for actions.
 *
 * <p>This class is thread-safe except that
 *
 * <ul>
 *   <li>{@link updateContext} and {@link updateInputData} must be called exclusively of any other
 *       methods.
 *   <li>This class relies on synchronized access to {@link env}. If there are other threads, that
 *       access {@link env}, they must also used synchronized access.
 * </ul>
 */
final class ActionFileSystem extends AbstractFileSystem implements ActionInputFileCache {
  private static final Logger LOGGER = Logger.getLogger(ActionFileSystem.class.getName());

  /**
   * Exec root and source roots.
   *
   * <p>First entry is exec root. Used to convert paths into exec paths.
   */
  private final LinkedHashSet<PathFragment> roots = new LinkedHashSet<>();

  /** exec path → artifact and metadata */
  private final Map<PathFragment, ArtifactAndMetadata> inputs;

  /** exec path → artifact and metadata */
  private final ImmutableMap<PathFragment, ArtifactAndMutableMetadata> outputs;

  /** digest → artifacts in {@link inputs} */
  private final ConcurrentHashMap<ByteString, Artifact> reverseMap;

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
      Map<Artifact, FileArtifactValue> inputData,
      Iterable<Artifact> allowedInputs,
      Iterable<Artifact> outputArtifacts) {
    try {
      Profiler.instance().startTask(ProfilerTask.ACTION_FS_STAGING, "staging");
      roots.add(computeExecRoot(outputArtifacts));

      // TODO(shahan): Underestimates because this doesn't account for discovered inputs. Improve
      // this estimate using data.
      this.reverseMap = new ConcurrentHashMap<>(inputData.size());

      HashMap<PathFragment, ArtifactAndMetadata> inputs = new HashMap<>();
      for (Map.Entry<Artifact, FileArtifactValue> entry : inputData.entrySet()) {
        Artifact input = entry.getKey();
        updateRootsIfSource(input);
        inputs.put(input.getExecPath(), new SimpleArtifactAndMetadata(input, entry.getValue()));
        updateReverseMapIfDigestExists(entry.getValue(), entry.getKey());
      }
      for (Artifact input : allowedInputs) {
        PathFragment execPath = input.getExecPath();
        inputs.computeIfAbsent(execPath, unused -> new OptionalInputArtifactAndMetadata(input));
        updateRootsIfSource(input);
      }
      this.inputs = inputs;

      validateRoots();

      this.outputs =
          Streams.stream(outputArtifacts)
              .collect(
                  ImmutableMap.toImmutableMap(
                      a -> a.getExecPath(), a -> new ArtifactAndMutableMetadata(a)));
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

  /** Input discovery changes the values of the input data map so it must be updated accordingly. */
  public void updateInputData(Map<Artifact, FileArtifactValue> inputData) {
    try {
      Profiler.instance().startTask(ProfilerTask.ACTION_FS_UPDATE, "update");
      boolean foundNewRoots = false;
      for (Map.Entry<Artifact, FileArtifactValue> entry : inputData.entrySet()) {
        ArtifactAndMetadata current = inputs.get(entry.getKey().getExecPath());
        if (current == null || isUnsetOptional(current)) {
          Artifact input = entry.getKey();
          inputs.put(input.getExecPath(), new SimpleArtifactAndMetadata(input, entry.getValue()));
          foundNewRoots = updateRootsIfSource(entry.getKey()) || foundNewRoots;
          updateReverseMapIfDigestExists(entry.getValue(), entry.getKey());
        }
      }
      if (foundNewRoots) {
        validateRoots();
      }
    } finally {
      Profiler.instance().completeTask(ProfilerTask.ACTION_FS_UPDATE);
    }
  }

  // -------------------- ActionInputFileCache implementation --------------------

  @Override
  @Nullable
  public FileArtifactValue getMetadata(ActionInput actionInput) {
    return apply(
        actionInput.getExecPath(),
        input -> {
          try {
            return input.getMetadata();
          } catch (IOException e) {
            // TODO(shahan): improve the handling of this error by propagating it correctly
            // through MetadataHandler.getMetadata().
            throw new IllegalStateException(e);
          }
        },
        output -> output.getMetadata(),
        () -> null);
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    // TODO(shahan): we assume this is never true, though the digests might be present. Should
    // this be relaxed for locally available source files?
    return false;
  }

  @Override
  @Nullable
  public Artifact getInputFromDigest(ByteString digest) {
    return reverseMap.get(digest);
  }

  @Override
  public Path getInputPath(ActionInput actionInput) {
    ArtifactAndMetadata input = inputs.get(actionInput.getExecPath());
    if (input != null) {
      return getPath(input.getArtifact().getPath().getPathString());
    }
    ArtifactAndMutableMetadata output = outputs.get(actionInput.getExecPath());
    if (output != null) {
      return getPath(output.getArtifact().getPath().getPathString());
    }
    // TODO(shahan): this might need to be relaxed
    throw new IllegalStateException(actionInput + " not found");
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
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
    return getMetadataOrThrowFileNotFound(path).getSize();
  }

  @Override
  public boolean delete(Path path) throws IOException {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected long getLastModifiedTime(Path path, boolean followSymlinks) throws IOException {
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
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
    ArtifactAndMetadata input = inputs.get(asExecPath(targetFragment));
    if (input == null) {
      throw new FileNotFoundException(
          createSymbolicLinkErrorMessage(
              linkPath, targetFragment, targetFragment + " is not an input."));
    }
    ArtifactAndMutableMetadata output = outputs.get(asExecPath(linkPath));
    if (output == null) {
      throw new FileNotFoundException(
          createSymbolicLinkErrorMessage(
              linkPath, targetFragment, linkPath + " is not an output."));
    }
    output.setMetadata(input.getMetadata());
  }

  @Override
  protected PathFragment readSymbolicLink(Path path) throws IOException {
    throw new UnsupportedOperationException(path.getPathString());
  }

  @Override
  protected boolean exists(Path path, boolean followSymlinks) {
    Preconditions.checkArgument(
        followSymlinks, "ActionFileSystem doesn't support no-follow: %s", path);
    return apply(path, input -> true, output -> output.getMetadata() != null, () -> false);
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
    // TODO(shahan): cleanup callers of this method and disable or maybe figure out a reasonable
    // implementation.
    LOGGER.severe("Raw read of path: " + path);
    return super.getInputStream(path);
  }

  @Override
  protected OutputStream getOutputStream(Path path, boolean append) throws IOException {
    // TODO(shahan): cleanup callers of this method and disable or maybe figure out a reasonable
    // implementation.
    LOGGER.severe("Raw write of path: " + path);
    return super.getOutputStream(path, append);
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
    for (PathFragment root : roots) {
      if (fragment.startsWith(root)) {
        return fragment.relativeTo(root);
      }
    }
    throw new IllegalArgumentException(fragment + " was not found under any known root: " + roots);
  }

  private boolean isOutput(Path path) {
    return outputs.containsKey(asExecPath(path));
  }

  /**
   * Lambda-based case implementation.
   *
   * <p>One of {@code inputOp} or {@code outputOp} will be called depending on whether {@code path}
   * is an input or output.
   */
  private <T> T apply(Path path, InputFileOperator<T> inputOp, OutputFileOperator<T> outputOp)
      throws IOException {
    PathFragment execPath = asExecPath(path);
    ArtifactAndMetadata input = inputs.get(execPath);
    if (input != null) {
      return inputOp.apply(input);
    }
    ArtifactAndMutableMetadata output = outputs.get(execPath);
    if (output != null) {
      return outputOp.apply(output);
    }
    throw new FileNotFoundException(path.getPathString());
  }

  /**
   * Apply variant that doesn't throw exceptions.
   *
   * <p>Useful for implementing existence-type methods.
   */
  private <T> T apply(
      Path path,
      Function<ArtifactAndMetadata, T> inputOp,
      Function<ArtifactAndMutableMetadata, T> outputOp,
      Supplier<T> notFoundOp) {
    return apply(asExecPath(path), inputOp, outputOp, notFoundOp);
  }

  private <T> T apply(
      PathFragment execPath,
      Function<ArtifactAndMetadata, T> inputOp,
      Function<ArtifactAndMutableMetadata, T> outputOp,
      Supplier<T> notFoundOp) {
    ArtifactAndMetadata input = inputs.get(execPath);
    if (input != null) {
      return inputOp.apply(input);
    }
    ArtifactAndMutableMetadata output = outputs.get(execPath);
    if (output != null) {
      return outputOp.apply(output);
    }
    return notFoundOp.get();
  }

  private boolean updateRootsIfSource(Artifact input) {
    if (input.isSourceArtifact()) {
      return roots.add(input.getRoot().getRoot().asPath().asFragment());
    }
    return false;
  }

  /**
   * The execution root is globally unique for a build so can be derived from any output.
   *
   * <p>Outputs must be nonempty.
   */
  private static PathFragment computeExecRoot(Iterable<Artifact> outputs) {
    Artifact derived = outputs.iterator().next();
    Preconditions.checkArgument(!derived.isSourceArtifact(), derived);
    PathFragment rootFragment = derived.getRoot().getRoot().asPath().asFragment();
    int rootSegments = rootFragment.segmentCount();
    int execSegments = derived.getRoot().getExecPath().segmentCount();
    return rootFragment.subFragment(0, rootSegments - execSegments);
  }

  /**
   * Verifies that no root is the prefix of any other root.
   *
   * <p>TODO(shahan): if this is insufficiently general, we can topologically order on the prefix
   * relation between roots.
   */
  private void validateRoots() {
    for (PathFragment root1 : roots) {
      for (PathFragment root2 : roots) {
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

  private static boolean isUnsetOptional(ArtifactAndMetadata input) {
    if (input instanceof OptionalInputArtifactAndMetadata) {
      OptionalInputArtifactAndMetadata optional = (OptionalInputArtifactAndMetadata) input;
      return !optional.hasMetadata();
    }
    return false;
  }

  private void updateReverseMapIfDigestExists(FileArtifactValue metadata, Artifact artifact) {
    if (metadata.getDigest() != null) {
      reverseMap.put(toByteString(metadata.getDigest()), artifact);
    }
  }

  private FileArtifactValue getMetadataOrThrowFileNotFound(Path path) throws IOException {
    return apply(
        path,
        input -> input.getMetadata(),
        output -> {
          if (output.getMetadata() == null) {
            throw new FileNotFoundException(path.getPathString());
          }
          return output.getMetadata();
        });
  }

  @Nullable
  private FileArtifactValue getMetadataUnchecked(Path path) {
    return apply(
        path,
        input -> {
          try {
            return input.getMetadata();
          } catch (IOException e) {
            // TODO(shahan): propagate this error correctly through higher level APIs.
            throw new IllegalStateException(e);
          }
        },
        output -> output.getMetadata(),
        () -> null);
  }

  @FunctionalInterface
  private static interface InputFileOperator<T> {
    T apply(ArtifactAndMetadata entry) throws IOException;
  }

  @FunctionalInterface
  private static interface OutputFileOperator<T> {
    T apply(ArtifactAndMutableMetadata entry) throws IOException;
  }

  @FunctionalInterface
  public static interface MetadataConsumer {
    void accept(Artifact artifact, FileArtifactValue value) throws IOException;
  }

  private abstract static class ArtifactAndMetadata {
    public abstract Artifact getArtifact();

    public abstract FileArtifactValue getMetadata() throws IOException;

    @Override
    public String toString() {
      String metadataText = null;
      try {
        metadataText = "" + getMetadata();
      } catch (IOException e) {
        metadataText = "Error getting metadata(" + e.getMessage() + ")";
      }
      return getArtifact() + ": " + metadataText;
    }
  }

  private static class SimpleArtifactAndMetadata extends ArtifactAndMetadata {
    private final Artifact artifact;
    private final FileArtifactValue metadata;

    private SimpleArtifactAndMetadata(Artifact artifact, FileArtifactValue metadata) {
      this.artifact = artifact;
      this.metadata = metadata;
    }

    @Override
    public Artifact getArtifact() {
      return artifact;
    }

    @Override
    public FileArtifactValue getMetadata() {
      return metadata;
    }
  }

  private class OptionalInputArtifactAndMetadata extends ArtifactAndMetadata {
    private final Artifact artifact;
    private volatile FileArtifactValue metadata = null;

    private OptionalInputArtifactAndMetadata(Artifact artifact) {
      this.artifact = artifact;
    }

    @Override
    public Artifact getArtifact() {
      return artifact;
    }

    @Override
    public FileArtifactValue getMetadata() throws IOException {
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
              // scanning. It might make sense to instead of bubbling this error out all the way
              // from within the distributor, to ensure that this metadata value exists when
              // creating the spawn from the include parser, which will require slightly fewer
              // layers of error propagation and there is some batching opportunity (across the
              // parallel expansion of the include scanner).
              synchronized (env) {
                metadata = (FileArtifactValue) env.getValue(ArtifactSkyKey.key(artifact, false));
              }
            } catch (InterruptedException e) {
              throw new InterruptedIOException(e.getMessage());
            }
            if (metadata == null) {
              throw new ActionExecutionFunction.MissingDepException();
            }
            updateReverseMapIfDigestExists(metadata, artifact);
          }
        }
      }
      return metadata;
    }

    public boolean hasMetadata() {
      return metadata != null;
    }
  }

  private class ArtifactAndMutableMetadata extends ArtifactAndMetadata {
    private final Artifact artifact;
    @Nullable private volatile FileArtifactValue metadata = null;

    @Override
    public Artifact getArtifact() {
      return artifact;
    }

    @Override
    @Nullable
    public FileArtifactValue getMetadata() {
      return metadata;
    }

    public void setMetadata(FileArtifactValue metadata) throws IOException {
      metadataConsumer.accept(artifact, metadata);
      this.metadata = metadata;
    }

    private ArtifactAndMutableMetadata(Artifact artifact) {
      this.artifact = artifact;
    }
  }
}
