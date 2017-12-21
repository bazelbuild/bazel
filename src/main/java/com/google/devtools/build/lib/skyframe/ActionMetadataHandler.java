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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.cache.Md5Digest;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Cache provided by an {@link ActionExecutionFunction}, allowing Blaze to obtain data from the
 * graph and to inject data (e.g. file digests) back into the graph. The cache can be in one of two
 * modes. After construction it acts as a cache for input and output metadata for the purpose of
 * checking for an action cache hit. When {@link #discardOutputMetadata} is called, it switches to
 * a mode where it calls chmod on output files before statting them. This is done here to ensure
 * that the chmod always comes before the stat in order to ensure that the stat is up to date.
 *
 * <p>Data for the action's inputs is injected into this cache on construction, using the Skyframe
 * graph as the source of truth.
 *
 * <p>As well, this cache collects data about the action's output files, which is used in three
 * ways. First, it is served as requested during action execution, primarily by the {@code
 * ActionCacheChecker} when determining if the action must be rerun, and then after the action is
 * run, to gather information about the outputs. Second, it is accessed by {@link ArtifactFunction}s
 * in order to construct {@link FileArtifactValue}s, and by this class itself to generate {@link
 * TreeArtifactValue}s. Third, the {@link FilesystemValueChecker} uses it to determine the set of
 * output files to check for inter-build modifications. Because all these use cases are slightly
 * different, we must occasionally store two versions of the data for a value. See {@link
 * #getAdditionalOutputData} for elaboration on the difference between these cases, and see the
 * javadoc for the various internal maps to see what is stored where.
 */
@VisibleForTesting
public class ActionMetadataHandler implements MetadataHandler {

  /**
   * Data for input artifacts. Immutable.
   *
   * <p>This should never be read directly. Use {@link #getInputFileArtifactValue} instead.</p>
   */
  private final Map<Artifact, FileArtifactValue> inputArtifactData;

  /** FileValues for each output Artifact. */
  private final ConcurrentMap<Artifact, FileValue> outputArtifactData =
      new ConcurrentHashMap<>();

  /**
   * Maps output TreeArtifacts to their contents. These maps are either injected or read
   * directly from the filesystem.
   * If the value is null, this means nothing was injected, and the output TreeArtifact
   * is to have its values read from disk instead.
   */
  private final ConcurrentMap<Artifact, Set<TreeFileArtifact>> outputDirectoryListings =
      new ConcurrentHashMap<>();

  /** Outputs that are to be omitted. */
  private final Set<Artifact> omittedOutputs = Sets.newConcurrentHashSet();

  /**
   * Contains RealArtifactValues when those values must be stored separately.
   * See {@link #getAdditionalOutputData()} for details.
   */
  private final ConcurrentMap<Artifact, FileArtifactValue> additionalOutputData =
      new ConcurrentHashMap<>();

  /**
   * Data for TreeArtifactValues, constructed from outputArtifactData and
   * additionalOutputFileData.
   */
  private final ConcurrentMap<Artifact, TreeArtifactValue> outputTreeArtifactData =
      new ConcurrentHashMap<>();

  /** Tracks which Artifacts have had metadata injected. */
  private final Set<Artifact> injectedFiles = Sets.newConcurrentHashSet();

  private final ImmutableSet<Artifact> outputs;

  /**
   * The timestamp granularity monitor for this build.
   * Use {@link #getTimestampGranularityMonitor(Artifact)} to fetch this member.
   */
  private final TimestampGranularityMonitor tsgm;

  /**
   * Whether the action is being executed or not; this flag is set to true in
   * {@link #discardOutputMetadata}.
   */
  private final AtomicBoolean executionMode = new AtomicBoolean(false);

  @VisibleForTesting
  public ActionMetadataHandler(Map<Artifact, FileArtifactValue> inputArtifactData,
      Iterable<Artifact> outputs,
      TimestampGranularityMonitor tsgm) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
    this.outputs = ImmutableSet.copyOf(outputs);
    this.tsgm = tsgm;
  }

  /**
   * Gets the {@link TimestampGranularityMonitor} to use for a given artifact.
   *
   * <p>If the artifact is of type "constant metadata", this returns null so that changes to such
   * artifacts do not tickle the timestamp granularity monitor, delaying the build for no reason.
   *
   * @param artifact the artifact for which to fetch the timestamp granularity monitor
   * @return the timestamp granularity monitor to use, which may be null
   */
  private TimestampGranularityMonitor getTimestampGranularityMonitor(Artifact artifact) {
    return artifact.isConstantMetadata() ? null : tsgm;
  }

  private static Metadata metadataFromValue(FileArtifactValue value) throws FileNotFoundException {
    if (value == FileArtifactValue.MISSING_FILE_MARKER
        || value == FileArtifactValue.OMITTED_FILE_MARKER) {
      throw new FileNotFoundException();
    }
    return value;
  }

  @Nullable
  private FileArtifactValue getInputFileArtifactValue(Artifact input) {
    if (outputs.contains(input)) {
      return null;
    }

    if (input.hasParent() && outputs.contains(input.getParent())) {
      return null;
    }

    return Preconditions.checkNotNull(inputArtifactData.get(input), input);
  }

  @Override
  public Metadata getMetadata(Artifact artifact) throws IOException {
    FileArtifactValue value = getInputFileArtifactValue(artifact);
    if (value != null) {
      return metadataFromValue(value);
    }
    if (artifact.isSourceArtifact()) {
      // A discovered input we didn't have data for.
      // TODO(bazel-team): Change this to an assertion once Skyframe has native input discovery, so
      // all inputs will already have metadata known.
      return null;
    } else if (artifact.isMiddlemanArtifact()) {
      // A middleman artifact's data was either already injected from the action cache checker using
      // #setDigestForVirtualArtifact, or it has the default middleman value.
      value = additionalOutputData.get(artifact);
      if (value != null) {
        return metadataFromValue(value);
      }
      value = FileArtifactValue.DEFAULT_MIDDLEMAN;
      FileArtifactValue oldValue = additionalOutputData.putIfAbsent(artifact, value);
      checkInconsistentData(artifact, oldValue, value);
      return metadataFromValue(value);
    } else if (artifact.isTreeArtifact()) {
      TreeArtifactValue setValue = getTreeArtifactValue(artifact);
      if (setValue != null && setValue != TreeArtifactValue.MISSING_TREE_ARTIFACT) {
        return setValue.getMetadata();
      }
      // We use FileNotFoundExceptions to determine if an Artifact was or wasn't found.
      // Calling code depends on this particular exception.
      throw new FileNotFoundException(artifact + " not found");
    }
    // It's an ordinary artifact.
    FileValue fileValue = outputArtifactData.get(artifact);
    if (fileValue != null) {
      // Non-middleman artifacts should only have additionalOutputData if they have
      // outputArtifactData. We don't assert this because of concurrency possibilities, but at least
      // we don't check additionalOutputData unless we expect that we might see the artifact there.
      value = additionalOutputData.get(artifact);
      // If additional output data is present for this artifact, we use it in preference to the
      // usual calculation.
      if (value != null) {
        return metadataFromValue(value);
      }
      if (!fileValue.exists()) {
        throw new FileNotFoundException(artifact.prettyPrint() + " does not exist");
      }
      return FileArtifactValue.createNormalFile(fileValue);
    }
    // We do not cache exceptions besides nonexistence here, because it is unlikely that the file
    // will be requested from this cache too many times.
    fileValue = constructFileValue(artifact, /*statNoFollow=*/ null);
    return maybeStoreAdditionalData(artifact, fileValue, null);
  }

  /**
   * Check that the new {@code data} we just calculated for an {@link Artifact} agrees with the
   * {@code oldData} (presumably calculated concurrently), if it was present.
   */
  // Not private only because used by SkyframeActionExecutor's metadata handler.
  static void checkInconsistentData(Artifact artifact,
      @Nullable Object oldData, Object data) throws IOException {
    if (oldData != null && !oldData.equals(data)) {
      // Another thread checked this file since we looked at the map, and got a different answer
      // than we did. Presumably the user modified the file between reads.
      throw new IOException("Data for " + artifact.prettyPrint() + " changed to " + data
          + " after it was calculated as " + oldData);
    }
  }

  /**
   * See {@link #getAdditionalOutputData} for why we sometimes need to store additional data, even
   * for normal (non-middleman) artifacts.
   */
  @Nullable
  private Metadata maybeStoreAdditionalData(Artifact artifact, FileValue data,
      @Nullable byte[] injectedDigest) throws IOException {
    if (!data.exists()) {
      // Nonexistent files should only occur before executing an action.
      throw new FileNotFoundException(artifact.prettyPrint() + " does not exist");
    }
    boolean isFile = data.isFile();
    if (isFile && !artifact.hasParent() && data.getDigest() != null) {
      // We do not need to store the FileArtifactValue separately -- the digest is in the file value
      // and that is all that is needed for this file's metadata.
      return FileArtifactValue.createNormalFile(data);
    }
    // Unfortunately, the FileValue does not contain enough information for us to calculate the
    // corresponding FileArtifactValue -- either the metadata must use the modified time, which we
    // do not expose in the FileValue, or the FileValue didn't store the digest So we store the
    // metadata separately.
    // Use the FileValue's digest if no digest was injected, or if the file can't be digested.
    injectedDigest = injectedDigest != null || !isFile ? injectedDigest : data.getDigest();
    FileArtifactValue value =
        FileArtifactValue.create(artifact, isFile, isFile ? data.getSize() : 0, injectedDigest);
    FileArtifactValue oldValue = additionalOutputData.putIfAbsent(artifact, value);
    checkInconsistentData(artifact, oldValue, value);
    return metadataFromValue(value);
  }

  @Override
  public void setDigestForVirtualArtifact(Artifact artifact, Md5Digest md5Digest) {
    Preconditions.checkArgument(artifact.isMiddlemanArtifact(), artifact);
    Preconditions.checkNotNull(md5Digest, artifact);
    additionalOutputData.put(
        artifact, FileArtifactValue.createProxy(md5Digest.getDigestBytesUnsafe()));
  }

  private Set<TreeFileArtifact> getTreeArtifactContents(Artifact artifact) {
    Preconditions.checkArgument(artifact.isTreeArtifact(), artifact);
    Set<TreeFileArtifact> contents = outputDirectoryListings.get(artifact);
    if (contents == null) {
      // Unfortunately, there is no such thing as a ConcurrentHashSet.
      contents = Collections.newSetFromMap(new ConcurrentHashMap<TreeFileArtifact, Boolean>());
      Set<TreeFileArtifact> oldContents = outputDirectoryListings.putIfAbsent(artifact, contents);
      // Avoid a race condition.
      if (oldContents != null) {
        contents = oldContents;
      }
    }
    return contents;
  }

  private TreeArtifactValue getTreeArtifactValue(Artifact artifact) throws IOException {
    TreeArtifactValue value = outputTreeArtifactData.get(artifact);
    if (value != null) {
      return value;
    }

    if (executionMode.get()) {
      // Preserve existing behavior: we don't set non-TreeArtifact directories
      // read only and executable. However, it's unusual for non-TreeArtifact outputs
      // to be directories.
      setTreeReadOnlyAndExecutable(artifact, PathFragment.EMPTY_FRAGMENT);
    }

    Set<TreeFileArtifact> registeredContents = outputDirectoryListings.get(artifact);
    if (registeredContents != null) {
      // Check that our registered outputs matches on-disk outputs. Only perform this check
      // when contents were explicitly registered.
      // TODO(bazel-team): Provide a way for actions to register empty TreeArtifacts.

      // By the time we're constructing TreeArtifactValues, use of the metadata handler
      // should be single threaded and there should be no race condition.
      // The current design of ActionMetadataHandler makes this hard to enforce.
      Set<PathFragment> paths = null;
      paths = TreeArtifactValue.explodeDirectory(artifact);
      Set<TreeFileArtifact> diskFiles = ActionInputHelper.asTreeFileArtifacts(artifact, paths);
      if (!diskFiles.equals(registeredContents)) {
        // There might be more than one error here. We first look for missing output files.
        Set<TreeFileArtifact> missingFiles = Sets.difference(registeredContents, diskFiles);
        if (!missingFiles.isEmpty()) {
          // Don't throw IOException--getMetadataMaybe() eats them.
          // TODO(bazel-team): Report this error in a better way when called by checkOutputs()
          // Currently it's hard to report this error without refactoring, since checkOutputs()
          // likes to substitute its own error messages upon catching IOException, and falls
          // through to unrecoverable error behavior on any other exception.
          throw new IOException("Output file " + missingFiles.iterator().next()
              + " was registered, but not present on disk");
        }

        Set<TreeFileArtifact> extraFiles = Sets.difference(diskFiles, registeredContents);
        // extraFiles cannot be empty
        throw new IOException(
            "File " + extraFiles.iterator().next().getParentRelativePath()
            + ", present in TreeArtifact " + artifact + ", was not registered");
      }

      value = constructTreeArtifactValue(registeredContents);
    } else {
      value = constructTreeArtifactValueFromFilesystem(artifact);
    }

    TreeArtifactValue oldValue = outputTreeArtifactData.putIfAbsent(artifact, value);
    checkInconsistentData(artifact, oldValue, value);
    return value;
  }

  private TreeArtifactValue constructTreeArtifactValue(Collection<TreeFileArtifact> contents)
      throws IOException {
    Map<TreeFileArtifact, FileArtifactValue> values =
        Maps.newHashMapWithExpectedSize(contents.size());

    for (TreeFileArtifact treeFileArtifact : contents) {
      FileArtifactValue cachedValue = additionalOutputData.get(treeFileArtifact);
      if (cachedValue == null) {
        FileValue fileValue = outputArtifactData.get(treeFileArtifact);
        // This is similar to what's present in getRealMetadataForArtifact, except
        // we get back the FileValue, not the metadata.
        // We do not cache exceptions besides nonexistence here, because it is unlikely that the
        // file will be requested from this cache too many times.
        if (fileValue == null) {
          try {
            fileValue = constructFileValue(treeFileArtifact, /*statNoFollow=*/ null);
          } catch (FileNotFoundException e) {
            String errorMessage = String.format(
                "Failed to resolve relative path %s inside TreeArtifact %s. "
                + "The associated file is either missing or is an invalid symlink.",
                treeFileArtifact.getParentRelativePath(),
                treeFileArtifact.getParent().getExecPathString());
            throw new IOException(errorMessage, e);
          }
        }

        // A minor hack: maybeStoreAdditionalData will force the data to be stored
        // in additionalOutputData.
        maybeStoreAdditionalData(treeFileArtifact, fileValue, null);
        cachedValue = Preconditions.checkNotNull(
            additionalOutputData.get(treeFileArtifact), treeFileArtifact);
      }

      values.put(treeFileArtifact, cachedValue);
    }

    return TreeArtifactValue.create(values);
  }

  private TreeArtifactValue constructTreeArtifactValueFromFilesystem(Artifact artifact)
      throws IOException {
    Preconditions.checkState(artifact.isTreeArtifact(), artifact);

    if (!artifact.getPath().isDirectory() || artifact.getPath().isSymbolicLink()) {
      return TreeArtifactValue.MISSING_TREE_ARTIFACT;
    }

    Set<PathFragment> paths = null;
    paths = TreeArtifactValue.explodeDirectory(artifact);
    // If you're reading tree artifacts from disk while outputDirectoryListings are being injected,
    // something has gone terribly wrong.
    Object previousDirectoryListing =
        outputDirectoryListings.put(artifact,
            Collections.newSetFromMap(new ConcurrentHashMap<TreeFileArtifact, Boolean>()));
    Preconditions.checkState(previousDirectoryListing == null,
        "Race condition while constructing TreArtifactValue: %s, %s",
        artifact, previousDirectoryListing);
    return constructTreeArtifactValue(ActionInputHelper.asTreeFileArtifacts(artifact, paths));
  }

  @Override
  public void addExpandedTreeOutput(TreeFileArtifact output) {
    Preconditions.checkState(executionMode.get());
    Set<TreeFileArtifact> values = getTreeArtifactContents(output.getParent());
    values.add(output);
  }

  @Override
  public Iterable<TreeFileArtifact> getExpandedOutputs(Artifact artifact) {
    return ImmutableSet.copyOf(getTreeArtifactContents(artifact));
  }

  @Override
  public void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest) {
    Preconditions.checkState(executionMode.get());
    // Assumption: any non-Artifact output is 'virtual' and should be ignored here.
    if (output instanceof Artifact) {
      final Artifact artifact = (Artifact) output;
      // We have to add the artifact to injectedFiles before calling constructFileValue to avoid
      // duplicate chmod calls.
      Preconditions.checkState(injectedFiles.add(artifact), artifact);
      FileValue fileValue;
      try {
        // This call may do an unnecessary call to Path#getFastDigest to see if the digest is
        // readily available. We cannot pass the digest in, though, because if it is not available
        // from the filesystem, this FileValue will not compare equal to another one created for the
        // same file, because the other one will be missing its digest.
        fileValue = constructFileValue(artifact, FileStatusWithDigestAdapter.adapt(statNoFollow));
        // Ensure the digest supplied matches the actual digest if it exists.
        byte[] fileDigest = fileValue.getDigest();
        if (fileDigest != null && !Arrays.equals(digest, fileDigest)) {
          BaseEncoding base16 = BaseEncoding.base16();
          String digestString = (digest != null) ? base16.encode(digest) : "null";
          String fileDigestString = base16.encode(fileDigest);
          throw new IllegalStateException("Expected digest " + digestString + " for artifact "
              + artifact + ", but got " + fileDigestString + " (" + fileValue + ")");
        }
      } catch (IOException e) {
        // Do nothing - we just failed to inject metadata. Real error handling will be done later,
        // when somebody will try to access that file.
        return;
      }
      // If needed, insert additional data. Note that this can only be true if the file is empty or
      // the filesystem does not support fast digests. Since we usually only inject digests when
      // running with a filesystem that supports fast digests, this is fairly unlikely.
      try {
        maybeStoreAdditionalData(artifact, fileValue, digest);
      } catch (IOException e) {
        if (fileValue.getSize() != 0) {
          // Empty files currently have their mtimes examined, and so could throw. No other files
          // should throw, since all filesystem access has already been done.
          throw new IllegalStateException(
              "Filesystem should not have been accessed while injecting data for "
                  + artifact.prettyPrint(), e);
        }
        // Ignore exceptions for empty files, as above.
      }
    }
  }

  @Override
  public void markOmitted(ActionInput output) {
    Preconditions.checkState(executionMode.get());
    if (output instanceof Artifact) {
      Artifact artifact = (Artifact) output;
      Preconditions.checkState(omittedOutputs.add(artifact), artifact);
      additionalOutputData.put(artifact, FileArtifactValue.OMITTED_FILE_MARKER);
    }
  }

  @Override
  public boolean artifactOmitted(Artifact artifact) {
    // TODO(ulfjack): this is currently unreliable, see the documentation on MetadataHandler.
    return omittedOutputs.contains(artifact);
  }

  @Override
  public void discardOutputMetadata() {
    boolean wasExecutionMode = executionMode.getAndSet(true);
    Preconditions.checkState(!wasExecutionMode);
    Preconditions.checkState(injectedFiles.isEmpty(),
        "Files cannot be injected before action execution: %s", injectedFiles);
    Preconditions.checkState(omittedOutputs.isEmpty(),
        "Artifacts cannot be marked omitted before action execution: %s", omittedOutputs);
    outputArtifactData.clear();
    outputDirectoryListings.clear();
    outputTreeArtifactData.clear();
    additionalOutputData.clear();
  }

  /** @return data for output files that was computed during execution. */
  Map<Artifact, FileValue> getOutputArtifactData() {
    return outputArtifactData;
  }

  /**
   * @return data for TreeArtifacts that was computed during execution. May contain copies of
   * {@link TreeArtifactValue#MISSING_TREE_ARTIFACT}.
   */
  Map<Artifact, TreeArtifactValue> getOutputTreeArtifactData() {
    return outputTreeArtifactData;
  }

  /**
   * Returns data for any output files whose metadata was not computable from the corresponding
   * entry in {@link #getOutputArtifactData}.
   *
   * <p>There are three reasons why we might not be able to compute metadata for an artifact from
   * the FileValue. First, middleman artifacts have no corresponding FileValues. Second, if
   * computing a file's digest is not fast, the FileValue does not do so, so a file on a filesystem
   * without fast digests has to have its metadata stored separately. Third, some files' metadata
   * (directories, empty files) contain their mtimes, which the FileValue does not expose, so that
   * has to be stored separately.
   *
   * <p>Note that for files that need digests, we can't easily inject the digest in the FileValue
   * because it would complicate equality-checking on subsequent builds -- if our filesystem doesn't
   * do fast digests, the comparison value would not have a digest.
   */
  Map<Artifact, FileArtifactValue> getAdditionalOutputData() {
    return additionalOutputData;
  }

  /**
   * Constructs a new FileValue, saves it, and checks inconsistent data. This calls chmod on the
   * file if we're in executionMode.
   */
  private FileValue constructFileValue(
      Artifact artifact, @Nullable FileStatusWithDigest statNoFollow)
          throws IOException {
    // We first chmod the output files before we construct the FileContentsProxy. The proxy may use
    // ctime, which is affected by chmod.
    if (executionMode.get()) {
      Preconditions.checkState(!artifact.isTreeArtifact());
      setPathReadOnlyAndExecutable(artifact);
    }

    FileValue value = fileValueFromArtifact(artifact, statNoFollow,
        getTimestampGranularityMonitor(artifact));
    FileValue oldFsValue = outputArtifactData.putIfAbsent(artifact, value);
    checkInconsistentData(artifact, oldFsValue, value);
    return value;
  }

  @VisibleForTesting
  static FileValue fileValueFromArtifact(Artifact artifact,
      @Nullable FileStatusWithDigest statNoFollow, @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    Path path = artifact.getPath();
    RootedPath rootedPath =
        RootedPath.toRootedPath(artifact.getRoot().getPath(), artifact.getRootRelativePath());
    if (statNoFollow == null) {
      statNoFollow = FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW));
      if (statNoFollow == null) {
        return FileValue.value(rootedPath, FileStateValue.NONEXISTENT_FILE_STATE_NODE,
            rootedPath, FileStateValue.NONEXISTENT_FILE_STATE_NODE);
      }
    }
    Path realPath = path;
    // We use FileStatus#isSymbolicLink over Path#isSymbolicLink to avoid the unnecessary stat
    // done by the latter.
    if (statNoFollow.isSymbolicLink()) {
      realPath = path.resolveSymbolicLinks();
      // We need to protect against symlink cycles since FileValue#value assumes it's dealing with a
      // file that's not in a symlink cycle.
      if (realPath.equals(path)) {
        throw new IOException("symlink cycle");
      }
    }
    RootedPath realRootedPath = RootedPath.toRootedPathMaybeUnderRoot(realPath,
        ImmutableList.of(artifact.getRoot().getPath()));
    FileStateValue fileStateValue;
    FileStateValue realFileStateValue;
    try {
      fileStateValue = FileStateValue.createWithStatNoFollow(rootedPath, statNoFollow, tsgm);
      // TODO(bazel-team): consider avoiding a 'stat' here when the symlink target hasn't changed
      // and is a source file (since changes to those are checked separately).
      realFileStateValue = realPath.equals(path) ? fileStateValue
          : FileStateValue.create(realRootedPath, tsgm);
    } catch (InconsistentFilesystemException e) {
      throw new IOException(e);
    }
    return FileValue.value(rootedPath, fileStateValue, realRootedPath, realFileStateValue);
  }

  private void setPathReadOnlyAndExecutable(Artifact artifact) throws IOException {
    // If the metadata was injected, we assume the mode is set correct and bail out early to avoid
    // the additional overhead of resetting it.
    if (injectedFiles.contains(artifact)) {
      return;
    }
    Path path = artifact.getPath();
    if (path.isFile(Symlinks.NOFOLLOW)) { // i.e. regular files only.
      // We trust the files created by the execution engine to be non symlinks with expected
      // chmod() settings already applied.
      path.chmod(0555);  // Sets the file read-only and executable.
    }
  }

  private void setTreeReadOnlyAndExecutable(Artifact parent, PathFragment subpath)
      throws IOException {
    Path path = parent.getPath().getRelative(subpath);
    if (path.isDirectory()) {
      path.chmod(0555);
      for (Path child : path.getDirectoryEntries()) {
        setTreeReadOnlyAndExecutable(parent, subpath.getChild(child.getBaseName()));
      }
    } else {
      setPathReadOnlyAndExecutable(ActionInputHelper.treeFileArtifact(parent, subpath));
    }
  }
}
