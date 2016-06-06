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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.cache.Digest;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue.TreeArtifactException;
import com.google.devtools.build.lib.util.Preconditions;
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

import javax.annotation.Nullable;

/**
 * Cache provided by an {@link ActionExecutionFunction}, allowing Blaze to obtain data from the
 * graph and to inject data (e.g. file digests) back into the graph.
 *
 * <p>Data for the action's inputs is injected into this cache on construction, using the graph as
 * the source of truth.
 *
 * <p>As well, this cache collects data about the action's output files, which is used in three
 * ways. First, it is served as requested during action execution, primarily by the {@code
 * ActionCacheChecker} when determining if the action must be rerun, and then after the action is
 * run, to gather information about the outputs. Second, it is accessed by {@link
 * ArtifactFunction}s in order to construct {@link ArtifactValue}, and by this class itself to
 * generate {@link TreeArtifactValue}s. Third, the {@link
 * FilesystemValueChecker} uses it to determine the set of output files to check for inter-build
 * modifications. Because all these use cases are slightly different, we must occasionally store two
 * versions of the data for a value. See {@link #getAdditionalOutputData} for elaboration on
 * the difference between these cases, and see the javadoc for the various internal maps to see
 * what is stored where.
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
  private final TimestampGranularityMonitor tsgm;

  @VisibleForTesting
  public ActionMetadataHandler(Map<Artifact, FileArtifactValue> inputArtifactData,
      Iterable<Artifact> outputs,
      TimestampGranularityMonitor tsgm) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
    this.outputs = ImmutableSet.copyOf(outputs);
    this.tsgm = tsgm;
  }

  @Override
  public Metadata getMetadataMaybe(Artifact artifact) {
    try {
      return getMetadata(artifact);
    } catch (IOException e) {
      return null;
    }
  }

  private static Metadata metadataFromValue(FileArtifactValue value) throws FileNotFoundException {
    if (value == FileArtifactValue.MISSING_FILE_MARKER
        || value == FileArtifactValue.OMITTED_FILE_MARKER) {
      throw new FileNotFoundException();
    }
    // If the file is empty or a directory, we need to return the mtime because the action cache
    // uses mtime to determine if this artifact has changed.  We do not optimize for this code
    // path (by storing the mtime somewhere) because we eventually may be switching to use digests
    // for empty files. We want this code path to go away somehow too for directories (maybe by
    // implementing FileSet in Skyframe).
    return value.getSize() > 0
        ? new Metadata(value.getDigest())
        : new Metadata(value.getModifiedTime());
  }

  @Override
  public Metadata getMetadata(Artifact artifact) throws IOException {
    Metadata metadata = getRealMetadata(artifact);
    return artifact.isConstantMetadata() ? Metadata.CONSTANT_METADATA : metadata;
  }

  @Nullable
  private FileArtifactValue getInputFileArtifactValue(ActionInput input) {
    if (outputs.contains(input) || !(input instanceof Artifact)) {
      return null;
    }
    return Preconditions.checkNotNull(inputArtifactData.get(input), input);
  }

  /**
   * Get the real (viz. on-disk) metadata for an Artifact.
   * A key assumption is that getRealMetadata() will be called for every Artifact in this
   * ActionMetadataHandler, to populate additionalOutputData and outputTreeArtifactData.
   *
   * <p>We cache data for constant-metadata artifacts, even though it is technically unnecessary,
   * because the data stored in this cache is consumed by various parts of Blaze via the {@link
   * ActionExecutionValue} (for now, {@link FilesystemValueChecker} and {@link ArtifactFunction}).
   * It is simpler for those parts if every output of the action is present in the cache. However,
   * we must not return the actual metadata for a constant-metadata artifact.
   */
  private Metadata getRealMetadata(Artifact artifact) throws IOException {
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
      return new Metadata(Preconditions.checkNotNull(fileValue.getDigest(), artifact));
    }
    // We do not cache exceptions besides nonexistence here, because it is unlikely that the file
    // will be requested from this cache too many times.
    fileValue = constructFileValue(artifact, null);
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
    if (!artifact.hasParent()) {
      // Artifacts may use either the "real" digest or the mtime, if the file is size 0.
      boolean isFile = data.isFile();
      boolean useDigest = DigestUtils.useFileDigest(isFile, isFile ? data.getSize() : 0);
      if (useDigest && data.getDigest() != null) {
        // We do not need to store the FileArtifactValue separately -- the digest is in the
        // file value and that is all that is needed for this file's metadata.
        return new Metadata(data.getDigest());
      }
      // Unfortunately, the FileValue does not contain enough information for us to calculate the
      // corresponding FileArtifactValue -- either the metadata must use the modified time, which we
      // do not expose in the FileValue, or the FileValue didn't store the digest So we store the
      // metadata separately.
      // Use the FileValue's digest if no digest was injected, or if the file can't be digested.
      injectedDigest = injectedDigest != null || !isFile ? injectedDigest : data.getDigest();
      FileArtifactValue value =
          FileArtifactValue.create(
              (Artifact) artifact, isFile, isFile ? data.getSize() : 0, injectedDigest);
      FileArtifactValue oldValue = additionalOutputData.putIfAbsent((Artifact) artifact, value);
      checkInconsistentData(artifact, oldValue, value);
      return metadataFromValue(value);
    } else {
      // We are dealing with artifacts inside a tree artifact.
      FileArtifactValue value =
          FileArtifactValue.createWithDigest(artifact.getPath(), injectedDigest, data.getSize());
      FileArtifactValue oldValue = additionalOutputData.putIfAbsent(artifact, value);
      checkInconsistentData(artifact, oldValue, value);
      return new Metadata(value.getDigest());
    }
  }

  @Override
  public void setDigestForVirtualArtifact(Artifact artifact, Digest digest) {
    Preconditions.checkArgument(artifact.isMiddlemanArtifact(), artifact);
    Preconditions.checkNotNull(digest, artifact);
    additionalOutputData.put(artifact,
        FileArtifactValue.createProxy(digest.asMetadata().digest));
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

    Set<TreeFileArtifact> registeredContents = outputDirectoryListings.get(artifact);
    if (registeredContents != null) {
      // Check that our registered outputs matches on-disk outputs. Only perform this check
      // when contents were explicitly registered.
      // TODO(bazel-team): Provide a way for actions to register empty TreeArtifacts.

      // By the time we're constructing TreeArtifactValues, use of the metadata handler
      // should be single threaded and there should be no race condition.
      // The current design of ActionMetadataHandler makes this hard to enforce.
      Set<PathFragment> paths = null;
      try {
        paths = TreeArtifactValue.explodeDirectory(artifact);
      } catch (TreeArtifactException e) {
        throw new IllegalStateException(e);
      }
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
          throw new IllegalStateException("Output file " + missingFiles.iterator().next()
              + " was registered, but not present on disk");
        }

        Set<TreeFileArtifact> extraFiles = Sets.difference(diskFiles, registeredContents);
        // extraFiles cannot be empty
        throw new IllegalStateException(
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
          fileValue = constructFileValue(treeFileArtifact, /*statNoFollow=*/ null);
          // A minor hack: maybeStoreAdditionalData will force the data to be stored
          // in additionalOutputData.
          maybeStoreAdditionalData(treeFileArtifact, fileValue, null);
        }
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
    try {
      paths = TreeArtifactValue.explodeDirectory(artifact);
    } catch (TreeArtifactException e) {
      throw new IllegalStateException(e);
    }
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
    Set<TreeFileArtifact> values = getTreeArtifactContents(output.getParent());
    values.add(output);
  }

  @Override
  public void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest) {
    // Assumption: any non-Artifact output is 'virtual' and should be ignored here.
    if (output instanceof Artifact) {
      final Artifact artifact = (Artifact) output;
      Preconditions.checkState(injectedFiles.add(artifact), artifact);
      FileValue fileValue;
      try {
        // This call may do an unnecessary call to Path#getFastDigest to see if the digest is
        // readily available. We cannot pass the digest in, though, because if it is not available
        // from the filesystem, this FileValue will not compare equal to another one created for the
        // same file, because the other one will be missing its digest.
        fileValue = fileValueFromArtifact(artifact,
            FileStatusWithDigestAdapter.adapt(statNoFollow), tsgm);
        // Ensure the digest supplied matches the actual digest if it exists.
        byte[] fileDigest = fileValue.getDigest();
        if (fileDigest != null && !Arrays.equals(digest, fileDigest)) {
          BaseEncoding base16 = BaseEncoding.base16();
          String digestString = (digest != null) ? base16.encode(digest) : "null";
          String fileDigestString = base16.encode(fileDigest);
          throw new IllegalStateException("Expected digest " + digestString + " for artifact "
              + artifact + ", but got " + fileDigestString + " (" + fileValue + ")");
        }
        outputArtifactData.put(artifact, fileValue);
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
    if (output instanceof Artifact) {
      Artifact artifact = (Artifact) output;
      Preconditions.checkState(omittedOutputs.add(artifact), artifact);
      additionalOutputData.put(artifact, FileArtifactValue.OMITTED_FILE_MARKER);
    }
  }

  @Override
  public boolean artifactOmitted(Artifact artifact) {
    return omittedOutputs.contains(artifact);
  }

  @Override
  public void discardOutputMetadata() {
    Preconditions.checkState(injectedFiles.isEmpty(),
        "Files cannot be injected before action execution: %s", injectedFiles);
    Preconditions.checkState(omittedOutputs.isEmpty(),
        "Artifacts cannot be marked omitted before action execution: %s", omittedOutputs);
    outputArtifactData.clear();
    outputDirectoryListings.clear();
    outputTreeArtifactData.clear();
    additionalOutputData.clear();
  }

  @Override
  public boolean artifactExists(Artifact artifact) {
    Preconditions.checkState(!artifactOmitted(artifact), artifact);
    return getMetadataMaybe(artifact) != null;
  }

  @Override
  public boolean isRegularFile(Artifact artifact) {
    // Currently this method is used only for genrule input directory checks. If we need to call
    // this on output artifacts too, this could be more efficient.
    FileArtifactValue value = getInputFileArtifactValue(artifact);
    if (value != null && value.isFile()) {
      return true;
    }
    return artifact.getPath().isFile();
  }

  @Override
  public boolean isInjected(Artifact file) {
    return injectedFiles.contains(file);
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

  /** Constructs a new FileValue, saves it, and checks inconsistent data. */
  FileValue constructFileValue(Artifact artifact, @Nullable FileStatusWithDigest statNoFollow)
      throws IOException {
    FileValue value = fileValueFromArtifact(artifact, statNoFollow, tsgm);
    FileValue oldFsValue = outputArtifactData.putIfAbsent(artifact, value);
    checkInconsistentData(artifact, oldFsValue, null);
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
}
