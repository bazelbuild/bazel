// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.common.collect.Sets;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.Digest;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.protobuf.ByteString;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
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
 * ArtifactFunction}s in order to construct {@link ArtifactValue}s. Third, the {@link
 * FilesystemValueChecker} uses it to determine the set of output files to check for inter-build
 * modifications. Because all these use cases are slightly different, we must occasionally store two
 * versions of the data for a value (see {@link #getAdditionalOutputData} for more.
 */
@VisibleForTesting
public class FileAndMetadataCache implements ActionInputFileCache, MetadataHandler {
  /** This should never be read directly. Use {@link #getInputFileArtifactValue} instead. */
  private final Map<Artifact, FileArtifactValue> inputArtifactData;
  private final Map<Artifact, Collection<Artifact>> expandedInputMiddlemen;
  private final File execRoot;
  private final Map<ByteString, Artifact> reverseMap = new ConcurrentHashMap<>();
  private final ConcurrentMap<Artifact, FileValue> outputArtifactData =
      new ConcurrentHashMap<>();
  private final Set<Artifact> omittedOutputs = Sets.newConcurrentHashSet();
  // See #getAdditionalOutputData for documentation of this field.
  private final ConcurrentMap<Artifact, FileArtifactValue> additionalOutputData =
      new ConcurrentHashMap<>();
  private final Set<Artifact> injectedArtifacts = Sets.newConcurrentHashSet();
  private final ImmutableSet<Artifact> outputs;
  @Nullable private final SkyFunction.Environment env;
  private final TimestampGranularityMonitor tsgm;

  private static final Interner<ByteString> BYTE_INTERNER = Interners.newWeakInterner();

  public FileAndMetadataCache(Map<Artifact, FileArtifactValue> inputArtifactData,
      Map<Artifact, Collection<Artifact>> expandedInputMiddlemen, File execRoot,
      Iterable<Artifact> outputs, @Nullable SkyFunction.Environment env,
      TimestampGranularityMonitor tsgm) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
    this.expandedInputMiddlemen = Preconditions.checkNotNull(expandedInputMiddlemen);
    this.execRoot = Preconditions.checkNotNull(execRoot);
    this.outputs = ImmutableSet.copyOf(outputs);
    this.env = env;
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
    if (value == FileArtifactValue.MISSING_FILE_MARKER) {
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
    FileArtifactValue value = inputArtifactData.get(input);
    if (value != null) {
      return value;
    }
    if (outputs.contains(input)) {
      // When this method is called to calculate the metadata of an artifact, the artifact may be an
      // output artifact. Don't try to do anything then.
      return null;
    }
    if (!(input instanceof Artifact)) {
      // Maybe we're being asked for some strange constructed ActionInput coming from runfiles or
      // similar. We have no information about such things.
      return null;
    }
    // TODO(bazel-team): Remove this codepath once Skyframe has native input discovery, so all
    // inputs will already have metadata known.
    // ActionExecutionFunction may have passed in null environment if this action does not
    // discover inputs. In which case we should not have gotten here.
    Preconditions.checkNotNull(env, input);
    Artifact artifact = (Artifact) input;
    if (artifact.isSourceArtifact()) {
      // We might have no artifact data for discovered source inputs, and it's not worth storing
      // it in this cache, because it won't be reused across actions -- while we could request an
      // artifact from the graph, we would have to be tolerant to it not yet being present in the
      // graph yet, which adds complexity. Instead, we let the undeclared inputs handler stat it, so
      // it can be reused.
      return null;
    } else {
      // This getValue call is not expected to return null, because if the artifact is a
      // transitive dependency of this action (as it should be), it will already have been built,
      // so this call will return a value.
      // This getValue call is theoretically less efficient for a subsequent incremental build
      // than it would be to do a bulk getValues call after action execution, as is done for
      // source inputs. However, since almost all nodes requested here are transitive deps of an
      // already-declared dependency, change pruning will request these values serially, but they
      // will already have been built. So the only penalty is restarting ParallelEvaluator#run, as
      // opposed to traversing the entire downward transitive closure on a single thread.
      value = (FileArtifactValue) env.getValue(
          FileArtifactValue.key(artifact, /*argument ignored for derived artifacts*/false));
      return value;
    }
  }

  /**
   * We cache data for constant-metadata artifacts, even though it is technically unnecessary,
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
    }
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
    fileValue = fileValueFromArtifact(artifact, null, tsgm);
    FileValue oldFileValue = outputArtifactData.putIfAbsent(artifact, fileValue);
    checkInconsistentData(artifact, oldFileValue, value);
    return maybeStoreAdditionalData(artifact, fileValue, null);
  }

  /** Expands one of the input middlemen artifacts of the corresponding action. */
  public Collection<Artifact> expandInputMiddleman(Artifact middlemanArtifact) {
    Preconditions.checkState(middlemanArtifact.isMiddlemanArtifact(), middlemanArtifact);
    Collection<Artifact> result = expandedInputMiddlemen.get(middlemanArtifact);
    // Note that result may be null for non-aggregating middlemen.
    return result == null ? ImmutableSet.<Artifact>of() : result;
  }

  /**
   * Check that the new {@code data} we just calculated for an {@code artifact} agrees with the
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
    boolean useDigest = DigestUtils.useFileDigest(artifact, isFile, isFile ? data.getSize() : 0);
    if (useDigest && data.getDigest() != null) {
      // We do not need to store the FileArtifactValue separately -- the digest is in the file value
      // and that is all that is needed for this file's metadata.
      return new Metadata(data.getDigest());
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
  public void setDigestForVirtualArtifact(Artifact artifact, Digest digest) {
    Preconditions.checkState(artifact.isMiddlemanArtifact(), artifact);
    Preconditions.checkNotNull(digest, artifact);
    additionalOutputData.put(artifact,
        FileArtifactValue.createMiddleman(digest.asMetadata().digest));
  }

  @Override
  public void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest) {
    if (output instanceof Artifact) {
      Artifact artifact = (Artifact) output;
      Preconditions.checkState(injectedArtifacts.add(artifact), artifact);
      FileValue fileValue;
      try {
        // This call may do an unnecessary call to Path#getFastDigest to see if the digest is
        // readily available. We cannot pass the digest in, though, because if it is not available
        // from the filesystem, this FileValue will not compare equal to another one created for the
        // same file, because the other one will be missing its digest.
        fileValue = fileValueFromArtifact(artifact, FileStatusWithDigestAdapter.adapt(statNoFollow),
            tsgm);
        byte[] fileDigest = fileValue.getDigest();
        Preconditions.checkState(fileDigest == null || Arrays.equals(digest, fileDigest),
            "%s %s %s", artifact, digest, fileDigest);
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
  public void discardMetadata(Collection<Artifact> artifactList) {
    Preconditions.checkState(injectedArtifacts.isEmpty(),
        "Artifacts cannot be injected before action execution: %s", injectedArtifacts);
    outputArtifactData.keySet().removeAll(artifactList);
    additionalOutputData.keySet().removeAll(artifactList);
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
    if (value != null && value.getDigest() != null) {
      return true;
    }
    return artifact.getPath().isFile();
  }

  @Override
  public boolean isInjected(Artifact artifact) {
    return injectedArtifacts.contains(artifact);
  }

  /**
   * @return data for output files that was computed during execution. Should include data for all
   * non-middleman artifacts.
   */
  Map<Artifact, FileValue> getOutputData() {
    return outputArtifactData;
  }

  /**
   * Returns data for any output files whose metadata was not computable from the corresponding
   * entry in {@link #getOutputData}.
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

  @Override
  public long getSizeInBytes(ActionInput input) throws IOException {
    FileArtifactValue metadata = getInputFileArtifactValue(input);
    if (metadata != null) {
      return metadata.getSize();
    }
    return -1;
  }

  @Nullable
  @Override
  public File getFileFromDigest(ByteString digest) throws IOException {
    Artifact artifact = reverseMap.get(digest);
    if (artifact != null) {
      String relPath = artifact.getExecPathString();
      return relPath.startsWith("/") ? new File(relPath) : new File(execRoot, relPath);
    }
    return null;
  }

  @Nullable
  @Override
  public ByteString getDigest(ActionInput input) throws IOException {
    FileArtifactValue value = getInputFileArtifactValue(input);
    if (value != null) {
      byte[] bytes = value.getDigest();
      if (bytes != null) {
        ByteString digest = ByteString.copyFrom(BaseEncoding.base16().lowerCase().encode(bytes)
            .getBytes(StandardCharsets.US_ASCII));
        reverseMap.put(BYTE_INTERNER.intern(digest), (Artifact) input);
        return digest;
      }
    }
    return null;
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    return reverseMap.containsKey(digest);
  }

  static FileValue fileValueFromArtifact(Artifact artifact,
      @Nullable FileStatusWithDigest statNoFollow, TimestampGranularityMonitor tsgm)
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
