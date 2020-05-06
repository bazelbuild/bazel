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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
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
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Cache provided by an {@link ActionExecutionFunction}, allowing Blaze to obtain data from the
 * graph and to inject data (e.g. file digests) back into the graph. The cache can be in one of two
 * modes. After construction it acts as a cache for input and output metadata for the purpose of
 * checking for an action cache hit. When {@link #discardOutputMetadata} is called, it switches to a
 * mode where it calls chmod on output files before statting them. This is done here to ensure that
 * the chmod always comes before the stat in order to ensure that the stat is up to date.
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
 * output files to check for inter-build modifications.
 */
@VisibleForTesting
public final class ActionMetadataHandler implements MetadataHandler {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Data for input artifacts. Immutable.
   *
   * <p>This should never be read directly. Use {@link #getInputFileArtifactValue} instead.
   */
  private final ActionInputMap inputArtifactData;
  private final boolean missingArtifactsAllowed;
  private final ImmutableMap<PathFragment, FileArtifactValue> filesetMapping;

  /** Outputs that are to be omitted. */
  private final Set<Artifact> omittedOutputs = Sets.newConcurrentHashSet();

  private final ImmutableSet<Artifact> outputs;

  /**
   * The timestamp granularity monitor for this build.
   * Use {@link #getTimestampGranularityMonitor(Artifact)} to fetch this member.
   */
  @Nullable
  private final TimestampGranularityMonitor tsgm;
  private final ArtifactPathResolver artifactPathResolver;
  private final Path execRoot;

  /**
   * Whether the action is being executed or not; this flag is set to true in {@link
   * #discardOutputMetadata}.
   */
  private final AtomicBoolean executionMode = new AtomicBoolean(false);

  private final OutputStore store;

  @VisibleForTesting
  public ActionMetadataHandler(
      ActionInputMap inputArtifactData,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      boolean missingArtifactsAllowed,
      Iterable<Artifact> outputs,
      @Nullable TimestampGranularityMonitor tsgm,
      ArtifactPathResolver artifactPathResolver,
      OutputStore store,
      Path execRoot) {
    this.inputArtifactData = Preconditions.checkNotNull(inputArtifactData);
    this.missingArtifactsAllowed = missingArtifactsAllowed;
    this.outputs = ImmutableSet.copyOf(outputs);
    this.tsgm = tsgm;
    this.artifactPathResolver = artifactPathResolver;
    this.execRoot = execRoot;
    this.filesetMapping = expandFilesetMapping(Preconditions.checkNotNull(expandedFilesets));
    this.store = store;
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
  @Nullable
  private TimestampGranularityMonitor getTimestampGranularityMonitor(Artifact artifact) {
    return artifact.isConstantMetadata() ? null : tsgm;
  }

  private static FileArtifactValue metadataFromValue(FileArtifactValue value)
      throws FileNotFoundException {
    if (value == FileArtifactValue.MISSING_FILE_MARKER
        || value == FileArtifactValue.OMITTED_FILE_MARKER) {
      throw new FileNotFoundException();
    }
    return value;
  }

  private ImmutableMap<PathFragment, FileArtifactValue> expandFilesetMapping(
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {
    if (execRoot == null) {
      return ImmutableMap.of();
    }

    Map<PathFragment, FileArtifactValue> filesetMap = new HashMap<>();
    for (Map.Entry<Artifact, ImmutableList<FilesetOutputSymlink>> entry : filesets.entrySet()) {
      try {
        FilesetManifest fileset =
            FilesetManifest.constructFilesetManifest(
                entry.getValue(), execRoot.asFragment(), RelativeSymlinkBehavior.RESOLVE);
        for (Map.Entry<String, FileArtifactValue> favEntry :
            fileset.getArtifactValues().entrySet()) {
          if (favEntry.getValue().getDigest() != null) {
            filesetMap.put(PathFragment.create(favEntry.getKey()), favEntry.getValue());
          }
        }
      } catch (IOException e) {
        // If we cannot get the FileArtifactValue, then we will make a FileSystem call to get the
        // digest, so it is okay to skip and continue here.
        logger.atWarning().log(
            "Could not properly get digest for %s", entry.getKey().getExecPath());
        continue;
      }
    }
    return ImmutableMap.copyOf(filesetMap);
  }

  public ArtifactPathResolver getArtifactPathResolver() {
    return artifactPathResolver;
  }

  @Nullable
  private FileArtifactValue getInputFileArtifactValue(Artifact input) {
    if (isKnownOutput(input)) {
      return null;
    }
    return inputArtifactData.getMetadata(input);
  }

  private boolean isKnownOutput(Artifact artifact) {
    return outputs.contains(artifact)
        || (artifact.hasParent() && outputs.contains(artifact.getParent()));
  }

  @Override
  public FileArtifactValue getMetadata(ActionInput actionInput) throws IOException {
    if (!(actionInput instanceof Artifact)) {
      PathFragment inputPath = actionInput.getExecPath();
      PathFragment filesetKeyPath =
          inputPath.startsWith(execRoot.asFragment())
              ? inputPath.relativeTo(execRoot.asFragment())
              : inputPath;
      return filesetMapping.get(filesetKeyPath);
    }

    Artifact artifact = (Artifact) actionInput;
    FileArtifactValue value = getInputFileArtifactValue(artifact);
    if (value != null) {
      return metadataFromValue(value);
    }

    if (artifact.isSourceArtifact()) {
      // A discovered input we didn't have data for.
      // TODO(bazel-team): Change this to an assertion once Skyframe has native input discovery, so
      // all inputs will already have metadata known.
      if (!missingArtifactsAllowed) {
        throw new IllegalStateException(String.format("null for %s", artifact));
      }
      return null;
    } else if (artifact.isMiddlemanArtifact()) {
      // A middleman artifact's data was either already injected from the action cache checker using
      // #setDigestForVirtualArtifact, or it has the default middleman value.
      value = store.getArtifactData(artifact);
      if (value != null) {
        return metadataFromValue(value);
      }
      value = FileArtifactValue.DEFAULT_MIDDLEMAN;
      store.putArtifactData(artifact, value);
      return metadataFromValue(value);
    } else if (artifact.isTreeArtifact()) {
      TreeArtifactValue setValue = getTreeArtifactValue((SpecialArtifact) artifact);
      if (setValue != null && !setValue.equals(TreeArtifactValue.MISSING_TREE_ARTIFACT)) {
        return setValue.getMetadata();
      }
      // We use FileNotFoundExceptions to determine if an Artifact was or wasn't found.
      // Calling code depends on this particular exception.
      throw new FileNotFoundException(artifact + " not found");
    }
    // Fallthrough: the artifact must be a non-tree, non-middleman output artifact.

    // Don't store metadata for output artifacts that are not declared outputs of the action.
    if (!isKnownOutput(artifact)) {
      // Throw in strict mode.
      if (!missingArtifactsAllowed) {
        throw new IllegalStateException(String.format("null for %s", artifact));
      }
      return null;
    }

    // Check for existing metadata. It may have been injected. In either case, this method is called
    // from SkyframeActionExecutor to make sure that we have metadata for all action outputs, as the
    // results are then stored in Skyframe (and the action cache).
    FileArtifactValue fileMetadata = store.getArtifactData(artifact);
    if (fileMetadata != null) {
      return metadataFromValue(fileMetadata);
    }

    // No existing metadata; this can happen if the output metadata is not injected after a spawn
    // is executed. SkyframeActionExecutor.checkOutputs calls this method for every output file of
    // the action, which hits this code path. Another possibility is that an action runs multiple
    // spawns, and a subsequent spawn requests the metadata of an output of a previous spawn.
    //
    // Stat the file. All output artifacts of an action are deleted before execution, so if a file
    // exists, it was most likely created by the current action. There is a race condition here if
    // an external process creates (or modifies) the file between the deletion and this stat, which
    // we cannot solve.
    //
    // We only cache nonexistence here, not file system errors. It is unlikely that the file will be
    // requested from this cache too many times.
    fileMetadata = constructFileArtifactValue(artifact, /*statNoFollow=*/ null);
    return maybeStoreAdditionalData(artifact, fileMetadata, null);
  }

  @Override
  public ActionInput getInput(String execPath) {
    return inputArtifactData.getInput(execPath);
  }

  private FileArtifactValue maybeStoreAdditionalData(
      Artifact artifact, FileArtifactValue data, @Nullable byte[] injectedDigest)
      throws IOException {
    if (data.getType() == FileStateType.NONEXISTENT) {
      // Nonexistent files should only occur before executing an action.
      throw new FileNotFoundException(artifact.prettyPrint() + " does not exist");
    }

    if (data.getType() == FileStateType.SYMLINK) {
      // We never create a FileArtifactValue for an unresolved symlink without a digest (calling
      // readlink() is easy, unlike checksumming a potentially huge file)
      Preconditions.checkState(data.getDigest() != null);
      return data;
    }

    boolean isFile = data.getType() == FileStateType.REGULAR_FILE;

    if (isFile && !artifact.hasParent() && data.getDigest() != null) {
      // We do not need to store the FileArtifactValue separately -- the digest is in the file value
      // and that is all that is needed for this file's metadata.
      return data;
    }

    final FileArtifactValue value;

    if (data.getType() == FileStateType.DIRECTORY) {
      // This branch is taken when the output of an action is a directory:
      //   - A Fileset (in this case, Blaze is correct)
      //   - A directory someone created in a local action (in this case, changes under the
      //     directory may not be detected since we use the mtime of the directory for
      //     up-to-dateness checks)
      //   - A symlink to a source directory due to Filesets
      value =
          FileArtifactValue.createForDirectoryWithMtime(
              artifactPathResolver.toPath(artifact).getLastModifiedTime());
    } else {
      // Unfortunately, the ArtifactFileMetadata does not contain enough information for us to
      // calculate the corresponding FileArtifactValue -- either the metadata must use the modified
      // time, which we do not expose in the ArtifactFileMetadata, or the ArtifactFileMetadata
      // didn't store the digest So we store the metadata separately.
      // Use the ArtifactFileMetadata's digest if no digest was injected, or if the file can't be
      // digested.
      if (injectedDigest == null && isFile) {
        injectedDigest =
            DigestUtils.getDigestOrFail(artifactPathResolver.toPath(artifact), data.getSize());
      }

      value =
          FileArtifactValue.createFromInjectedDigest(
              data, injectedDigest, !artifact.isConstantMetadata());
    }

    store.putArtifactData(artifact, value);
    return metadataFromValue(value);
  }

  @Override
  public void setDigestForVirtualArtifact(Artifact artifact, byte[] digest) {
    Preconditions.checkArgument(artifact.isMiddlemanArtifact(), artifact);
    Preconditions.checkNotNull(digest, artifact);
    store.putArtifactData(artifact, FileArtifactValue.createProxy(digest));
  }

  private TreeArtifactValue getTreeArtifactValue(SpecialArtifact artifact) throws IOException {
    TreeArtifactValue value = store.getTreeArtifactData(artifact);
    if (value != null) {
      return value;
    }

    if (executionMode.get()) {
      // Preserve existing behavior: we don't set non-TreeArtifact directories
      // read only and executable. However, it's unusual for non-TreeArtifact outputs
      // to be directories.
      if (artifactPathResolver.toPath(artifact).isDirectory()) {
        setTreeReadOnlyAndExecutable(artifact, PathFragment.EMPTY_FRAGMENT);
      } else {
        setPathReadOnlyAndExecutable(
            ActionInputHelper.treeFileArtifact(artifact, PathFragment.EMPTY_FRAGMENT));
      }
    }

    Set<TreeFileArtifact> registeredContents = store.getTreeArtifactContents(artifact);
    if (registeredContents != null) {
      // Check that our registered outputs matches on-disk outputs. Only perform this check
      // when contents were explicitly registered.
      // TODO(bazel-team): Provide a way for actions to register empty TreeArtifacts.

      // By the time we're constructing TreeArtifactValues, use of the metadata handler
      // should be single threaded and there should be no race condition.
      // The current design of ActionMetadataHandler makes this hard to enforce.
      Set<PathFragment> paths =
          TreeArtifactValue.explodeDirectory(artifactPathResolver.toPath(artifact));
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
          throw new IOException(
              "Output file "
                  + missingFiles.iterator().next()
                  + " was registered, but not present on disk");
        }

        Set<TreeFileArtifact> extraFiles = Sets.difference(diskFiles, registeredContents);
        // extraFiles cannot be empty
        throw new IOException(
            "File "
                + extraFiles.iterator().next().getParentRelativePath()
                + ", present in TreeArtifact "
                + artifact
                + ", was not registered");
      }

      value = constructTreeArtifactValue(registeredContents);
    } else {
      value = constructTreeArtifactValueFromFilesystem(artifact);
    }

    store.putTreeArtifactData(artifact, value);
    return value;
  }

  private TreeArtifactValue constructTreeArtifactValue(Collection<TreeFileArtifact> contents)
      throws IOException {
    Map<TreeFileArtifact, FileArtifactValue> values =
        Maps.newHashMapWithExpectedSize(contents.size());

    for (TreeFileArtifact treeFileArtifact : contents) {
      FileArtifactValue fileMetadata = store.getArtifactData(treeFileArtifact);
      // This is similar to what's present in getRealMetadataForArtifact, except
      // we get back the ArtifactFileMetadata, not the metadata.
      // We do not cache exceptions besides nonexistence here, because it is unlikely that the
      // file will be requested from this cache too many times.
      if (fileMetadata == null) {
        try {
          fileMetadata = constructFileArtifactValue(treeFileArtifact, /*statNoFollow=*/ null);
        } catch (FileNotFoundException e) {
          String errorMessage =
              String.format(
                  "Failed to resolve relative path %s inside TreeArtifact %s. "
                      + "The associated file is either missing or is an invalid symlink.",
                  treeFileArtifact.getParentRelativePath(),
                  treeFileArtifact.getParent().getExecPathString());
          throw new IOException(errorMessage, e);
        }

        // TODO(janakr): we don't actually want the metadata for this TreeFileArtifact stored in the
        //  main metadata cache as of cl/297927844. Refactor and remove.
        // A minor hack: maybeStoreAdditionalData will force the data to be stored via
        // store.putAdditionalOutputData, if the underlying OutputStore supports it.
        fileMetadata = maybeStoreAdditionalData(treeFileArtifact, fileMetadata, null);
      }

      values.put(treeFileArtifact, fileMetadata);
    }

    return TreeArtifactValue.create(values);
  }

  private TreeArtifactValue constructTreeArtifactValueFromFilesystem(SpecialArtifact artifact)
      throws IOException {
    Preconditions.checkState(artifact.isTreeArtifact(), artifact);

    // Make sure the tree artifact root is a regular directory. Note that this is how the Action
    // is initialized, so this should hold unless the Action itself has deleted the root.
    if (!artifactPathResolver.toPath(artifact).isDirectory(Symlinks.NOFOLLOW)) {
      return TreeArtifactValue.MISSING_TREE_ARTIFACT;
    }

    Set<PathFragment> paths =
        TreeArtifactValue.explodeDirectory(artifactPathResolver.toPath(artifact));
    // If you're reading tree artifacts from disk while tree artifact contents are being injected,
    // something has gone terribly wrong.
    Object previousContents = store.getTreeArtifactContents(artifact);
    Preconditions.checkState(
        previousContents == null,
        "Race condition while constructing TreeArtifactValue: %s, %s",
        artifact,
        previousContents);
    return constructTreeArtifactValue(ActionInputHelper.asTreeFileArtifacts(artifact, paths));
  }

  @Override
  public void addExpandedTreeOutput(TreeFileArtifact output) {
    Preconditions.checkState(executionMode.get());
    store.addTreeArtifactContents(output.getParent(), output);
  }

  @Override
  public Iterable<TreeFileArtifact> getExpandedOutputs(Artifact artifact) {
    Set<TreeFileArtifact> contents = store.getTreeArtifactContents(artifact);
    return contents != null ? ImmutableSet.copyOf(contents) : ImmutableSet.of();
  }

  @Override
  public void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest) {
    Preconditions.checkState(executionMode.get());
    Preconditions.checkState(!output.isSymlink());

    // Assumption: any non-Artifact output is 'virtual' and should be ignored here.
    if (output instanceof Artifact) {
      final Artifact artifact = (Artifact) output;
      // We have to add the artifact to injectedFiles before calling constructFileArtifactValue
      // to avoid duplicate chmod calls.
      store.injectedFiles().add(artifact);
      FileArtifactValue fileMetadata;
      try {
        // This call may do an unnecessary call to Path#getFastDigest to see if the digest is
        // readily available. We cannot pass the digest in, though, because if it is not available
        // from the filesystem, this ArtifactFileMetadata will not compare equal to another one
        // created for the
        // same file, because the other one will be missing its digest.
        fileMetadata =
            constructFileArtifactValue(artifact, FileStatusWithDigestAdapter.adapt(statNoFollow));
        // Ensure the digest supplied matches the actual digest if it exists.
        byte[] fileDigest = fileMetadata.getDigest();
        if (fileDigest != null && !Arrays.equals(digest, fileDigest)) {
          BaseEncoding base16 = BaseEncoding.base16();
          String digestString = (digest != null) ? base16.encode(digest) : "null";
          String fileDigestString = base16.encode(fileDigest);
          throw new IllegalStateException(
              "Expected digest "
                  + digestString
                  + " for artifact "
                  + artifact
                  + ", but got "
                  + fileDigestString
                  + " ("
                  + fileMetadata
                  + ")");
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
        maybeStoreAdditionalData(artifact, fileMetadata, digest);
      } catch (IOException e) {
        throw new IllegalStateException(
            "Filesystem should not have been accessed while injecting data for "
                + artifact.prettyPrint(),
            e);
      }
    }
  }

  @Override
  public void injectRemoteFile(
      Artifact output, byte[] digest, long size, int locationIndex, String actionId) {
    Preconditions.checkArgument(
        isKnownOutput(output), output + " is not a declared output of this action");
    Preconditions.checkArgument(
        !output.isTreeArtifact(),
        "injectRemoteFile must not be " + "called on TreeArtifacts '%s'",
        output);
    Preconditions.checkState(
        executionMode.get(), "Tried to inject %s outside of execution", output);
    store.injectRemoteFile(output, digest, size, locationIndex, actionId);
  }

  @Override
  public void injectRemoteDirectory(
      SpecialArtifact output, Map<PathFragment, RemoteFileArtifactValue> children) {
    Preconditions.checkArgument(
        isKnownOutput(output), output + " is not a declared output of this action");
    Preconditions.checkArgument(output.isTreeArtifact(), "output must be a tree artifact");
    Preconditions.checkState(
        executionMode.get(), "Tried to inject %s outside of execution.", output);

    ImmutableMap.Builder<TreeFileArtifact, FileArtifactValue> childFileValues =
        ImmutableMap.builder();
    for (Map.Entry<PathFragment, RemoteFileArtifactValue> child : children.entrySet()) {
      childFileValues.put(
          ActionInputHelper.treeFileArtifact(output, child.getKey()), child.getValue());
    }

    TreeArtifactValue treeArtifactValue = TreeArtifactValue.create(childFileValues.build());
    store.putTreeArtifactData(output, treeArtifactValue);
  }

  @Override
  public void markOmitted(ActionInput output) {
    Preconditions.checkState(executionMode.get());
    if (output instanceof Artifact) {
      Artifact artifact = (Artifact) output;
      Preconditions.checkState(omittedOutputs.add(artifact), artifact);
      store.putArtifactData(artifact, FileArtifactValue.OMITTED_FILE_MARKER);
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
    Preconditions.checkState(
        store.injectedFiles().isEmpty(),
        "Files cannot be injected before action execution: %s",
        store.injectedFiles());
    Preconditions.checkState(
        omittedOutputs.isEmpty(),
        "Artifacts cannot be marked omitted before action execution: %s",
        omittedOutputs);
    store.clear();
  }

  @Override
  public void resetOutputs(Iterable<Artifact> outputs) {
    Preconditions.checkState(
        executionMode.get(), "resetOutputs() should only be called from within a running action.");
    for (Artifact output : outputs) {
      omittedOutputs.remove(output);
      store.remove(output);
    }
  }

  OutputStore getOutputStore() {
    return store;
  }

  /**
   * Constructs a new {@link FileArtifactValue}, saves it, and checks inconsistent data. This calls
   * chmod on the file if we're in executionMode.
   */
  private FileArtifactValue constructFileArtifactValue(
      Artifact artifact, @Nullable FileStatusWithDigest statNoFollow) throws IOException {
    // We first chmod the output files before we construct the FileContentsProxy. The proxy may use
    // ctime, which is affected by chmod.
    if (executionMode.get()) {
      Preconditions.checkState(!artifact.isTreeArtifact());
      setPathReadOnlyAndExecutable(artifact);
    }

    FileArtifactValue value =
        fileArtifactValueFromArtifact(
            artifact, artifactPathResolver, statNoFollow, getTimestampGranularityMonitor(artifact));
    store.putArtifactData(artifact, value);
    return value;
  }

  private static FileArtifactValue fileArtifactValueFromStat(
      RootedPath rootedPath,
      FileStatusWithDigest stat,
      boolean isConstantMetadata,
      TimestampGranularityMonitor tsgm)
      throws IOException {
    if (stat == null) {
      return FileArtifactValue.MISSING_FILE_MARKER;
    }

    FileStateValue fileStateValue = FileStateValue.createWithStatNoFollow(rootedPath, stat, tsgm);

    if (stat.isDirectory()) {
      return FileArtifactValue.createForDirectoryWithMtime(stat.getLastModifiedTime());
    } else {
      return FileArtifactValue.createForNormalFile(
          fileStateValue.getDigest(),
          fileStateValue.getContentsProxy(),
          stat.getSize(),
          !isConstantMetadata);
    }
  }

  @VisibleForTesting
  ImmutableMap<PathFragment, FileArtifactValue> getFilesetMapping() {
    return filesetMapping;
  }

  @VisibleForTesting
  static FileArtifactValue fileArtifactValueFromArtifact(
      Artifact artifact,
      @Nullable FileStatusWithDigest statNoFollow,
      @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    return fileArtifactValueFromArtifact(
        artifact, ArtifactPathResolver.IDENTITY, statNoFollow, tsgm);
  }

  private static FileArtifactValue fileArtifactValueFromArtifact(
      Artifact artifact,
      ArtifactPathResolver artifactPathResolver,
      @Nullable FileStatusWithDigest statNoFollow,
      @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    Preconditions.checkState(!artifact.isTreeArtifact());
    Preconditions.checkState(!artifact.isMiddlemanArtifact());

    Path pathNoFollow = artifactPathResolver.toPath(artifact);
    RootedPath rootedPathNoFollow =
        RootedPath.toRootedPath(
            artifactPathResolver.transformRoot(artifact.getRoot().getRoot()),
            artifact.getRootRelativePath());
    if (statNoFollow == null) {
      statNoFollow = FileStatusWithDigestAdapter.adapt(pathNoFollow.statIfFound(Symlinks.NOFOLLOW));
    }

    if (statNoFollow == null || !statNoFollow.isSymbolicLink()) {
      return fileArtifactValueFromStat(
          rootedPathNoFollow, statNoFollow, artifact.isConstantMetadata(), tsgm);
    }

    if (artifact.isSymlink()) {
      return FileArtifactValue.createForUnresolvedSymlink(pathNoFollow.readSymbolicLink());
    }

    // We use FileStatus#isSymbolicLink over Path#isSymbolicLink to avoid the unnecessary stat
    // done by the latter.  We need to protect against symlink cycles since
    // ArtifactFileMetadata#value assumes it's dealing with a file that's not in a symlink cycle.
    Path realPath = pathNoFollow.resolveSymbolicLinks();
    if (realPath.equals(pathNoFollow)) {
      throw new IOException("symlink cycle");
    }

    RootedPath realRootedPath =
        RootedPath.toRootedPathMaybeUnderRoot(
            realPath,
            ImmutableList.of(artifactPathResolver.transformRoot(artifact.getRoot().getRoot())));

    // TODO(bazel-team): consider avoiding a 'stat' here when the symlink target hasn't changed
    // and is a source file (since changes to those are checked separately).
    FileStatus realStat = realRootedPath.asPath().statIfFound(Symlinks.NOFOLLOW);
    FileStatusWithDigest realStatWithDigest = FileStatusWithDigestAdapter.adapt(realStat);
    return fileArtifactValueFromStat(
        realRootedPath, realStatWithDigest, artifact.isConstantMetadata(), tsgm);
  }

  private void setPathReadOnlyAndExecutable(Artifact artifact) throws IOException {
    // If the metadata was injected, we assume the mode is set correct and bail out early to avoid
    // the additional overhead of resetting it.
    if (store.injectedFiles().contains(artifact)) {
      return;
    }
    Path path = artifactPathResolver.toPath(artifact);
    if (path.isFile(Symlinks.NOFOLLOW)) { // i.e. regular files only.
      // We trust the files created by the execution engine to be non symlinks with expected
      // chmod() settings already applied.
      path.chmod(0555); // Sets the file read-only and executable.
    }
  }

  private void setTreeReadOnlyAndExecutable(SpecialArtifact parent, PathFragment subpath)
      throws IOException {
    Path path = artifactPathResolver.toPath(parent).getRelative(subpath);
    path.chmod(0555);
    Collection<Dirent> dirents = path.readdir(Symlinks.FOLLOW);
    for (Dirent dirent : dirents) {
      if (dirent.getType() == Type.DIRECTORY) {
        setTreeReadOnlyAndExecutable(parent, subpath.getChild(dirent.getName()));
      } else {
        setPathReadOnlyAndExecutable(
            ActionInputHelper.treeFileArtifact(parent, subpath.getChild(dirent.getName())));
      }
    }
  }
}
