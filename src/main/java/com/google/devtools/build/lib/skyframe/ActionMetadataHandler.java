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
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FileArtifactValue;
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

  /**
   * If {@code value} represents an existing file, returns it as is, otherwise throws {@link
   * FileNotFoundException}.
   */
  private static FileArtifactValue checkExists(FileArtifactValue value, Artifact artifact)
      throws FileNotFoundException {
    if (value == FileArtifactValue.MISSING_FILE_MARKER
        || value == FileArtifactValue.OMITTED_FILE_MARKER) {
      throw new FileNotFoundException(artifact + " does not exist");
    }
    return Preconditions.checkNotNull(value, artifact);
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
      return checkExists(value, artifact);
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
        return checkExists(value, artifact);
      }
      value = FileArtifactValue.DEFAULT_MIDDLEMAN;
      store.putArtifactData(artifact, value);
      return checkExists(value, artifact);
    } else if (artifact.isTreeArtifact()) {
      TreeArtifactValue setValue = getTreeArtifactValue((SpecialArtifact) artifact);
      if (setValue != null && !setValue.equals(TreeArtifactValue.MISSING_TREE_ARTIFACT)) {
        return setValue.getMetadata();
      }
      // We use FileNotFoundExceptions to determine if an Artifact was or wasn't found.
      // Calling code depends on this particular exception.
      throw new FileNotFoundException(artifact + " not found");
    }

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
    value = store.getArtifactData(artifact);
    if (value != null) {
      return checkExists(value, artifact);
    }
    // This artifact was not injected directly to the store, but it may have been injected as part
    // of a tree artifact.
    // TODO(jhorvitz): We can skip this for action template expansion artifacts.
    if (artifact.hasParent()) {
      TreeArtifactValue tree = store.getTreeArtifactData(artifact.getParent());
      if (tree != null) {
        value = tree.getChildValues().get(artifact);
        if (value != null) {
          return checkExists(value, artifact);
        }
      }
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
    value = constructFileArtifactValueFromFilesystem(artifact);
    store.putArtifactData(artifact, value);
    return checkExists(value, artifact);
  }

  @Override
  public ActionInput getInput(String execPath) {
    return inputArtifactData.getInput(execPath);
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
            TreeFileArtifact.createTreeOutput(artifact, PathFragment.EMPTY_FRAGMENT));
      }
    }

    value = constructTreeArtifactValueFromFilesystem(artifact);
    store.putTreeArtifactData(artifact, value);
    return value;
  }

  private TreeArtifactValue constructTreeArtifactValueFromFilesystem(SpecialArtifact parent)
      throws IOException {
    Preconditions.checkState(parent.isTreeArtifact(), parent);

    // Make sure the tree artifact root is a regular directory. Note that this is how the Action
    // is initialized, so this should hold unless the Action itself has deleted the root.
    if (!artifactPathResolver.toPath(parent).isDirectory(Symlinks.NOFOLLOW)) {
      return TreeArtifactValue.MISSING_TREE_ARTIFACT;
    }

    Set<PathFragment> paths =
        TreeArtifactValue.explodeDirectory(artifactPathResolver.toPath(parent));

    Map<TreeFileArtifact, FileArtifactValue> values = Maps.newHashMapWithExpectedSize(paths.size());
    for (PathFragment path : paths) {
      TreeFileArtifact treeFileArtifact = TreeFileArtifact.createTreeOutput(parent, path);
      FileArtifactValue fileMetadata = store.getArtifactData(treeFileArtifact);
      if (fileMetadata == null) {
        try {
          fileMetadata = constructFileArtifactValueFromFilesystem(treeFileArtifact);
        } catch (FileNotFoundException e) {
          String errorMessage =
              String.format(
                  "Failed to resolve relative path %s inside TreeArtifact %s. "
                      + "The associated file is either missing or is an invalid symlink.",
                  treeFileArtifact.getParentRelativePath(),
                  treeFileArtifact.getParent().getExecPathString());
          throw new IOException(errorMessage, e);
        }
      }

      values.put(treeFileArtifact, fileMetadata);
    }

    return TreeArtifactValue.create(values);
  }

  @Override
  public ImmutableSet<TreeFileArtifact> getExpandedOutputs(Artifact artifact) {
    TreeArtifactValue treeArtifact = store.getTreeArtifactData(artifact);
    return treeArtifact != null ? treeArtifact.getChildren() : ImmutableSet.of();
  }

  @Override
  public FileArtifactValue constructMetadataForDigest(
      Artifact output, FileStatus statNoFollow, byte[] digest) throws IOException {
    Preconditions.checkState(executionMode.get());
    Preconditions.checkState(!output.isSymlink());
    Preconditions.checkNotNull(digest);

    // We have to add the artifact to injectedFiles before calling constructFileArtifactValue to
    // avoid duplicate chmod calls.
    store.injectedFiles().add(output);

    // TODO(jhorvitz): This unnecessarily calls getFastDigest even though we know the digest.
    return constructFileArtifactValue(
        output, FileStatusWithDigestAdapter.adapt(statNoFollow), digest);
  }

  @Override
  public void injectFile(Artifact output, FileArtifactValue metadata) {
    Preconditions.checkArgument(
        isKnownOutput(output), "%s is not a declared output of this action", output);
    Preconditions.checkArgument(
        !output.isTreeArtifact(), "injectFile must not be called on TreeArtifacts: %s", output);
    Preconditions.checkState(
        executionMode.get(), "Tried to inject %s outside of execution", output);
    store.injectOutputData(output, metadata);
  }

  @Override
  public void injectDirectory(
      SpecialArtifact output, Map<TreeFileArtifact, FileArtifactValue> children) {
    Preconditions.checkArgument(
        isKnownOutput(output), "%s is not a declared output of this action", output);
    Preconditions.checkArgument(output.isTreeArtifact(), "output must be a tree artifact");
    Preconditions.checkState(
        executionMode.get(), "Tried to inject %s outside of execution", output);
    store.putTreeArtifactData(output, TreeArtifactValue.create(children));
  }

  @Override
  public void markOmitted(Artifact output) {
    Preconditions.checkState(
        executionMode.get(), "Tried to mark %s omitted outside of execution", output);
    boolean newlyOmitted = omittedOutputs.add(output);
    if (output.isTreeArtifact()) {
      // Tolerate marking a tree artifact as omitted multiple times so that callers don't have to
      // deduplicate when a tree artifact has several omitted children.
      if (newlyOmitted) {
        store.putTreeArtifactData((SpecialArtifact) output, TreeArtifactValue.OMITTED_TREE_MARKER);
      }
    } else {
      Preconditions.checkState(newlyOmitted, "%s marked as omitted twice", output);
      store.putArtifactData(output, FileArtifactValue.OMITTED_FILE_MARKER);
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
   * Constructs a new {@link FileArtifactValue} by reading from the file system and checks
   * inconsistent data. This calls chmod on the file if we're in execution mode.
   */
  private FileArtifactValue constructFileArtifactValueFromFilesystem(Artifact artifact)
      throws IOException {
    return constructFileArtifactValue(artifact, /*statNoFollow=*/ null, /*injectedDigest=*/ null);
  }

  /**
   * Constructs a new {@link FileArtifactValue} and checks inconsistent data. This calls chmod on
   * the file if we're in execution mode.
   */
  private FileArtifactValue constructFileArtifactValue(
      Artifact artifact,
      @Nullable FileStatusWithDigest statNoFollow,
      @Nullable byte[] injectedDigest)
      throws IOException {
    // We first chmod the output files before we construct the FileContentsProxy. The proxy may use
    // ctime, which is affected by chmod.
    if (executionMode.get()) {
      Preconditions.checkState(!artifact.isTreeArtifact());
      setPathReadOnlyAndExecutable(artifact);
    }

    FileArtifactValue value =
        fileArtifactValueFromArtifact(
            artifact, artifactPathResolver, statNoFollow, getTimestampGranularityMonitor(artifact));

    // Ensure the injected digest supplied matches the actual digest if it exists.
    byte[] fileDigest = value.getDigest();
    if (fileDigest != null
        && injectedDigest != null
        && !Arrays.equals(injectedDigest, fileDigest)) {
      throw new IllegalStateException(
          String.format(
              "Expected digest %s for artifact %s, but got %s (%s)",
              BaseEncoding.base16().encode(injectedDigest),
              artifact,
              BaseEncoding.base16().encode(fileDigest),
              value));
    }

    FileStateType type = value.getType();

    if (!type.exists()) {
      // Nonexistent files should only occur before executing an action.
      throw new FileNotFoundException(artifact.prettyPrint() + " does not exist");
    }

    if (type.isSymlink()) {
      // We never create a FileArtifactValue for an unresolved symlink without a digest (calling
      // readlink() is easy, unlike checksumming a potentially huge file).
      Preconditions.checkNotNull(fileDigest, "%s missing digest", value);
      return value;
    }

    if (type.isFile() && !artifact.hasParent() && fileDigest != null) {
      // We do not need to store the FileArtifactValue separately -- the digest is in the file value
      // and that is all that is needed for this file's metadata.
      return value;
    }

    if (type.isDirectory()) {
      // This branch is taken when the output of an action is a directory:
      //   - A Fileset (in this case, Blaze is correct)
      //   - A directory someone created in a local action (in this case, changes under the
      //     directory may not be detected since we use the mtime of the directory for
      //     up-to-dateness checks)
      //   - A symlink to a source directory due to Filesets
      return FileArtifactValue.createForDirectoryWithMtime(
          artifactPathResolver.toPath(artifact).getLastModifiedTime());
    }

    if (injectedDigest == null && type.isFile()) {
      injectedDigest =
          DigestUtils.getDigestOrFail(artifactPathResolver.toPath(artifact), value.getSize());
    }
    return FileArtifactValue.createFromInjectedDigest(
        value, injectedDigest, !artifact.isConstantMetadata());
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
            TreeFileArtifact.createTreeOutput(parent, subpath.getChild(dirent.getName())));
      }
    }
  }
}
