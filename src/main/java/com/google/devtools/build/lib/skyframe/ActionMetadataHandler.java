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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.util.concurrent.TimeUnit.MINUTES;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Handler provided by {@link ActionExecutionFunction} which allows the execution engine to obtain
 * {@linkplain FileArtifactValue metadata} about inputs and outputs and to store metadata about an
 * action's outputs for purposes of creating the final {@link ActionExecutionValue}.
 *
 * <p>The handler can be in one of two modes. After construction, it acts as a cache for input and
 * output metadata while {@link ActionCacheChecker} determines whether the action needs to be
 * executed. If the action needs to be executed (i.e. no action cache hit), {@link
 * #prepareForActionExecution} is called. This call switches the handler to a mode where it accepts
 * {@linkplain MetadataInjector injected output data}, or otherwise obtains metadata from the
 * filesystem. Freshly created output files are set read-only and executable <em>before</em>
 * statting them to ensure that the stat's ctime is up to date.
 *
 * <p>After action execution, {@link #getMetadata} should be called on each of the action's outputs
 * (except those that were {@linkplain #artifactOmitted omitted}) to ensure that declared outputs
 * were in fact created and are valid.
 */
final class ActionMetadataHandler implements MetadataHandler {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Creates a new metadata handler.
   *
   * <p>If the handler is for use during input discovery, calling {@link #getMetadata} with an
   * artifact which is neither in {@code inputArtifactData} nor {@code outputs} is tolerated and
   * will return {@code null}. To subsequently transform the handler for regular action execution
   * (where such a call is not permitted), use {@link #transformAfterInputDiscovery}.
   */
  static ActionMetadataHandler create(
      ActionInputMap inputArtifactData,
      boolean forInputDiscovery,
      boolean archivedTreeArtifactsEnabled,
      ImmutableSet<Artifact> outputs,
      TimestampGranularityMonitor tsgm,
      ArtifactPathResolver artifactPathResolver,
      PathFragment execRoot,
      PathFragment derivedPathPrefix,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets) {
    return new ActionMetadataHandler(
        inputArtifactData,
        forInputDiscovery,
        archivedTreeArtifactsEnabled,
        outputs,
        tsgm,
        artifactPathResolver,
        execRoot,
        derivedPathPrefix,
        createFilesetMapping(expandedFilesets, execRoot),
        new OutputStore());
  }

  private final ActionInputMap inputArtifactData;
  private final boolean forInputDiscovery;
  private final boolean archivedTreeArtifactsEnabled;
  private final ImmutableMap<PathFragment, FileArtifactValue> filesetMapping;

  private final Set<Artifact> omittedOutputs = Sets.newConcurrentHashSet();
  private final ImmutableSet<Artifact> outputs;

  private final TimestampGranularityMonitor tsgm;
  private final ArtifactPathResolver artifactPathResolver;
  private final PathFragment execRoot;
  private final PathFragment derivedPathPrefix;

  private final AtomicBoolean executionMode = new AtomicBoolean(false);
  private final OutputStore store;

  private ActionMetadataHandler(
      ActionInputMap inputArtifactData,
      boolean forInputDiscovery,
      boolean archivedTreeArtifactsEnabled,
      ImmutableSet<Artifact> outputs,
      TimestampGranularityMonitor tsgm,
      ArtifactPathResolver artifactPathResolver,
      PathFragment execRoot,
      PathFragment derivedPathPrefix,
      ImmutableMap<PathFragment, FileArtifactValue> filesetMapping,
      OutputStore store) {
    this.inputArtifactData = checkNotNull(inputArtifactData);
    this.forInputDiscovery = forInputDiscovery;
    this.archivedTreeArtifactsEnabled = archivedTreeArtifactsEnabled;
    this.outputs = checkNotNull(outputs);
    this.tsgm = checkNotNull(tsgm);
    this.artifactPathResolver = checkNotNull(artifactPathResolver);
    this.execRoot = checkNotNull(execRoot);
    this.derivedPathPrefix = checkNotNull(derivedPathPrefix);
    this.filesetMapping = checkNotNull(filesetMapping);
    this.store = checkNotNull(store);
  }

  /**
   * Returns a new handler mostly identical to this one, except uses the given {@code store} and
   * does not permit {@link #getMetadata} to be called with an artifact which is neither in inputs
   * nor outputs.
   *
   * <p>The returned handler will be in the mode for action cache checking. To prepare it for action
   * execution, call {@link #prepareForActionExecution}.
   *
   * <p>This method is designed to be called after input discovery when a fresh handler is needed
   * but all of the parameters in {@link #create} would be the same as the original handler.
   */
  ActionMetadataHandler transformAfterInputDiscovery(OutputStore store) {
    return new ActionMetadataHandler(
        inputArtifactData,
        /*forInputDiscovery=*/ false,
        archivedTreeArtifactsEnabled,
        outputs,
        tsgm,
        artifactPathResolver,
        execRoot,
        derivedPathPrefix,
        filesetMapping,
        store);
  }

  /**
   * If {@code value} represents an existing file, returns it as is, otherwise throws {@link
   * FileNotFoundException}.
   */
  private static FileArtifactValue checkExists(FileArtifactValue value, Artifact artifact)
      throws FileNotFoundException {
    if (FileArtifactValue.MISSING_FILE_MARKER.equals(value)
        || FileArtifactValue.OMITTED_FILE_MARKER.equals(value)) {
      throw new FileNotFoundException(artifact + " does not exist");
    }
    return checkNotNull(value, artifact);
  }

  /**
   * If {@code value} represents an existing tree artifact, returns it as is, otherwise throws
   * {@link FileNotFoundException}.
   */
  private static TreeArtifactValue checkExists(TreeArtifactValue value, Artifact artifact)
      throws FileNotFoundException {
    if (TreeArtifactValue.MISSING_TREE_ARTIFACT.equals(value)
        || TreeArtifactValue.OMITTED_TREE_MARKER.equals(value)) {
      throw new FileNotFoundException(artifact + " does not exist");
    }
    return checkNotNull(value, artifact);
  }

  private static ImmutableMap<PathFragment, FileArtifactValue> createFilesetMapping(
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets, PathFragment execRoot) {
    Map<PathFragment, FileArtifactValue> filesetMap = new HashMap<>();
    for (Map.Entry<Artifact, ImmutableList<FilesetOutputSymlink>> entry : filesets.entrySet()) {
      try {
        FilesetManifest fileset =
            FilesetManifest.constructFilesetManifest(
                entry.getValue(), execRoot, RelativeSymlinkBehavior.RESOLVE);
        for (Map.Entry<String, FileArtifactValue> favEntry :
            fileset.getArtifactValues().entrySet()) {
          if (favEntry.getValue().getDigest() != null) {
            filesetMap.put(PathFragment.create(favEntry.getKey()), favEntry.getValue());
          }
        }
      } catch (IOException e) {
        // If we cannot get the FileArtifactValue, then we will make a FileSystem call to get the
        // digest, so it is okay to skip and continue here.
        logger.atWarning().withCause(e).log(
            "Could not properly get digest for %s", entry.getKey().getExecPath());
      }
    }
    return ImmutableMap.copyOf(filesetMap);
  }

  private boolean isKnownOutput(Artifact artifact) {
    return outputs.contains(artifact)
        || (artifact.hasParent() && outputs.contains(artifact.getParent()));
  }

  @Override
  @Nullable
  public FileArtifactValue getMetadata(ActionInput actionInput) throws IOException {
    if (!(actionInput instanceof Artifact)) {
      PathFragment inputPath = actionInput.getExecPath();
      PathFragment filesetKeyPath =
          inputPath.startsWith(execRoot) ? inputPath.relativeTo(execRoot) : inputPath;
      return filesetMapping.get(filesetKeyPath);
    }

    Artifact artifact = (Artifact) actionInput;
    FileArtifactValue value;

    if (!isKnownOutput(artifact)) {
      value = inputArtifactData.getMetadata(artifact);
      if (value != null) {
        return checkExists(value, artifact);
      }
      checkState(forInputDiscovery, "%s is not present in declared outputs: %s", artifact, outputs);
      return null;
    }

    if (artifact.isMiddlemanArtifact()) {
      // A middleman artifact's data was either already injected from the action cache checker using
      // #setDigestForVirtualArtifact, or it has the default middleman value.
      value = store.getArtifactData(artifact);
      if (value != null) {
        return checkExists(value, artifact);
      }
      store.putArtifactData(artifact, FileArtifactValue.DEFAULT_MIDDLEMAN);
      return FileArtifactValue.DEFAULT_MIDDLEMAN;
    }

    if (artifact.isTreeArtifact()) {
      TreeArtifactValue tree = getTreeArtifactValue((SpecialArtifact) artifact);
      return tree.getMetadata();
    }

    if (artifact.isChildOfDeclaredDirectory()) {
      TreeArtifactValue tree = getTreeArtifactValue(artifact.getParent());
      value = tree.getChildValues().getOrDefault(artifact, FileArtifactValue.MISSING_FILE_MARKER);
      return checkExists(value, artifact);
    }

    value = store.getArtifactData(artifact);
    if (value != null) {
      return checkExists(value, artifact);
    }

    // No existing metadata; this can happen if the output metadata is not injected after a spawn
    // is executed. SkyframeActionExecutor.checkOutputs calls this method for every output file of
    // the action, which hits this code path. Another possibility is that an action runs multiple
    // spawns, and a subsequent spawn requests the metadata of an output of a previous spawn.

    // If necessary, we first call chmod the output file. The FileArtifactValue may use a
    // FileContentsProxy, which is based on ctime (affected by chmod).
    if (executionMode.get()) {
      setPathReadOnlyAndExecutableIfFile(artifactPathResolver.toPath(artifact));
    }

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
    checkArgument(artifact.isMiddlemanArtifact(), artifact);
    checkNotNull(digest, artifact);
    store.putArtifactData(artifact, FileArtifactValue.createProxy(digest));
  }

  @Override
  public TreeArtifactValue getTreeArtifactValue(SpecialArtifact artifact) throws IOException {
    checkState(artifact.isTreeArtifact(), "%s is not a tree artifact", artifact);

    TreeArtifactValue value = store.getTreeArtifactData(artifact);
    if (value != null) {
      return checkExists(value, artifact);
    }

    value = constructTreeArtifactValueFromFilesystem(artifact);
    store.putTreeArtifactData(artifact, value);
    return checkExists(value, artifact);
  }

  private TreeArtifactValue constructTreeArtifactValueFromFilesystem(SpecialArtifact parent)
      throws IOException {
    Path treeDir = artifactPathResolver.toPath(parent);
    boolean chmod = executionMode.get();

    // Make sure the tree artifact root is a regular directory. Note that this is how the action is
    // initialized, so this should hold unless the action itself has deleted the root.
    if (!treeDir.isDirectory(Symlinks.NOFOLLOW)) {
      if (chmod) {
        setPathReadOnlyAndExecutableIfFile(treeDir);
      }
      return TreeArtifactValue.MISSING_TREE_ARTIFACT;
    }

    if (chmod) {
      setPathReadOnlyAndExecutable(treeDir);
    }

    TreeArtifactValue.Builder tree = TreeArtifactValue.newBuilder(parent);

    TreeArtifactValue.visitTree(
        treeDir,
        (parentRelativePath, type) -> {
          if (chmod && type != Dirent.Type.SYMLINK) {
            setPathReadOnlyAndExecutable(treeDir.getRelative(parentRelativePath));
          }
          if (type == Dirent.Type.DIRECTORY) {
            return; // The final TreeArtifactValue does not contain child directories.
          }
          TreeFileArtifact child = TreeFileArtifact.createTreeOutput(parent, parentRelativePath);
          FileArtifactValue metadata;
          try {
            metadata = constructFileArtifactValueFromFilesystem(child);
          } catch (FileNotFoundException e) {
            String errorMessage =
                String.format(
                    "Failed to resolve relative path %s inside TreeArtifact %s. "
                        + "The associated file is either missing or is an invalid symlink.",
                    parentRelativePath, treeDir);
            throw new IOException(errorMessage, e);
          }

          tree.putChild(child, metadata);
        });

    if (archivedTreeArtifactsEnabled) {
      ArchivedTreeArtifact archivedTreeArtifact =
          ArchivedTreeArtifact.create(parent, derivedPathPrefix);
      FileStatus statNoFollow =
          artifactPathResolver.toPath(archivedTreeArtifact).statIfFound(Symlinks.NOFOLLOW);
      if (statNoFollow != null) {
        tree.setArchivedRepresentation(
            archivedTreeArtifact,
            constructFileArtifactValue(
                archivedTreeArtifact,
                FileStatusWithDigestAdapter.adapt(statNoFollow),
                /*injectedDigest=*/ null));
      } else {
        logger.atInfo().atMostEvery(5, MINUTES).log(
            "Archived tree artifact: %s not created", archivedTreeArtifact);
      }
    }

    return tree.build();
  }

  @Override
  public ImmutableSet<TreeFileArtifact> getTreeArtifactChildren(SpecialArtifact treeArtifact) {
    checkArgument(treeArtifact.isTreeArtifact(), "%s is not a tree artifact", treeArtifact);
    TreeArtifactValue tree = store.getTreeArtifactData(treeArtifact);
    return tree != null ? tree.getChildren() : ImmutableSet.of();
  }

  @Override
  public FileArtifactValue constructMetadataForDigest(
      Artifact output, FileStatus statNoFollow, byte[] digest) throws IOException {
    checkArgument(!output.isSymlink(), "%s is a symlink", output);
    checkNotNull(digest, "Missing digest for %s", output);
    checkNotNull(statNoFollow, "Missing stat for %s", output);
    checkState(
        executionMode.get(), "Tried to construct metadata for %s outside of execution", output);

    // We already have a stat, so no need to call chmod.
    return constructFileArtifactValue(
        output, FileStatusWithDigestAdapter.adapt(statNoFollow), digest);
  }

  @Override
  public void injectFile(Artifact output, FileArtifactValue metadata) {
    checkArgument(isKnownOutput(output), "%s is not a declared output of this action", output);
    checkArgument(
        !output.isTreeArtifact() && !output.isChildOfDeclaredDirectory(),
        "Tree artifacts and their children must be injected via injectTree: %s",
        output);
    checkState(executionMode.get(), "Tried to inject %s outside of execution", output);

    store.putArtifactData(output, metadata);
  }

  @Override
  public void injectTree(SpecialArtifact output, TreeArtifactValue tree) {
    checkArgument(isKnownOutput(output), "%s is not a declared output of this action", output);
    checkArgument(output.isTreeArtifact(), "Output must be a tree artifact: %s", output);
    checkState(executionMode.get(), "Tried to inject %s outside of execution", output);
    checkArgument(
        archivedTreeArtifactsEnabled == tree.getArchivedRepresentation().isPresent(),
        "Archived representation presence mismatched for: %s with archivedTreeArtifactsEnabled: %s",
        tree,
        archivedTreeArtifactsEnabled);

    store.putTreeArtifactData(output, tree);
  }

  @Override
  public void markOmitted(Artifact output) {
    checkState(executionMode.get(), "Tried to mark %s omitted outside of execution", output);
    boolean newlyOmitted = omittedOutputs.add(output);
    if (output.isTreeArtifact()) {
      // Tolerate marking a tree artifact as omitted multiple times so that callers don't have to
      // deduplicate when a tree artifact has several omitted children.
      if (newlyOmitted) {
        store.putTreeArtifactData((SpecialArtifact) output, TreeArtifactValue.OMITTED_TREE_MARKER);
      }
    } else {
      checkState(newlyOmitted, "%s marked as omitted twice", output);
      store.putArtifactData(output, FileArtifactValue.OMITTED_FILE_MARKER);
    }
  }

  @Override
  public boolean artifactOmitted(Artifact artifact) {
    return omittedOutputs.contains(artifact);
  }

  @Override
  public void resetOutputs(Iterable<? extends Artifact> outputs) {
    checkState(
        executionMode.get(), "resetOutputs() should only be called from within a running action.");
    for (Artifact output : outputs) {
      omittedOutputs.remove(output);
      store.remove(output);
    }
  }

  /**
   * Informs this handler that the action is about to be executed.
   *
   * <p>Any stale metadata cached in the underlying {@link OutputStore} from action cache checking
   * is cleared.
   */
  void prepareForActionExecution() {
    checkState(!executionMode.getAndSet(true), "Already in execution mode");
    store.clear();
  }

  /**
   * Returns the underlying {@link OutputStore} containing metadata cached during the lifetime of
   * this handler.
   *
   * <p>The store may be passed to {@link ActionExecutionValue#createFromOutputStore}.
   */
  OutputStore getOutputStore() {
    return store;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("outputs", outputs)
        .add("store", store)
        .add("inputArtifactDataSize", inputArtifactData.size())
        .toString();
  }

  /** Constructs a new {@link FileArtifactValue} by reading from the file system. */
  private FileArtifactValue constructFileArtifactValueFromFilesystem(Artifact artifact)
      throws IOException {
    return constructFileArtifactValue(artifact, /*statNoFollow=*/ null, /*injectedDigest=*/ null);
  }

  /** Constructs a new {@link FileArtifactValue}, optionally taking a known stat and digest. */
  private FileArtifactValue constructFileArtifactValue(
      Artifact artifact,
      @Nullable FileStatusWithDigest statNoFollow,
      @Nullable byte[] injectedDigest)
      throws IOException {
    checkState(!artifact.isTreeArtifact(), "%s is a tree artifact", artifact);

    FileArtifactValue value =
        fileArtifactValueFromArtifact(
            artifact,
            artifactPathResolver,
            statNoFollow,
            injectedDigest != null,
            // Prevent constant metadata artifacts from notifying the timestamp granularity monitor
            // and potentially delaying the build for no reason.
            artifact.isConstantMetadata() ? null : tsgm);

    // Ensure that we don't have both an injected digest and a digest from the filesystem.
    byte[] fileDigest = value.getDigest();
    if (fileDigest != null && injectedDigest != null) {
      throw new IllegalStateException(
          String.format(
              "Digest %s was injected for artifact %s, but got %s from the filesystem (%s)",
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
      // We always create a FileArtifactValue for an unresolved symlink with a digest (calling
      // readlink() is easy, unlike checksumming a potentially huge file).
      checkNotNull(fileDigest, "%s missing digest", value);
      return value;
    }

    if (type.isFile() && fileDigest != null) {
      // The digest is in the file value and that is all that is needed for this file's metadata.
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
      // We don't have an injected digest and there is no digest in the file value (which attempts a
      // fast digest). Manually compute the digest instead.
      injectedDigest =
          DigestUtils.manuallyComputeDigest(artifactPathResolver.toPath(artifact), value.getSize());
    }
    return FileArtifactValue.createFromInjectedDigest(
        value, injectedDigest, /*isShareable=*/ !artifact.isConstantMetadata());
  }

  /**
   * Constructs a {@link FileArtifactValue} for a regular (non-tree, non-middleman) artifact for the
   * purpose of determining whether an existing {@link FileArtifactValue} is still valid.
   *
   * <p>The returned metadata may be compared with metadata present in an {@link
   * ActionExecutionValue} using {@link FileArtifactValue#couldBeModifiedSince} to check for
   * inter-build modifications.
   */
  static FileArtifactValue fileArtifactValueFromArtifact(
      Artifact artifact,
      @Nullable FileStatusWithDigest statNoFollow,
      @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    return fileArtifactValueFromArtifact(
        artifact,
        ArtifactPathResolver.IDENTITY,
        statNoFollow,
        /*digestWillBeInjected=*/ false,
        tsgm);
  }

  private static FileArtifactValue fileArtifactValueFromArtifact(
      Artifact artifact,
      ArtifactPathResolver artifactPathResolver,
      @Nullable FileStatusWithDigest statNoFollow,
      boolean digestWillBeInjected,
      @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    checkState(!artifact.isTreeArtifact() && !artifact.isMiddlemanArtifact(), artifact);

    Path pathNoFollow = artifactPathResolver.toPath(artifact);
    RootedPath rootedPathNoFollow =
        RootedPath.toRootedPath(
            artifactPathResolver.transformRoot(artifact.getRoot().getRoot()),
            artifact.getRootRelativePath());
    if (statNoFollow == null) {
      // Stat the file. All output artifacts of an action are deleted before execution, so if a file
      // exists, it was most likely created by the current action. There is a race condition here if
      // an external process creates (or modifies) the file between the deletion and this stat,
      // which we cannot solve.
      statNoFollow = FileStatusWithDigestAdapter.adapt(pathNoFollow.statIfFound(Symlinks.NOFOLLOW));
    }

    if (statNoFollow == null || !statNoFollow.isSymbolicLink()) {
      return fileArtifactValueFromStat(
          rootedPathNoFollow,
          statNoFollow,
          digestWillBeInjected,
          artifact.isConstantMetadata(),
          tsgm);
    }

    if (artifact.isSymlink()) {
      return FileArtifactValue.createForUnresolvedSymlink(pathNoFollow);
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
        realRootedPath,
        realStatWithDigest,
        digestWillBeInjected,
        artifact.isConstantMetadata(),
        tsgm);
  }

  private static FileArtifactValue fileArtifactValueFromStat(
      RootedPath rootedPath,
      FileStatusWithDigest stat,
      boolean digestWillBeInjected,
      boolean isConstantMetadata,
      @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    if (stat == null) {
      return FileArtifactValue.MISSING_FILE_MARKER;
    }

    FileStateValue fileStateValue =
        FileStateValue.createWithStatNoFollow(rootedPath, stat, digestWillBeInjected, tsgm);

    return stat.isDirectory()
        ? FileArtifactValue.createForDirectoryWithMtime(stat.getLastModifiedTime())
        : FileArtifactValue.createForNormalFile(
            fileStateValue.getDigest(),
            fileStateValue.getContentsProxy(),
            stat.getSize(),
            /*isShareable=*/ !isConstantMetadata);
  }

  private static void setPathReadOnlyAndExecutableIfFile(Path path) throws IOException {
    if (path.isFile(Symlinks.NOFOLLOW)) {
      setPathReadOnlyAndExecutable(path);
    }
  }

  private static void setPathReadOnlyAndExecutable(Path path) throws IOException {
    path.chmod(0555);
  }
}
