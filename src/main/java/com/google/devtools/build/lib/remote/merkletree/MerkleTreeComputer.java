package com.google.devtools.build.lib.remote.merkletree;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Predicates.alwaysFalse;
import static com.google.common.base.Predicates.alwaysTrue;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.vfs.PathFragment.HIERARCHICAL_COMPARATOR;
import static java.util.Comparator.comparing;
import static java.util.Map.entry;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;
import static java.util.concurrent.CompletableFuture.supplyAsync;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.NodeProperties;
import build.bazel.remote.execution.v2.NodeProperty;
import com.github.benmanes.caffeine.cache.AsyncCache;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.Scrubber;
import com.google.devtools.build.lib.remote.Scrubber.SpawnScrubber;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext.CachePolicy;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.AbstractCollection;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Computes a Merkle tree representing the inputs of a {@link Spawn} in a {@link
 * Action#getInputRootDigest()}.
 *
 * <p>Remote execution should allow a developer to run up to thousands of actions remotely in
 * parallel on a regular machine. As a result, this class is optimized with the following goals in
 * the order of decreasing importance:
 *
 * <ul>
 *   <li>Above all else, keep peak memory usage as low as possible so that Bazel doesn't OOM when
 *       running many remote execution actions in parallel. Allocations with sizes linear in the
 *       number of spawn inputs should be avoided if possible and kept as small as possible.
 *   <li>Make incremental builds as fast as possible.
 *   <li>Keep the size of caches kept between builds as small as possible.
 */
public final class MerkleTreeComputer {
  // This class achieves its goals via the following observations:
  //
  // * Incremental builds typically have a large number of cache hits and the only information about
  //   a Merkle tree needed to check for a cache hit is the root digest. We can thus avoid
  //   materializing the full tree unless the cache check fails.
  // * Certain special artifacts are known to form self-contained Merkle trees that never intersect
  //   with any other Merkle tree. This includes tree artifacts, runfiles directories and source
  //   directories. The Merkle trees for such artifacts can be computed and uploaded to the remote
  //   cache independently. Only their root digest has to be kept around for inclusion in other
  //   Merkle trees.
  // * FileArtifactValue's fully describe their contents and can thus be used as cache keys for
  //   inter-build caches. By using a weak reference for the key, they can be cleaned up
  //   automatically as soon as their contents change or they are no longer relevant.
  // * Instead of basing caching decisions on a particular remote cache TTL, we can optimistically
  //   assume that every blob that has been uploaded is still in the cache if we support a mode in
  //   which all blobs can be forcibly recomputed and re-uploaded on missing digests.
  // * While the inputs of a spawn naturally form a map of paths to contents, this map doesn't have
  //   to be materialized in memory. Instead, it suffices to maintain a list of inputs that is
  //   sorted by lazily computed paths. This drastically reduces peak memory usage.
  // * When visiting the inputs of a spawn in hierarchical order, once a directory is left once,
  //   it will never be entered again. At that point, the proto describing it can be built and
  //   digested and intermediate structures are no longer needed.

  private static final NodeProperties TOOL_NODE_PROPERTIES =
      NodeProperties.newBuilder()
          .addProperties(NodeProperty.newBuilder().setName("bazel_tool_input"))
          .build();
  private static final List<Map.Entry<PathFragment, ActionInput>> END_OF_INPUTS_SENTINEL =
      ImmutableList.of(entry(PathFragment.EMPTY_FRAGMENT, VirtualActionInput.EMPTY_MARKER));
  private static final PathFragment ROOT_FAKE_PATH_SEGMENT = PathFragment.create("root");

  // Building Merkle trees mostly involves computing hashes of protos and is thus CPU-bound.
  // TODO: Source directories are also visited on this pool in a single-threaded manner.
  private static final ExecutorService MERKLE_TREE_BUILD_POOL =
      Executors.newFixedThreadPool(
          Runtime.getRuntime().availableProcessors(),
          Thread.ofPlatform().name("merkle-tree-build-", 0).factory());
  // Uploading Merkle trees mostly involves waiting on networking futures, for which virtual threads
  // are ideal.
  private static final ExecutorService MERKLE_TREE_UPLOAD_POOL =
      Executors.newThreadPerTaskExecutor(
          Thread.ofVirtual().name("merkle-tree-upload-", 0).factory());

  private static final Cache<FileArtifactValue, MerkleTree.RootOnly> persistentToolSubTreeCache =
      Caffeine.newBuilder().weakKeys().build();
  private static final Cache<FileArtifactValue, MerkleTree.RootOnly> persistentNonToolSubTreeCache =
      Caffeine.newBuilder().weakKeys().build();

  @Nullable private static volatile Scrubber lastScrubber;

  private final DigestUtil digestUtil;
  @Nullable private final MerkleTreeUploader remoteExecutionCache;
  private final String buildRequestId;
  private final String commandId;
  private final Digest emptyDigest;
  private final MerkleTree.Uploadable emptyTree;
  private final AsyncCache<InFlightCacheKey, MerkleTree.RootOnly> inFlightSubTreeCache =
      Caffeine.newBuilder().buildAsync();

  public MerkleTreeComputer(
      DigestUtil digestUtil,
      @Nullable MerkleTreeUploader remoteExecutionCache,
      String buildRequestId,
      String commandId) {
    this.digestUtil = digestUtil;
    this.remoteExecutionCache = remoteExecutionCache;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    var emptyBlob = new byte[0];
    this.emptyDigest = digestUtil.compute(emptyBlob);
    this.emptyTree =
        new MerkleTree.Uploadable(
            new MerkleTree.RootOnly.BlobsUploaded(emptyDigest, 0, 0),
            ImmutableMap.of(emptyDigest, emptyBlob));
  }

  /**
   * A representation of the inputs to a remotely executed action represented as a Merkle tree.
   *
   * <p>Every tree has a digest, which is the digest of the tree's root directory. The subtrees and
   * the blobs they contain may have been discarded or never computed in the first place, for
   * example, because they have already been uploaded to the remote cache or because the tree is
   * being built only to check for a remote cache hit.
   */
  public sealed interface MerkleTree {
    /** The digest of the tree's root directory. */
    Digest digest();

    /** The total number of regular files and symlinks in this tree, including all subtrees. */
    long inputFiles();

    /**
     * The total number of content bytes in this tree, including all subtrees. This includes both
     * file contents and the protos describing directories.
     */
    long inputBytes();

    /** Returns the root of this tree, which may be the current instance. */
    RootOnly root();

    /**
     * A {@link MerkleTree} that doesn't retain any blobs, either because they have already been
     * uploaded or because only the root digest is needed (e.g., for a remote cache check).
     */
    sealed interface RootOnly extends MerkleTree {
      @Override
      default RootOnly root() {
        return this;
      }

      /**
       * A {@link MerkleTree} that retains no blobs since all of them have recently been uploaded to
       * the remote cache.
       */
      record BlobsUploaded(Digest digest, long inputFiles, long inputBytes) implements RootOnly {}

      /**
       * A {@link MerkleTree} that retains no blobs since they were discarded during the computation
       * (e.g., because they aren't needed for a remote cache check).
       */
      record BlobsDiscarded(Digest digest, long inputFiles, long inputBytes) implements RootOnly {}
    }

    /** A {@link MerkleTree} that retains all blobs that still need to be uploaded. */
    final class Uploadable implements MerkleTree {
      private final RootOnly.BlobsUploaded root;
      private final ImmutableMap<Digest, /* byte[] | Path | VirtualActionInput */ Object> blobs;

      private Uploadable(RootOnly.BlobsUploaded root, ImmutableMap<Digest, Object> blobs) {
        this.root = root;
        this.blobs = blobs;
      }

      public Digest digest() {
        return root().digest();
      }

      public long inputFiles() {
        return root().inputFiles();
      }

      public long inputBytes() {
        return root().inputBytes();
      }

      public ImmutableSet<Digest> allDigests() {
        return blobs.keySet();
      }

      @VisibleForTesting
      public ImmutableMap<Digest, Object> blobs() {
        return blobs;
      }

      @Override
      public RootOnly root() {
        return root;
      }

      /**
       * Returns a future that tracks the upload of the blob with the given digest, or {@link
       * Optional#empty()} if there is no blob with the given digest.
       */
      public Optional<ListenableFuture<Void>> upload(
          MerkleTreeUploader uploader,
          RemoteActionExecutionContext context,
          RemotePathResolver remotePathResolver,
          Digest digest) {
        return switch (blobs.get(digest)) {
          case byte[] data -> Optional.of(uploader.upload(context, digest, data));
          case Path path -> Optional.of(uploader.upload(context, remotePathResolver, digest, path));
          case VirtualActionInput virtualActionInput ->
              Optional.of(uploader.upload(context, digest, virtualActionInput));
          case null -> Optional.empty();
          default -> throw new IllegalStateException("Unexpected blob type: " + blobs.get(digest));
        };
      }
    }
  }

  /** The basic cache operations needed to upload a {@link MerkleTree} and its associated blobs. */
  public interface MerkleTreeUploader {
    ListenableFuture<Void> upload(RemoteActionExecutionContext context, Digest digest, byte[] data);

    ListenableFuture<Void> upload(
        RemoteActionExecutionContext context,
        RemotePathResolver remotePathResolver,
        Digest digest,
        Path path);

    ListenableFuture<Void> upload(
        RemoteActionExecutionContext context, Digest digest, VirtualActionInput virtualActionInput);

    /**
     * Ensures that all inputs as well as metadata protos in the given Merkle tree are present in
     * the remote cache by querying for and uploading missing blobs.
     *
     * @param force if true, all blobs in the tree are uploaded even if they have already been
     *     uploaded before.
     */
    void ensureInputsPresent(
        RemoteActionExecutionContext context,
        MerkleTree.Uploadable merkleTree,
        boolean force,
        RemotePathResolver remotePathResolver)
        throws IOException, InterruptedException;
  }

  /** Specifies which blobs should be retained in the Merkle tree. */
  public enum BlobPolicy {
    /**
     * No blobs are retained and the returned MerkleTree is a {@link MerkleTree.RootOnly}.
     *
     * <p>This is the most lightweight policy. It always suffices when checking for a remote cache
     * hit and may suffice for remote execution when a cache hit is expected.
     */
    DISCARD,

    /**
     * Retains all blobs in the tree that aren't contained in subtrees that have already been
     * uploaded.
     *
     * <p>Only blobs that have been uploaded to the remote cache during the lifetime of the Bazel
     * server are omitted from this tree. This usually suffices for remote execution unless the
     * remote cache loses entries.
     */
    KEEP,

    /**
     * Retains all blobs in the tree and also forces the reupload of all subtrees.
     *
     * <p>This is only needed in exceptional cases, such as when the remote cache has lost entries
     * while the Bazel server is running.
     */
    KEEP_AND_REUPLOAD,
  }

  /**
   * The key type for the cache used to deduplicate ongoing computations and possibly uploading of
   * sub-Merkle trees.
   *
   * @param metadata the metadata of the aggregate {@link ActionInput} that forms the subtree
   * @param isTool whether the subtree consists of tool inputs
   * @param uploadBlobs whether the blobs in this tree will be uploaded
   */
  private record InFlightCacheKey(
      FileArtifactValue metadata, boolean isTool, boolean uploadBlobs) {}

  /**
   * Builds a Merkle tree for the inputs of a {@link Spawn}.
   *
   * @param toolInputs the set of paths of inputs that are considered tools. Note that these paths
   *     are not exec paths, but those returned as keys by {@link
   *     com.google.devtools.build.lib.exec.SpawnInputExpander#getInputMapping}, i.e., they have
   *     already been subject to path mapping and runfiles tree as well as tree artifact expansion.
   *     Callers have to ensure that paths within an aggregate artifact are either all tools or all
   *     non-tools.
   * @param scrubber the invocation-global scrubber, or null if no scrubbing should be performed
   * @param blobPolicy used to decide which blobs should be retained in the returned Merkle tree. If
   *     {@code KEEP_AND_REUPLOAD} is used, all blobs in the tree are retained and the resulting
   *     Merkle tree will be a {@link MerkleTree.Uploadable}.
   * @throws LostInputsExecException if inputs to this spawn that are remote-only have been
   *     discovered to be missing from the remote cache. Action or build rewinding may be able to
   *     recover from this.
   */
  public MerkleTree buildForSpawn(
      Spawn spawn,
      Set<PathFragment> toolInputs,
      @Nullable Scrubber scrubber,
      SpawnExecutionContext spawnExecutionContext,
      RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy)
      throws IOException, InterruptedException, LostInputsExecException {
    // The scrubber is a per-invocation setting and invocations do not overlap, so it can be tracked
    // in a static variable.
    if (!Objects.equals(scrubber, lastScrubber)) {
      synchronized (MerkleTreeComputer.class) {
        if (!Objects.equals(scrubber, lastScrubber)) {
          persistentToolSubTreeCache.invalidateAll();
          persistentNonToolSubTreeCache.invalidateAll();
          inFlightSubTreeCache.synchronous().invalidateAll();
          lastScrubber = scrubber;
        }
      }
    }
    var spawnInputs = spawn.getInputFiles().toList();
    // Add output directories to inputs so that they are created as empty directories by the
    // executor. The spec only requires the executor to create the parent directory of an output
    // directory, which differs from the behavior of both local and sandboxed execution.
    var outputDirectories =
        spawn.getOutputFiles().stream()
            .filter(output -> output instanceof Artifact artifact && artifact.isTreeArtifact())
            .map(outputDir -> new EmptyInputDirectory(outputDir.getExecPath()))
            .collect(toImmutableList());
    // Reduce peak memory usage by avoiding the allocation of intermediate arrays and sorted map, as
    // well as the prolonged retention of mapped paths. All of these can be reconstructed on-the-fly
    // while iterating over the inputs, only the sorted order has to be retained.
    var allInputs =
        ImmutableList.sortedCopyOf(
            comparing(
                input -> getOutputPath(input, remotePathResolver, spawn.getPathMapper()),
                HIERARCHICAL_COMPARATOR),
            concat(spawnInputs, outputDirectories));
    var metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, "subtree", spawn.getResourceOwner());
    var remoteActionExecutionContext =
        RemoteActionExecutionContext.create(
            spawn,
            spawnExecutionContext,
            metadata,
            CachePolicy.REMOTE_CACHE_ONLY,
            CachePolicy.NO_CACHE);
    Predicate<PathFragment> isToolInput;
    if (toolInputs.isEmpty() || remotePathResolver.getWorkingDirectory().isEmpty()) {
      isToolInput = toolInputs::contains;
    } else {
      isToolInput =
          path -> toolInputs.contains(path.relativeTo(remotePathResolver.getWorkingDirectory()));
    }
    try {
      return build(
          Lists.transform(
              allInputs,
              input ->
                  entry(getOutputPath(input, remotePathResolver, spawn.getPathMapper()), input)),
          isToolInput,
          scrubber != null ? scrubber.forSpawn(spawn) : null,
          spawnExecutionContext.getInputMetadataProvider(),
          spawnExecutionContext.getPathResolver(),
          remoteActionExecutionContext,
          remotePathResolver,
          blobPolicy);
    } catch (BulkTransferException e) {
      e.getLostArtifacts(spawnExecutionContext.getInputMetadataProvider()::getInput)
          .throwIfNotEmpty();
      throw e;
    }
  }

  private static PathFragment getOutputPath(
      ActionInput input, RemotePathResolver remotePathResolver, PathMapper pathMapper) {
    return remotePathResolver
        .getWorkingDirectory()
        .getRelative(pathMapper.map(input.getExecPath()));
  }

  /** An {@link ActionInput} backed by an absolute {@link Path}. */
  private static class ActionInputWithPath extends ActionInputHelper.BasicActionInput {
    final Path path;

    ActionInputWithPath(Path path) {
      this.path = path;
    }

    @Override
    public String getExecPathString() {
      return path.getPathString();
    }

    @Override
    public PathFragment getExecPath() {
      return path.asFragment();
    }
  }

  private static final ArtifactPathResolver actionInputWithPathResolver =
      new ArtifactPathResolver() {
        @Override
        public Path toPath(ActionInput actionInput) {
          return ((ActionInputWithPath) actionInput).path;
        }

        @Override
        public Path convertPath(Path path) {
          throw new UnsupportedOperationException();
        }

        @Override
        public Root transformRoot(Root root) {
          throw new UnsupportedOperationException();
        }
      };

  /**
   * Builds a Merkle tree for a set of files and their logical paths.
   *
   * <p>The only use outside testing is by repository rules. Use {@link #buildForSpawn} for
   * everything else.
   */
  public MerkleTree.Uploadable buildForFiles(Map<PathFragment, Path> inputs)
      throws IOException, InterruptedException {
    // BlobPolicy.KEEP_AND_REUPLOAD always results in a MerkleTree.Uploadable.
    return (MerkleTree.Uploadable)
        build(
            Lists.transform(
                ImmutableList.sortedCopyOf(
                    Map.Entry.comparingByKey(HIERARCHICAL_COMPARATOR), inputs.entrySet()),
                e -> entry(e.getKey(), new ActionInputWithPath(e.getValue()))),
            alwaysFalse(),
            /* spawnScrubber= */ null,
            StaticInputMetadataProvider.empty(),
            actionInputWithPathResolver,
            /* remoteActionExecutionContext= */ null,
            /* remotePathResolver= */ null,
            BlobPolicy.KEEP_AND_REUPLOAD);
  }

  private MerkleTree build(
      Collection<? extends Map.Entry<PathFragment, ? extends ActionInput>> sortedInputs,
      Predicate<PathFragment> isToolInput,
      @Nullable SpawnScrubber spawnScrubber,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy)
      throws IOException, InterruptedException {
    if (sortedInputs.isEmpty()) {
      return emptyTree;
    }

    getFromFuture(
        cacheSubTrees(
            sortedInputs,
            isToolInput,
            metadataProvider,
            artifactPathResolver,
            remoteActionExecutionContext,
            remotePathResolver,
            blobPolicy));

    long inputFiles = 0;
    long inputBytes = 0;
    var blobs = ImmutableMap.<Digest, Object>builder();
    Deque<Directory.Builder> directoryStack = new ArrayDeque<>();
    directoryStack.push(Directory.newBuilder());

    PathFragment currentParent = PathFragment.EMPTY_FRAGMENT;
    PathFragment lastSourceDirPath = null;
    for (var entry : Iterables.concat(sortedInputs, END_OF_INPUTS_SENTINEL)) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }

      PathFragment path = entry.getKey();
      if (spawnScrubber != null && spawnScrubber.shouldOmitInput(path)) {
        continue;
      }
      if (lastSourceDirPath != null) {
        if (path.startsWith(lastSourceDirPath)) {
          // The input is part of a source directory that has already been added to the tree.
          continue;
        }
        lastSourceDirPath = null;
      }
      ActionInput input = entry.getValue();
      PathFragment newParent = path.getParentDirectory();
      if (!currentParent.equals(newParent)) {
        PathFragment commonPrefix;
        PathFragment fragmentToPop;
        if (newParent != null) {
          commonPrefix = findCommonPrefix(currentParent, newParent);
          fragmentToPop = currentParent.relativeTo(commonPrefix);
        } else {
          fragmentToPop = ROOT_FAKE_PATH_SEGMENT.getRelative(currentParent);
          // Unused.
          commonPrefix = null;
        }
        for (String dirToPop : fragmentToPop.splitToListOfSegments().reverse()) {
          byte[] directoryBlob = directoryStack.pop().build().toByteArray();
          Digest directoryBlobDigest = digestUtil.compute(directoryBlob);
          if (blobPolicy != BlobPolicy.DISCARD && directoryBlobDigest.getSizeBytes() != 0) {
            blobs.put(directoryBlobDigest, directoryBlob);
          }
          inputBytes += directoryBlobDigest.getSizeBytes();
          var topDirectory = directoryStack.peek();
          if (topDirectory == null) {
            var builtBlobs = blobs.buildKeepingLast();
            if (blobPolicy == BlobPolicy.DISCARD) {
              // Make sure that we didn't unnecessarily retain any blobs.
              checkState(builtBlobs.isEmpty());
              return new MerkleTree.RootOnly.BlobsDiscarded(
                  directoryBlobDigest, inputFiles, inputBytes);
            } else {
              return new MerkleTree.Uploadable(
                  new MerkleTree.RootOnly.BlobsUploaded(
                      directoryBlobDigest, inputFiles, inputBytes),
                  builtBlobs);
            }
          }
          topDirectory
              .addDirectoriesBuilder()
              .setName(internalToUnicode(dirToPop))
              .setDigest(directoryBlobDigest);
        }
        for (int i = 0; i < newParent.segmentCount() - commonPrefix.segmentCount(); i++) {
          directoryStack.push(Directory.newBuilder());
        }
        currentParent = newParent;
      }

      Directory.Builder currentDirectory = checkNotNull(directoryStack.peek());
      String name = internalToUnicode(path.getBaseName());
      var nodeProperties = isToolInput.test(path) ? TOOL_NODE_PROPERTIES : null;

      switch (input) {
        case Artifact treeArtifact when treeArtifact.isTreeArtifact() -> {
          var subTreeRoot =
              getFromFuture(
                  computeForTreeArtifactIfAbsent(
                      metadataProvider.getTreeMetadata(treeArtifact),
                      path,
                      isToolInput,
                      metadataProvider,
                      artifactPathResolver,
                      remoteActionExecutionContext,
                      remotePathResolver,
                      blobPolicy));
          currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTreeRoot.digest());
          inputFiles += subTreeRoot.inputFiles();
          inputBytes += subTreeRoot.inputBytes();
        }
        case Artifact runfilesArtifact when runfilesArtifact.isRunfilesTree() -> {
          var subTreeRoot =
              getFromFuture(
                  computeForRunfilesTreeIfAbsent(
                      metadataProvider.getRunfilesMetadata(runfilesArtifact),
                      path,
                      isToolInput,
                      metadataProvider,
                      artifactPathResolver,
                      remoteActionExecutionContext,
                      remotePathResolver,
                      blobPolicy));
          currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTreeRoot.digest());
          inputFiles += subTreeRoot.inputFiles();
          inputBytes += subTreeRoot.inputBytes();
        }
        case Artifact symlink when symlink.isSymlink() -> {
          Path symlinkPath = artifactPathResolver.toPath(symlink);
          var builder =
              currentDirectory
                  .addSymlinksBuilder()
                  .setName(name)
                  .setTarget(internalToUnicode(symlinkPath.readSymbolicLink().getPathString()));
          if (nodeProperties != null) {
            builder.setNodeProperties(nodeProperties);
          }
          inputFiles++;
        }
        case Artifact fileOrSourceDirectory -> {
          var metadata =
              checkNotNull(
                  metadataProvider.getInputMetadata(fileOrSourceDirectory),
                  "missing metadata: %s",
                  fileOrSourceDirectory);
          if (metadata.getType() == FileStateType.DIRECTORY) {
            var subTreeRoot =
                getFromFuture(
                    computeIfAbsent(
                        metadata,
                        () ->
                            explodeDirectory(artifactPathResolver.toPath(fileOrSourceDirectory))
                                .entrySet(),
                        isToolInput.test(path),
                        metadataProvider,
                        artifactPathResolver,
                        remoteActionExecutionContext,
                        remotePathResolver,
                        blobPolicy));
            currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTreeRoot.digest());
            inputFiles += subTreeRoot.inputFiles();
            inputBytes += subTreeRoot.inputBytes();
            // The source directory subsumes all children paths, which may be staged separately as
            // individual files or subdirectories. We rely on the inputs being sorted such that a
            // path is directly succeeded by all its children.
            if (lastSourceDirPath == null || !path.startsWith(lastSourceDirPath)) {
              lastSourceDirPath = path;
            }
          } else {
            var digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
            addFile(currentDirectory, name, digest, nodeProperties);
            if (blobPolicy != BlobPolicy.DISCARD && digest.getSizeBytes() != 0) {
              blobs.put(digest, artifactPathResolver.toPath(fileOrSourceDirectory));
            }
            inputFiles++;
            inputBytes += digest.getSizeBytes();
          }
        }
        case VirtualActionInput virtualActionInput -> {
          var digest = digestUtil.compute(virtualActionInput);
          addFile(currentDirectory, name, digest, nodeProperties);
          if (blobPolicy != BlobPolicy.DISCARD && digest.getSizeBytes() != 0) {
            blobs.put(digest, virtualActionInput);
          }
          inputFiles++;
          inputBytes += digest.getSizeBytes();
        }
        case EmptyInputDirectory ignored ->
            currentDirectory.addDirectoriesBuilder().setName(name).setDigest(emptyDigest);
        case null -> {
          // This is a sentinel value for an empty file. This case only occurs when this method is
          // called from computeForRunfilesTreeIfAbsent.
          addFile(currentDirectory, name, emptyDigest, nodeProperties);
          inputFiles++;
        }
        default -> {
          // The input is not represented by a known subtype of ActionInput. Bare ActionInputs
          // arise from exploded source directories or tests.
          Path inputPath = artifactPathResolver.toPath(input);
          var digest = digestUtil.compute(inputPath);
          addFile(currentDirectory, name, digest, nodeProperties);
          if (blobPolicy != BlobPolicy.DISCARD && digest.getSizeBytes() != 0) {
            blobs.put(digest, inputPath);
          }
          inputFiles++;
          inputBytes += digest.getSizeBytes();
        }
      }
    }

    throw new IllegalStateException("not reached");
  }

  private CompletableFuture<?> cacheSubTrees(
      Collection<? extends Map.Entry<PathFragment, ? extends ActionInput>> sortedInputs,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      RemoteActionExecutionContext remoteActionExecutionContext,
      RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy)
      throws IOException {
    ArrayList<CompletableFuture<?>> subTreeFutures = new ArrayList<>();
    for (var entry : sortedInputs) {
      var future =
          maybeCacheSubtree(
              entry.getValue(),
              entry.getKey(),
              isToolInput,
              metadataProvider,
              artifactPathResolver,
              remoteActionExecutionContext,
              remotePathResolver,
              blobPolicy);
      if (future != null) {
        subTreeFutures.add(future);
      }
    }
    return allOf(subTreeFutures.toArray(CompletableFuture[]::new));
  }

  @Nullable
  private CompletableFuture<?> maybeCacheSubtree(
      @Nullable ActionInput input,
      PathFragment mappedExecPath,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy)
      throws IOException {
    return switch (input) {
      case Artifact artifact when artifact.isTreeArtifact() ->
          computeForTreeArtifactIfAbsent(
              metadataProvider.getTreeMetadata(artifact),
              mappedExecPath,
              isToolInput,
              metadataProvider,
              artifactPathResolver,
              remoteActionExecutionContext,
              remotePathResolver,
              blobPolicy);
      case Artifact artifact when artifact.isRunfilesTree() ->
          computeForRunfilesTreeIfAbsent(
              metadataProvider.getRunfilesMetadata(artifact),
              mappedExecPath,
              isToolInput,
              metadataProvider,
              artifactPathResolver,
              remoteActionExecutionContext,
              remotePathResolver,
              blobPolicy);
      case Artifact artifact when artifact.isSourceArtifact() -> {
        var metadata =
            checkNotNull(
                metadataProvider.getInputMetadata(artifact), "missing metadata: %s", artifact);
        if (metadata.getType() != FileStateType.DIRECTORY) {
          yield null;
        }
        yield computeIfAbsent(
            metadata,
            () -> explodeDirectory(artifactPathResolver.toPath(artifact)).entrySet(),
            isToolInput.test(mappedExecPath),
            metadataProvider,
            artifactPathResolver,
            remoteActionExecutionContext,
            remotePathResolver,
            blobPolicy);
      }
      case null, default -> null;
    };
  }

  private CompletableFuture<MerkleTree.RootOnly> computeForRunfilesTreeIfAbsent(
      RunfilesArtifactValue runfilesArtifactValue,
      PathFragment mappedExecPath,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy) {
    // A runfiles tree contains either only tool inputs or only non-tool inputs. It always contains
    // at least one artifact at its canonical location: the executable for which it has been
    // created.
    var artifactAtCanonicalLocation =
        checkNotNull(
            getOneElement(
                runfilesArtifactValue
                    .getRunfilesTree()
                    .getArtifactsAtCanonicalLocationsForLogging()),
            "runfiles tree contains no artifacts at canonical location: %s",
            mappedExecPath);
    var fullPath =
        mappedExecPath.getChild("_main").getRelative(artifactAtCanonicalLocation.getRunfilesPath());
    boolean isTool = isToolInput.test(fullPath);
    // mappedExecPath and isToolInput must not be used below as they aren't part of the cache key -
    // use isTool instead.
    return computeIfAbsent(
        runfilesArtifactValue.getMetadata(),
        () ->
            ImmutableList.sortedCopyOf(
                Map.Entry.comparingByKey(HIERARCHICAL_COMPARATOR),
                // Values in this entry set may be null, which represents an empty runfile.
                runfilesArtifactValue.getRunfilesTree().getMapping().entrySet()),
        isTool,
        metadataProvider,
        artifactPathResolver,
        remoteActionExecutionContext,
        remotePathResolver,
        blobPolicy);
  }

  private CompletableFuture<MerkleTree.RootOnly> computeForTreeArtifactIfAbsent(
      TreeArtifactValue treeArtifactValue,
      PathFragment mappedExecPath,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy) {
    // A tree artifact contains either only tool inputs or only non-tool inputs.
    boolean isTool =
        !treeArtifactValue.getChildren().isEmpty()
            && isToolInput.test(
                mappedExecPath.getRelative(
                    treeArtifactValue.getChildren().first().getParentRelativePath()));
    // mappedExecPath and isToolInput must not be used below as they aren't part of the cache key -
    // use isTool instead.
    return computeIfAbsent(
        treeArtifactValue.getMetadata(),
        () ->
            Lists.transform(
                ImmutableList.sortedCopyOf(
                    comparing(
                        Artifact.TreeFileArtifact::getParentRelativePath, HIERARCHICAL_COMPARATOR),
                    treeArtifactValue.getChildren()),
                child -> entry(child.getParentRelativePath(), child)),
        isTool,
        metadataProvider,
        artifactPathResolver,
        remoteActionExecutionContext,
        remotePathResolver,
        blobPolicy);
  }

  private interface SortedInputsSupplier {
    Collection<? extends Map.Entry<PathFragment, ? extends ActionInput>> compute()
        throws IOException, InterruptedException;
  }

  private CompletableFuture<MerkleTree.RootOnly> computeIfAbsent(
      FileArtifactValue metadata,
      SortedInputsSupplier sortedInputsSupplier,
      boolean isTool,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      BlobPolicy blobPolicy) {
    var persistentCache = isTool ? persistentToolSubTreeCache : persistentNonToolSubTreeCache;
    if (blobPolicy == BlobPolicy.KEEP_AND_REUPLOAD) {
      persistentCache.invalidate(metadata);
    } else {
      var cachedRoot = persistentCache.getIfPresent(metadata);
      if (cachedRoot != null
          && (blobPolicy == BlobPolicy.DISCARD
              || cachedRoot instanceof MerkleTree.RootOnly.BlobsUploaded)) {
        return completedFuture(cachedRoot);
      }
    }
    var inFlightCacheKey = new InFlightCacheKey(metadata, isTool, blobPolicy != BlobPolicy.DISCARD);
    if (blobPolicy == BlobPolicy.KEEP_AND_REUPLOAD) {
      inFlightSubTreeCache.synchronous().invalidate(inFlightCacheKey);
    }
    return inFlightSubTreeCache
        .get(
            inFlightCacheKey,
            (key, unusedExecutor) -> {
              // There is a window in which a concurrent call may have removed the in-flight cache
              // entry while this one had already passed the check above. Recheck the persistent
              // cache to avoid unnecessary work.
              var cachedRoot = persistentCache.getIfPresent(metadata);
              if (cachedRoot != null
                  && (blobPolicy == BlobPolicy.DISCARD
                      || cachedRoot instanceof MerkleTree.RootOnly.BlobsUploaded)) {
                return completedFuture(cachedRoot);
              }
              // An ongoing computation with blobs can be reused for one that doesn't require them.
              if (blobPolicy == BlobPolicy.DISCARD) {
                var inFlightComputation =
                    inFlightSubTreeCache.getIfPresent(
                        new InFlightCacheKey(metadata, isTool, /* uploadBlobs= */ true));
                if (inFlightComputation != null) {
                  return inFlightComputation;
                }
              }
              return supplyAsync(
                      () -> {
                        try {
                          // Subtrees either consist entirely of tool inputs or don't contain any.
                          // The same applies to scrubbed inputs.
                          return build(
                              sortedInputsSupplier.compute(),
                              isTool ? alwaysTrue() : alwaysFalse(),
                              /* spawnScrubber= */ null,
                              metadataProvider,
                              artifactPathResolver,
                              remoteActionExecutionContext,
                              remotePathResolver,
                              blobPolicy);
                        } catch (IOException e) {
                          throw new WrappedException(e);
                        } catch (InterruptedException e) {
                          throw new WrappedException(e);
                        }
                      },
                      MERKLE_TREE_BUILD_POOL)
                  .thenApplyAsync(
                      merkleTree -> {
                        if (merkleTree instanceof MerkleTree.Uploadable uploadable) {
                          try {
                            if (remoteExecutionCache != null) {
                              remoteExecutionCache.ensureInputsPresent(
                                  remoteActionExecutionContext,
                                  uploadable,
                                  blobPolicy == BlobPolicy.KEEP_AND_REUPLOAD,
                                  remotePathResolver);
                            }
                          } catch (IOException e) {
                            throw new WrappedException(e);
                          } catch (InterruptedException e) {
                            throw new WrappedException(e);
                          }
                        }
                        // Move the computed root to the persistent cache so that it can be reused
                        // by later builds.
                        persistentCache
                            .asMap()
                            .compute(
                                metadata,
                                (unused, oldRoot) -> {
                                  // Don't downgrade the cached root from one indicating that its
                                  // blobs have been uploaded.
                                  return oldRoot instanceof MerkleTree.RootOnly.BlobsUploaded
                                      ? oldRoot
                                      : merkleTree.root();
                                });
                        return merkleTree.root();
                      },
                      MERKLE_TREE_UPLOAD_POOL);
            })
        // This part of the future must be kept outside the cache lambda to avoid recursive updates
        // to the in-flight cache.
        .thenApply(
            root -> {
              // Clean up the in-flight cache so that it doesn't grow unboundedly.
              inFlightSubTreeCache.synchronous().invalidate(inFlightCacheKey);
              return root;
            });
  }

  private static <T> T getFromFuture(Future<T> future) throws IOException, InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      if (e.getCause() instanceof WrappedException wrappedException) {
        wrappedException.unwrapAndThrow();
        // Not reached.
      }
      Throwables.throwIfUnchecked(e.getCause());
      throw new IllegalStateException(e);
    }
  }

  private static void addFile(
      Directory.Builder directory,
      String name,
      Digest digest,
      @Nullable NodeProperties nodeProperties) {
    var builder =
        directory
            .addFilesBuilder()
            .setName(name)
            .setDigest(digest)
            // We always treat files as executable since Bazel will `chmod 555` on the output
            // files of an action within ActionOutputMetadataStore#getMetadata after action
            // execution if no metadata was injected. We can't use real executable bit of the
            // file until this behavior is changed. See
            // https://github.com/bazelbuild/bazel/issues/13262 for more details.
            .setIsExecutable(true);
    if (nodeProperties != null) {
      builder.setNodeProperties(nodeProperties);
    }
  }

  private static PathFragment findCommonPrefix(PathFragment path1, PathFragment path2) {
    int commonSegments = 0;
    var segments2 = path2.segments().iterator();
    for (String segment : path1.segments()) {
      if (!segments2.hasNext()) {
        break;
      }
      String segment2 = segments2.next();
      if (!segment.equals(segment2)) {
        break;
      }
      commonSegments++;
    }
    return path1.subFragment(0, commonSegments);
  }

  private static ImmutableSortedMap<PathFragment, ActionInput> explodeDirectory(Path dirPath)
      throws IOException, InterruptedException {
    var inputs = ImmutableSortedMap.<PathFragment, ActionInput>orderedBy(HIERARCHICAL_COMPARATOR);
    explodeDirectory(PathFragment.EMPTY_FRAGMENT, dirPath, inputs);
    return inputs.buildOrThrow();
  }

  private static void explodeDirectory(
      PathFragment relPath, Path dirPath, ImmutableMap.Builder<PathFragment, ActionInput> inputs)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    Collection<Dirent> entries = dirPath.getRelative(relPath).readdir(Symlinks.FOLLOW);
    for (Dirent entry : entries) {
      String basename = entry.getName();
      PathFragment path = relPath.getChild(basename);
      switch (entry.getType()) {
        case FILE ->
            inputs.put(path, ActionInputHelper.fromPath(dirPath.getRelative(path).asFragment()));
        case DIRECTORY -> explodeDirectory(path, dirPath, inputs);
        default ->
            throw new IOException(
                "The file type of '%s' is not supported.".formatted(dirPath.getRelative(path)));
      }
    }
  }

  @Nullable
  private static <T> T getOneElement(NestedSet<T> nestedSet) {
    ImmutableList<T> leaves = nestedSet.getLeaves();
    if (!leaves.isEmpty()) {
      return leaves.getFirst();
    }
    ImmutableList<NestedSet<T>> nonLeaves = nestedSet.getNonLeaves();
    for (NestedSet<T> nonLeaf : nonLeaves) {
      T leaf = getOneElement(nonLeaf);
      if (leaf != null) {
        return leaf;
      }
    }
    return null;
  }

  /**
   * Returns an immutable view of the concatenation of two collections.
   *
   * <p>Use this over the unsized {@link Iterators#concat} to avoid intermediate allocations of
   * ArrayLists in methods such as {@link ImmutableList#sortedCopyOf}.
   */
  private static <T> Collection<T> concat(
      Collection<? extends T> first, Collection<? extends T> second) {
    if (first.isEmpty()) {
      return (Collection<T>) second;
    }
    if (second.isEmpty()) {
      return (Collection<T>) first;
    }
    return new AbstractCollection<>() {
      @Override
      public Iterator<T> iterator() {
        return Iterators.concat(first.iterator(), second.iterator());
      }

      @Override
      public int size() {
        return first.size() + second.size();
      }
    };
  }

  private static class EmptyInputDirectory extends ActionInputHelper.BasicActionInput {
    private final PathFragment execPath;

    EmptyInputDirectory(PathFragment execPath) {
      this.execPath = execPath;
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

  private static final class WrappedException extends RuntimeException {
    private WrappedException(IOException cause) {
      super(cause);
    }

    private WrappedException(InterruptedException cause) {
      super(cause);
    }

    public void unwrapAndThrow() throws IOException, InterruptedException {
      Throwables.throwIfInstanceOf(getCause(), IOException.class);
      Throwables.throwIfInstanceOf(getCause(), InterruptedException.class);
      throw new IllegalStateException(getCause());
    }

    @Override
    public Throwable fillInStackTrace() {
      // Don't fill in the stack trace to avoid unnecessary overhead.
      return this;
    }
  }
}
