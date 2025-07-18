package com.google.devtools.build.lib.remote.merkletree.v2;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Predicates.alwaysFalse;
import static com.google.common.base.Predicates.alwaysTrue;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;
import static java.util.concurrent.CompletableFuture.supplyAsync;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.NodeProperties;
import build.bazel.remote.execution.v2.NodeProperty;
import com.github.benmanes.caffeine.cache.AsyncCache;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
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
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.UncheckedIOException;
import java.util.AbstractCollection;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Predicate;
import javax.annotation.Nullable;

public final class MerkleTreeComputer {

  private static final NodeProperties TOOL_NODE_PROPERTIES =
      NodeProperties.newBuilder()
          .addProperties(NodeProperty.newBuilder().setName("bazel_tool_input"))
          .build();
  private static final List<Map.Entry<PathFragment, ActionInput>> END_OF_INPUTS_SENTINEL =
      ImmutableList.of(Map.entry(PathFragment.EMPTY_FRAGMENT, VirtualActionInput.EMPTY_MARKER));
  private static final PathFragment ROOT_FAKE_PATH_SEGMENT = PathFragment.create("root");

  private static final ExecutorService MERKLE_TREE_BUILD_POOL =
      Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
  private static final ExecutorService MERKLE_TREE_UPLOAD_POOL =
      Executors.newVirtualThreadPerTaskExecutor();
  private static final Cache<FileArtifactValue, MerkleTreeRoot> persistentToolSubTreeCache =
      Caffeine.newBuilder().weakKeys().build();
  private static final Cache<FileArtifactValue, MerkleTreeRoot> persistentNonToolSubTreeCache =
      Caffeine.newBuilder().weakKeys().build();
  @Nullable private static volatile Scrubber lastScrubber;

  private final DigestUtil digestUtil;
  @Nullable private final MerkleTreeUploader remoteExecutionCache;
  private final String buildRequestId;
  private final String commandId;

  private final MerkleTree emptyTree;
  private final AsyncCache<Object, MerkleTreeRoot> inFlightSubTreeCache =
      Caffeine.newBuilder().executor(MERKLE_TREE_BUILD_POOL).buildAsync();

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
    var emptyDigest = digestUtil.compute(emptyBlob);
    this.emptyTree =
        new MerkleTree(
            new MerkleTreeRoot(emptyDigest, 0, 0), ImmutableMap.of(emptyDigest, emptyBlob));
  }

  private record MerkleTreeRoot(Digest digest, long inputFiles, long inputBytes) {}

  public static final class MerkleTree {
    private final MerkleTreeRoot root;
    private final ImmutableMap<Digest, Object> blobs;

    private MerkleTree(MerkleTreeRoot root, ImmutableMap<Digest, Object> blobs) {
      this.root = root;
      this.blobs = blobs;
    }

    public Digest rootDigest() {
      return root.digest();
    }

    public long inputFiles() {
      return root.inputFiles;
    }

    public long inputBytes() {
      return root.inputBytes;
    }

    public ImmutableSet<Digest> allDigests() {
      return blobs.keySet();
    }

    @VisibleForTesting
    public ImmutableMap<Digest, Object> blobs() {
      return blobs;
    }

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

  public interface MerkleTreeUploader {
    ListenableFuture<Void> upload(RemoteActionExecutionContext context, Digest digest, byte[] data);

    ListenableFuture<Void> upload(
        RemoteActionExecutionContext context,
        RemotePathResolver remotePathResolver,
        Digest digest,
        Path path);

    ListenableFuture<Void> upload(
        RemoteActionExecutionContext context, Digest digest, VirtualActionInput virtualActionInput);

    void ensureInputsPresent(
        RemoteActionExecutionContext context,
        MerkleTree merkleTree,
        boolean force,
        RemotePathResolver remotePathResolver)
        throws IOException, InterruptedException;
  }

  public enum SubTreePolicy {
    DISCARD,
    UPLOAD,
    FORCE_UPLOAD,
  }

  public MerkleTree buildForSpawn(
      Spawn spawn,
      Predicate<PathFragment> isToolInput,
      @Nullable Scrubber scrubber,
      SpawnExecutionContext spawnExecutionContext,
      RemotePathResolver remotePathResolver,
      SubTreePolicy subTreePolicy)
      throws IOException, InterruptedException, LostInputsExecException {
    if (!Objects.equals(scrubber, lastScrubber)) {
      persistentToolSubTreeCache.invalidateAll();
      persistentNonToolSubTreeCache.invalidateAll();
      lastScrubber = scrubber;
    }
    var spawnInputs = spawn.getInputFiles().toList();
    // Add output directories to inputs so that they are created as empty directories by the
    // executor. The spec only requires the executor to create the parent directory of an output
    // directory, which differs from the behavior of both local and sandboxed execution.
    var outputDirectories =
        spawn.getOutputFiles().stream()
            .filter(output -> output instanceof Artifact artifact && artifact.isTreeArtifact())
            .map(outputDir -> new InputDirectory(outputDir.getExecPath()))
            .collect(toImmutableList());
    // Reduce peak memory usage by avoiding the allocation of intermediate arrays and TreeMaps, as
    // well as the prolonged retention of mapped paths.
    var allInputs =
        ImmutableList.sortedCopyOf(
            Comparator.comparing(
                input -> getOutputPath(input, remotePathResolver, spawn.getPathMapper())),
            new AbstractCollection<ActionInput>() {
              @Override
              public Iterator<ActionInput> iterator() {
                return Iterables.concat(spawnInputs, outputDirectories).iterator();
              }

              @Override
              public int size() {
                return spawnInputs.size() + outputDirectories.size();
              }
            });
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
    try {
      return build(
          Lists.transform(
              allInputs,
              input ->
                  Map.entry(
                      getOutputPath(input, remotePathResolver, spawn.getPathMapper()), input)),
          isToolInput,
          scrubber != null ? scrubber.forSpawn(spawn) : null,
          spawnExecutionContext.getInputMetadataProvider(),
          spawnExecutionContext.getPathResolver(),
          remoteActionExecutionContext,
          remotePathResolver,
          subTreePolicy);
    } catch (BulkTransferException e) {
      e.getLostArtifacts(spawnExecutionContext.getInputMetadataProvider()::getInput)
          .throwIfNotEmpty();
      throw e;
    }
  }

  // TODO: This may not be the correct path to test isToolInput on with the sibling repository
  // layout enabled, but this isn't covered by any tests.
  private static PathFragment getOutputPath(
      ActionInput input, RemotePathResolver remotePathResolver, PathMapper pathMapper) {
    return PathFragment.create(remotePathResolver.getWorkingDirectory())
        .getRelative(pathMapper.map(input.getExecPath()));
  }

  public MerkleTree buildForFiles(SortedMap<PathFragment, Path> inputs)
      throws IOException, InterruptedException {
    ArtifactPathResolver absolutePathResolver;
    if (inputs.isEmpty()) {
      absolutePathResolver = ArtifactPathResolver.IDENTITY;
    } else {
      Path firstPath = inputs.firstEntry().getValue();
      Path absoluteRoot = firstPath.getFileSystem().getPath(firstPath.asFragment().getDriveStr());
      absolutePathResolver = ArtifactPathResolver.forExecRoot(absoluteRoot);
    }
    return build(
        Collections2.transform(
            inputs.entrySet(),
            e -> Map.entry(e.getKey(), ActionInputHelper.fromPath(e.getValue().asFragment()))),
        alwaysFalse(),
        /* spawnScrubber= */ null,
        StaticInputMetadataProvider.empty(),
        absolutePathResolver,
        /* remoteActionExecutionContext= */ null,
        /* remotePathResolver= */ null,
        SubTreePolicy.UPLOAD);
  }

  private MerkleTree build(
      Collection<? extends Map.Entry<PathFragment, ? extends ActionInput>> sortedInputs,
      Predicate<PathFragment> isToolInput,
      @Nullable SpawnScrubber spawnScrubber,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      SubTreePolicy subTreePolicy)
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
            subTreePolicy));

    long inputFiles = 0;
    long inputBytes = 0;
    var blobs = ImmutableMap.<Digest, Object>builder();
    Deque<Directory.Builder> directoryStack = new ArrayDeque<>();
    directoryStack.push(Directory.newBuilder());

    PathFragment currentParent = PathFragment.EMPTY_FRAGMENT;
    for (var entry : Iterables.concat(sortedInputs, END_OF_INPUTS_SENTINEL)) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }

      PathFragment path = entry.getKey();
      if (spawnScrubber != null && spawnScrubber.shouldOmitInput(path)) {
        continue;
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
          if (subTreePolicy != SubTreePolicy.DISCARD && directoryBlobDigest.getSizeBytes() != 0) {
            blobs.put(directoryBlobDigest, directoryBlob);
          }
          inputBytes += directoryBlobDigest.getSizeBytes();
          var topDirectory = directoryStack.peek();
          if (topDirectory == null) {
            return new MerkleTree(
                new MerkleTreeRoot(directoryBlobDigest, inputFiles, inputBytes),
                blobs.buildKeepingLast());
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
                      subTreePolicy));
          currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTreeRoot.digest());
          inputFiles += subTreeRoot.inputFiles();
          inputBytes += subTreeRoot.inputBytes();
        }
        case Artifact runfilesTreeArtifact when runfilesTreeArtifact.isRunfilesTree() -> {
          var subTreeRoot =
              getFromFuture(
                  computeForRunfilesTreeIfAbsent(
                      metadataProvider.getRunfilesMetadata(runfilesTreeArtifact),
                      path,
                      isToolInput,
                      metadataProvider,
                      artifactPathResolver,
                      remoteActionExecutionContext,
                      remotePathResolver,
                      subTreePolicy));
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
                        subTreePolicy));
            currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTreeRoot.digest());
            inputFiles += subTreeRoot.inputFiles();
            inputBytes += subTreeRoot.inputBytes();
          } else {
            var digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
            addFile(currentDirectory, name, digest, nodeProperties);
            if (subTreePolicy != SubTreePolicy.DISCARD && digest.getSizeBytes() != 0) {
              blobs.put(digest, artifactPathResolver.toPath(fileOrSourceDirectory));
            }
            inputFiles++;
            inputBytes += digest.getSizeBytes();
          }
        }
        case VirtualActionInput virtualActionInput -> {
          var digest = digestUtil.compute(virtualActionInput);
          addFile(currentDirectory, name, digest, nodeProperties);
          if (subTreePolicy != SubTreePolicy.DISCARD && digest.getSizeBytes() != 0) {
            blobs.put(digest, virtualActionInput);
          }
          inputFiles++;
          inputBytes += digest.getSizeBytes();
        }
        case InputDirectory ignored ->
            currentDirectory
                .addDirectoriesBuilder()
                .setName(name)
                .setDigest(digestUtil.emptyDigest());
        case null -> {
          // This is a sentinel value for an empty file.
          addFile(currentDirectory, name, digestUtil.emptyDigest(), nodeProperties);
          inputFiles++;
        }
        default -> {
          // The input is not represented by a known subtype of ActionInput. Bare ActionInputs
          // arise from exploded source directories or tests.
          Path inputPath = artifactPathResolver.toPath(input);
          var digest = digestUtil.compute(inputPath);
          addFile(currentDirectory, name, digest, nodeProperties);
          if (subTreePolicy != SubTreePolicy.DISCARD && digest.getSizeBytes() != 0) {
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
      SubTreePolicy subTreePolicy)
      throws IOException, InterruptedException {
    ArrayList<CompletableFuture<?>> subTreeFutures = new ArrayList<>();
    for (var entry : sortedInputs) {
      var future =
          cacheSubTree(
              entry.getValue(),
              entry.getKey(),
              isToolInput,
              metadataProvider,
              artifactPathResolver,
              remoteActionExecutionContext,
              remotePathResolver,
              subTreePolicy);
      if (future != null) {
        subTreeFutures.add(future);
      }
    }
    return allOf(subTreeFutures.toArray(CompletableFuture[]::new));
  }

  @Nullable
  private CompletableFuture<?> cacheSubTree(
      ActionInput input,
      PathFragment mappedExecPath,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      SubTreePolicy subTreePolicy)
      throws IOException {
    if (!(input instanceof Artifact artifact)) {
      return null;
    }
    if (artifact.isTreeArtifact()) {
      return computeForTreeArtifactIfAbsent(
          metadataProvider.getTreeMetadata(artifact),
          mappedExecPath,
          isToolInput,
          metadataProvider,
          artifactPathResolver,
          remoteActionExecutionContext,
          remotePathResolver,
          subTreePolicy);
    }
    if (artifact.isRunfilesTree()) {
      return computeForRunfilesTreeIfAbsent(
          metadataProvider.getRunfilesMetadata(artifact),
          mappedExecPath,
          isToolInput,
          metadataProvider,
          artifactPathResolver,
          remoteActionExecutionContext,
          remotePathResolver,
          subTreePolicy);
    }
    if (artifact.isSourceArtifact()) {
      var metadata =
          checkNotNull(
              metadataProvider.getInputMetadata(artifact), "missing metadata: %s", artifact);
      if (metadata.getType() == FileStateType.DIRECTORY) {
        return computeIfAbsent(
            metadata,
            () -> explodeDirectory(artifactPathResolver.toPath(artifact)).entrySet(),
            isToolInput.test(mappedExecPath),
            metadataProvider,
            artifactPathResolver,
            remoteActionExecutionContext,
            remotePathResolver,
            subTreePolicy);
      }
    }
    return null;
  }

  private CompletableFuture<MerkleTreeRoot> computeForRunfilesTreeIfAbsent(
      RunfilesArtifactValue runfilesArtifactValue,
      PathFragment mappedExecPath,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      SubTreePolicy subTreePolicy) {
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
    // mappedExecPath and isToolInput must not be used below as they aren't part of the cache key.
    return computeIfAbsent(
        runfilesArtifactValue.getMetadata(),
        () -> runfilesArtifactValue.getRunfilesTree().getMapping().entrySet(),
        isTool,
        metadataProvider,
        artifactPathResolver,
        remoteActionExecutionContext,
        remotePathResolver,
        subTreePolicy);
  }

  private CompletableFuture<MerkleTreeRoot> computeForTreeArtifactIfAbsent(
      TreeArtifactValue treeArtifactValue,
      PathFragment mappedExecPath,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      SubTreePolicy subTreePolicy) {
    // A tree artifact contains either only tool inputs or only non-tool inputs.
    boolean isTool =
        !treeArtifactValue.getChildren().isEmpty()
            && isToolInput.test(
                mappedExecPath.getRelative(
                    treeArtifactValue.getChildren().first().getParentRelativePath()));
    // mappedExecPath and isToolInput must not be used below as they aren't part of the cache key.
    return computeIfAbsent(
        treeArtifactValue.getMetadata(),
        () ->
            Collections2.transform(
                // getChildren() returns a set that is sorted by exec path, which gives the same
                // order as sorting by parent relative path.
                treeArtifactValue.getChildren(),
                child -> Map.entry(child.getParentRelativePath(), child)),
        isTool,
        metadataProvider,
        artifactPathResolver,
        remoteActionExecutionContext,
        remotePathResolver,
        subTreePolicy);
  }

  private interface SortedInputsSupplier {
    Collection<? extends Map.Entry<PathFragment, ? extends ActionInput>> compute()
        throws IOException, InterruptedException;
  }

  private CompletableFuture<MerkleTreeRoot> computeIfAbsent(
      FileArtifactValue metadata,
      SortedInputsSupplier sortedInputsSupplier,
      boolean isTool,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver,
      @Nullable RemoteActionExecutionContext remoteActionExecutionContext,
      @Nullable RemotePathResolver remotePathResolver,
      SubTreePolicy subTreePolicy) {
    var persistentCache = isTool ? persistentToolSubTreeCache : persistentNonToolSubTreeCache;
    if (subTreePolicy == SubTreePolicy.FORCE_UPLOAD) {
      persistentCache.invalidate(metadata);
    } else {
      var cachedRoot = persistentCache.getIfPresent(metadata);
      if (cachedRoot != null) {
        return completedFuture(cachedRoot);
      }
    }
    var inFlightCacheKey = inFlightCacheKeyFor(metadata, isTool);
    if (subTreePolicy == SubTreePolicy.FORCE_UPLOAD) {
      inFlightSubTreeCache.synchronous().invalidate(inFlightCacheKey);
    }
    return inFlightSubTreeCache.get(
        inFlightCacheKey,
        (key, buildExecutor) -> {
          // There is a window in which a concurrent call may have removed the
          // in-flight cache entry while this one had already passed the check
          // above. Recheck the persistent cache to avoid unnecessary work.
          var cachedRoot = persistentCache.getIfPresent(metadata);
          if (cachedRoot != null) {
            return completedFuture(cachedRoot);
          }
          return supplyAsync(
                  () -> {
                    try {
                      // Subtrees either consist entirely of tool inputs or don't contain
                      // any. The same applies to scrubbed inputs.
                      return build(
                          sortedInputsSupplier.compute(),
                          isTool ? alwaysTrue() : alwaysFalse(),
                          /* spawnScrubber= */ null,
                          metadataProvider,
                          artifactPathResolver,
                          remoteActionExecutionContext,
                          remotePathResolver,
                          subTreePolicy);
                    } catch (IOException e) {
                      throw new UncheckedIOException(e);
                    } catch (InterruptedException e) {
                      throw new UncheckedIOException(new InterruptedIOException());
                    }
                  },
                  MERKLE_TREE_BUILD_POOL)
              .thenApplyAsync(
                  merkleTree -> {
                    try {
                      if (remoteExecutionCache != null && subTreePolicy != SubTreePolicy.DISCARD) {
                        remoteExecutionCache.ensureInputsPresent(
                            remoteActionExecutionContext,
                            merkleTree,
                            subTreePolicy == SubTreePolicy.FORCE_UPLOAD,
                            remotePathResolver);
                      }
                    } catch (IOException e) {
                      throw new UncheckedIOException(e);
                    } catch (InterruptedException e) {
                      throw new UncheckedIOException(new InterruptedIOException());
                    }
                    return merkleTree.root;
                  },
                  MERKLE_TREE_UPLOAD_POOL)
              // Move the computed root to the persistent cache so that it can be reused by later
              // builds as well as to keep memory usage low: the in-flight cache retains full trees,
              // the persistent cache only retains the root's digest.
              .thenApply(
                  root -> {
                    persistentCache.put(metadata, root);
                    inFlightSubTreeCache.synchronous().invalidate(inFlightCacheKey);
                    return root;
                  });
        });
  }

  private static <T> T getFromFuture(Future<T> future) throws IOException, InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      if (e.getCause() instanceof UncheckedIOException uncheckedIOException) {
        if (uncheckedIOException.getCause() instanceof InterruptedIOException) {
          throw new InterruptedException();
        }
        throw uncheckedIOException.getCause();
      }
      Throwables.throwIfUnchecked(e.getCause());
      throw new IllegalStateException(e);
    }
  }

  private static Object inFlightCacheKeyFor(FileArtifactValue metadata, boolean isTool) {
    record ToolFileArtifactValue(FileArtifactValue metadata) {}
    return isTool ? new ToolFileArtifactValue(metadata) : metadata;
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

  private static SortedMap<PathFragment, ActionInput> explodeDirectory(Path dirPath)
      throws IOException, InterruptedException {
    SortedMap<PathFragment, ActionInput> inputs = new TreeMap<>();
    explodeDirectory(PathFragment.EMPTY_FRAGMENT, dirPath, inputs);
    return inputs;
  }

  private static void explodeDirectory(
      PathFragment relPath, Path dirPath, SortedMap<PathFragment, ActionInput> inputs)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    Collection<Dirent> entries = dirPath.getRelative(relPath).readdir(Symlinks.FOLLOW);
    for (Dirent entry : entries) {
      String basename = entry.getName();
      PathFragment path = relPath.getChild(basename);
      switch (entry.getType()) {
        case FILE -> inputs.put(path, ActionInputHelper.fromPath(relPath));
        case DIRECTORY -> explodeDirectory(path, dirPath, inputs);
        default ->
            throw new IOException(
                "Unsupported file type of %s: %s".formatted(path, entry.getType()));
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

  private static class InputDirectory extends ActionInputHelper.BasicActionInput {
    private final PathFragment execPath;

    InputDirectory(PathFragment execPath) {
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
}
