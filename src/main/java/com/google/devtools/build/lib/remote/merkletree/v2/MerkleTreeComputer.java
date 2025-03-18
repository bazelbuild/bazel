package com.google.devtools.build.lib.remote.merkletree.v2;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.NodeProperties;
import build.bazel.remote.execution.v2.NodeProperty;
import com.github.benmanes.caffeine.cache.AsyncCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.UncheckedIOException;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.function.Predicate;

public final class MerkleTreeComputer {

  private static final NodeProperties TOOL_NODE_PROPERTIES =
      NodeProperties.newBuilder()
          .addProperties(NodeProperty.newBuilder().setName("bazel_tool_input"))
          .build();
  private static final List<Map.Entry<PathFragment, ActionInput>> END_OF_INPUTS_SENTINEL =
      ImmutableList.of(Map.entry(PathFragment.EMPTY_FRAGMENT, VirtualActionInput.EMPTY_MARKER));
  private static final PathFragment ROOT_FAKE_PATH_SEGMENT = PathFragment.create("root");

  private static final AsyncCache<FileArtifactValue, MerkleTree> directoryDigestCache =
      Caffeine.newBuilder().weakKeys().buildAsync();

  private final DigestUtil digestUtil;
  private final MerkleTree emptyTree;

  public MerkleTreeComputer(DigestUtil digestUtil) {
    this.digestUtil = digestUtil;
    this.emptyTree = new MerkleTree(digestUtil.compute(new byte[0]), 0, 0, ImmutableMap.of());
  }

  // TODO: Drop blobs after they have been uploaded.
  public record MerkleTree(
      Digest rootDigest, long inputFiles, long inputBytes, ImmutableMap<Digest, Object> blobs) {
    public Optional<ListenableFuture<Void>> upload(BlobUploader uploader, Digest digest) {
      return switch (blobs.get(digest)) {
        case byte[] data -> Optional.of(uploader.upload(digest, data));
        case Path path -> Optional.of(uploader.upload(digest, path));
        case VirtualActionInput virtualActionInput ->
            Optional.of(uploader.upload(digest, virtualActionInput));
        case null -> Optional.empty();
        default -> throw new IllegalStateException("Unexpected blob type: " + blobs.get(digest));
      };
    }
  }

  public interface BlobUploader {
    ListenableFuture<Void> upload(Digest digest, byte[] data);

    ListenableFuture<Void> upload(Digest digest, Path file);

    ListenableFuture<Void> upload(Digest digest, VirtualActionInput virtualActionInput);
  }

  public MerkleTree buildForSpawn(
      Spawn spawn,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, InterruptedException {
    TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
    PathMapper pathMapper = spawn.getPathMapper();
    for (ActionInput input : spawn.getInputFiles().toList()) {
      inputMap.put(pathMapper.map(input.getExecPath()), input);
    }
    // Add output directories to inputs so that they are created as empty directories by the
    // executor. The spec only requires the executor to create the parent directory of an output
    // directory, which differs from the behavior of both local and sandboxed execution.
    for (ActionInput output : spawn.getOutputFiles()) {
      if (output instanceof Artifact artifact && artifact.isTreeArtifact()) {
        var mappedPath = pathMapper.map(artifact.getExecPath());
        inputMap.put(mappedPath, new InputDirectory(mappedPath));
      }
    }
    return build(inputMap, isToolInput, metadataProvider, artifactPathResolver);
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
        Maps.transformValues(inputs, path -> ActionInputHelper.fromPath(path.asFragment())),
        Predicates.alwaysFalse(),
        StaticInputMetadataProvider.empty(),
        absolutePathResolver);
  }

  private MerkleTree build(
      SortedMap<PathFragment, ? extends ActionInput> inputs,
      Predicate<PathFragment> isToolInput,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, InterruptedException {
    if (inputs.isEmpty()) {
      return emptyTree;
    }

    // TODO: Precompute the self-contained subtrees corresponding to tree artifacts and runfiles
    // trees before starting the compute the full tree. This can help reduce peak memory usage.

    long inputFiles = 0;
    long inputBytes = 0;
    // inputs.size() is unlikely to be the correct size, but it's a better lower bound than the
    // default (4).
    var blobs = ImmutableMap.<Digest, Object>builderWithExpectedSize(inputs.size());
    Deque<Directory.Builder> directoryStack = new ArrayDeque<>();
    directoryStack.push(Directory.newBuilder());

    PathFragment currentParent = PathFragment.EMPTY_FRAGMENT;
    for (var entry : Iterables.concat(inputs.entrySet(), END_OF_INPUTS_SENTINEL)) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }

      PathFragment path = entry.getKey();
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
          blobs.put(directoryBlobDigest, directoryBlob);
          inputBytes += directoryBlobDigest.getSizeBytes();
          var topDirectory = directoryStack.peek();
          if (topDirectory == null) {
            return new MerkleTree(
                directoryBlobDigest, inputFiles, inputBytes, blobs.buildKeepingLast());
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
      var nodeProperties =
          isToolInput.test(path) ? TOOL_NODE_PROPERTIES : NodeProperties.getDefaultInstance();

      switch (input) {
        case Artifact treeArtifact when treeArtifact.isTreeArtifact() -> {
          var treeArtifactValue = metadataProvider.getTreeMetadata(treeArtifact);
          boolean rootIsTool =
              !treeArtifactValue.getChildren().isEmpty()
                  && isToolInput.test(
                      path.getRelative(
                          treeArtifactValue.getChildren().first().getParentRelativePath()));
          var subTree =
              computeForTreeArtifactIfAbsent(
                  treeArtifactValue, rootIsTool, metadataProvider, artifactPathResolver);
          currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTree.rootDigest);
          blobs.putAll(subTree.blobs());
          inputFiles += subTree.inputFiles;
          inputBytes += subTree.inputBytes;
        }
        case Artifact runfilesTreeArtifact when runfilesTreeArtifact.isRunfilesTree() -> {
          var runfilesArtifactValue = metadataProvider.getRunfilesMetadata(runfilesTreeArtifact);
          var subTree =
              computeForRunfilesTreeIfAbsent(
                  runfilesArtifactValue, metadataProvider, artifactPathResolver);
          currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTree.rootDigest);
          blobs.putAll(subTree.blobs());
          inputFiles += subTree.inputFiles;
          inputBytes += subTree.inputBytes;
        }
        case Artifact symlink when symlink.isSymlink() -> {
          Path symlinkPath = artifactPathResolver.toPath(symlink);
          currentDirectory
              .addSymlinksBuilder()
              .setName(name)
              .setTarget(internalToUnicode(symlinkPath.readSymbolicLink().getPathString()))
              .setNodeProperties(nodeProperties);
          inputFiles++;
        }
        case Artifact fileOrSourceDirectory -> {
          var metadata =
              checkNotNull(
                  metadataProvider.getInputMetadata(fileOrSourceDirectory),
                  "missing metadata: %s",
                  fileOrSourceDirectory);
          if (metadata.getType() == FileStateType.DIRECTORY) {
            var subTree =
                computeIfAbsent(
                    metadata,
                    () -> explodeDirectory(artifactPathResolver.toPath(fileOrSourceDirectory)),
                    isToolInput.test(path),
                    metadataProvider,
                    artifactPathResolver);
            currentDirectory.addDirectoriesBuilder().setName(name).setDigest(subTree.rootDigest);
            blobs.putAll(subTree.blobs());
            inputFiles += subTree.inputFiles;
            inputBytes += subTree.inputBytes;
          } else {
            var digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
            addFile(currentDirectory, name, digest, nodeProperties);
            blobs.put(digest, fileOrSourceDirectory.getPath());
            inputFiles++;
            inputBytes += digest.getSizeBytes();
          }
        }
        case VirtualActionInput virtualActionInput -> {
          var digest = digestUtil.compute(virtualActionInput);
          addFile(currentDirectory, name, digest, nodeProperties);
          blobs.put(digest, virtualActionInput);
          inputFiles++;
          inputBytes += digest.getSizeBytes();
        }
        case InputDirectory ignored -> {
          currentDirectory
              .addDirectoriesBuilder()
              .setName(name)
              .setDigest(digestUtil.emptyDigest());
        }
        default -> {
          Path inputPath = artifactPathResolver.toPath(input);
          var digest = digestUtil.compute(inputPath);
          addFile(currentDirectory, name, digest, nodeProperties);
          blobs.put(digest, inputPath);
          inputFiles++;
          inputBytes += digest.getSizeBytes();
        }
      }
    }

    throw new IllegalStateException("not reached");
  }

  private MerkleTree computeForRunfilesTreeIfAbsent(
      RunfilesArtifactValue runfilesArtifactValue,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, InterruptedException {
    return computeIfAbsent(
        runfilesArtifactValue.getMetadata(),
        () -> runfilesArtifactValue.getRunfilesTree().getMapping(),
        /* TODO */ false,
        metadataProvider,
        artifactPathResolver);
  }

  private MerkleTree computeForTreeArtifactIfAbsent(
      TreeArtifactValue treeArtifactValue,
      boolean rootIsTool,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, InterruptedException {
    return computeIfAbsent(
        treeArtifactValue.getMetadata(),
        () ->
            treeArtifactValue.getChildren().stream()
                .collect(
                    toImmutableSortedMap(
                        PathFragment::compareTo,
                        Artifact.TreeFileArtifact::getParentRelativePath,
                        treeFile -> treeFile)),
        rootIsTool,
        metadataProvider,
        artifactPathResolver);
  }

  private interface SortedInputsSupplier {
    SortedMap<PathFragment, ? extends ActionInput> compute() throws IOException;
  }

  private MerkleTree computeIfAbsent(
      FileArtifactValue metadata,
      SortedInputsSupplier sortedInputsSupplier,
      boolean rootIsTool,
      InputMetadataProvider metadataProvider,
      ArtifactPathResolver artifactPathResolver)
      throws IOException, InterruptedException {
    try {
      return directoryDigestCache
          .get(
              metadata,
              unused -> {
                try {
                  return build(
                      sortedInputsSupplier.compute(),
                      // TODO: Include rootIsTool in the cache key.
                      rootIsTool ? unused2 -> true : unused2 -> false,
                      metadataProvider,
                      artifactPathResolver);
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                } catch (InterruptedException e) {
                  throw new UncheckedIOException(new InterruptedIOException());
                }
              })
          .get();
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

  private static void addFile(
      Directory.Builder directory, String name, Digest digest, NodeProperties nodeProperties) {
    directory
        .addFilesBuilder()
        .setName(name)
        .setDigest(digest)
        // We always treat files as executable since Bazel will `chmod 555` on the output
        // files of an action within ActionOutputMetadataStore#getMetadata after action
        // execution if no metadata was injected. We can't use real executable bit of the
        // file until this behavior is changed. See
        // https://github.com/bazelbuild/bazel/issues/13262 for more details.
        .setIsExecutable(true)
        .setNodeProperties(nodeProperties);
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
      throws IOException {
    SortedMap<PathFragment, ActionInput> inputs = new TreeMap<>();
    explodeDirectory(PathFragment.EMPTY_FRAGMENT, dirPath, inputs);
    return inputs;
  }

  private static void explodeDirectory(
      PathFragment relPath, Path dirPath, SortedMap<PathFragment, ActionInput> inputs)
      throws IOException {
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
