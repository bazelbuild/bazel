// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.actions.FileStateType.SYMLINK;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.sparselyAggregateWriteStatuses;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.DIRECTORY_KEY_DELIMITER;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.FILE_KEY_DELIMITER;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.MAX_KEY_LENGTH;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.computeCacheKey;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantFileData.CONSTANT_FILE;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantListingData.CONSTANT_LISTING;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ConstantNodeData.CONSTANT_NODE;
import static com.google.devtools.build.lib.util.TestType.isInTest;
import static com.google.devtools.build.lib.vfs.RootedPath.toRootedPath;
import static java.lang.Math.max;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.github.luben.zstd.ZstdOutputStream;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.BundledFileSystem;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes.NestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes.NestedFileOpNodesWithSource;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.serialization.KeyBytesProvider;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.StringKey;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.WriteStatus;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FileDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FileDataInfoOrFuture;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FileInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureFileDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureListingDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureNodeDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ListingDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ListingDataInfoOrFuture;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.ListingInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeDataInfoOrFuture;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.NodeInvalidationDataInfo;
import com.google.devtools.build.lib.skyframe.serialization.proto.DirectoryListingInvalidationData;
import com.google.devtools.build.lib.skyframe.serialization.proto.FileInvalidationData;
import com.google.devtools.build.lib.skyframe.serialization.proto.Symlink;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Records {@link FileKey}, {@link DirectoryListingKey} or {@link AbstractNestedFileOpNodes}
 * invalidation to a remote {@link KeyValueWriter}.
 */
final class FileDependencySerializer {

  @VisibleForTesting public static final int COMPRESSION_NUM_BYTES_THRESHOLD = 580;
  private final LongVersionGetter versionGetter;
  private final InMemoryGraph graph;
  private final KeyValueWriter writer;

  private final ValueOrFutureMap<FileKey, FileDataInfoOrFuture, FileDataInfo, FutureFileDataInfo>
      fileDataInfo =
          new ValueOrFutureMap<>(
              new ConcurrentHashMap<>(),
              FutureFileDataInfo::new,
              this::populateFutureFileDataInfo,
              FutureFileDataInfo.class);

  private final ValueOrFutureMap<
          DirectoryListingKey, ListingDataInfoOrFuture, ListingDataInfo, FutureListingDataInfo>
      listingDataInfo =
          new ValueOrFutureMap<>(
              new ConcurrentHashMap<>(),
              FutureListingDataInfo::new,
              this::populateFutureListingDataInfo,
              FutureListingDataInfo.class);

  FileDependencySerializer(
      LongVersionGetter versionGetter, InMemoryGraph graph, KeyValueWriter writer) {
    this.versionGetter = versionGetter;
    this.graph = graph;
    this.writer = writer;
  }

  /**
   * Stores data about a {@code node} and its transitive dependencies in {@link #writer} to be used
   * for invalidation.
   *
   * <p>The resulting data can be embedded in reverse deps of {@code node} and used to invalidate
   * them by checking against a list of changed files and directory listings.
   *
   * <p>See comments at {@link FileInvalidationData} and {@link DirectoryListingInvalidationData}
   * for more details about the data being persisted.
   */
  InvalidationDataInfoOrFuture registerDependency(FileOpNode node) {
    switch (node) {
      case FileKey file:
        return registerDependency(file);
      case DirectoryListingKey listing:
        return registerDependency(listing);
      case AbstractNestedFileOpNodes nested:
        return registerDependency(nested);
    }
  }

  FileDataInfoOrFuture registerDependency(FileKey key) {
    return fileDataInfo.getValueOrFuture(key);
  }

  ListingDataInfoOrFuture registerDependency(DirectoryListingKey key) {
    return listingDataInfo.getValueOrFuture(key);
  }

  /**
   * Registers a dependency on the set of transitive dependencies represented by {@code node}.
   *
   * <p>Uploads the result to the {@link #writer}.
   */
  NodeDataInfoOrFuture registerDependency(AbstractNestedFileOpNodes node) {
    var reference = (NodeDataInfoOrFuture) node.getSerializationScratch();
    if (reference != null) {
      return reference;
    }

    FutureNodeDataInfo future;
    synchronized (node) {
      reference = (NodeDataInfoOrFuture) node.getSerializationScratch();
      if (reference != null) {
        return reference;
      }

      future = new FutureNodeDataInfo(node);
      node.setSerializationScratch(future);
    }

    // If this is reached, this thread owns `future` and must set its value.
    try {
      return populateFutureNodeDataInfo(future);
    } finally {
      future.verifyComplete();
    }
  }

  /**
   * Populates the {@link FileDataInfoOrFuture} for the given {@link FutureFileDataInfo}.
   *
   * <p>This method is responsible for resolving the {@link FileKey} and its dependencies, and
   * uploading the resulting {@link FileInvalidationData} to the {@link #writer}.
   *
   * @param future The {@link FutureFileDataInfo} to populate.
   * @return The populated {@link FileDataInfoOrFuture}.
   */
  FileDataInfoOrFuture populateFutureFileDataInfo(FutureFileDataInfo future) {
    FileKey key = future.key();
    RootedPath rootedPath = key.argument();
    RootedPath parentRootedPath;
    // Builtin files don't change.
    if ((rootedPath.getRoot().getFileSystem() instanceof BundledFileSystem)
        // Assumes that the root folder doesn't change.
        || (parentRootedPath = rootedPath.getParentDirectory()) == null) {
      return future.completeWith(CONSTANT_FILE);
    }

    InMemoryNodeEntry nodeEntry = graph.getIfPresent(key);
    if (nodeEntry == null) {
      return future.failWith(new MissingSkyframeEntryException(key));
    }
    var value = (FileValue) nodeEntry.getValue();
    RootedPath realRootedPath = value.realRootedPath(rootedPath);

    long initialMtsv;
    if (value.isDirectory()) {
      // Matches the behavior of PathVersionGetter.getVersionForExistingPathInternal.
      initialMtsv = LongVersionGetter.MINIMAL;
    } else {
      try {
        initialMtsv = getVersion(realRootedPath, value.exists());
      } catch (IOException e) {
        return future.failWith(e);
      }
    }
    var uploader =
        new FileInvalidationDataUploader(
            /* rootedPath= */ rootedPath,
            /* parentRootedPath= */ parentRootedPath,
            /* realRootedPath= */ realRootedPath,
            value.exists(),
            initialMtsv);
    // The following steps are performed to ensure that ancestors and ancestor symlinks are resolved
    // to compute the correct MTSV:
    // 1. Call fullyResolvePath to register all the parents of the current rootedPath as
    //    dependencies first.
    // 2. The transform() method takes the output of the first parameter (a future) and passes it to
    //    the second parameter (a function).
    // 3. The output of fullyResolvePath is Void, so the transform method is only being used as a
    //    stop to not trigger the upload of the current rootedPath till its parents have been
    //    registered.
    // 4. The uploader itself is a Function that directly returns a FileDataInfo but gets wrapped as
    //    a future by the transform method.
    // 5. The upload happens through the put() operation in the writer inside the uploader.
    return future.completeWith(
        Futures.transform(
            fullyResolvePath(value.isSymlink() ? value.getUnresolvedLinkTarget() : null, uploader),
            uploader,
            directExecutor()));
  }

  /**
   * Performs the upload of the {@link FileInvalidationData} once resolution is complete in the
   * {@link #apply} callback.
   */
  private final class FileInvalidationDataUploader
      implements Function<Void, FileInvalidationDataInfo> {
    private final RootedPath rootedPath;
    private final RootedPath parentRootedPath;
    private final RootedPath realRootedPath;
    private final boolean exists;

    private final FileInvalidationData.Builder data = FileInvalidationData.newBuilder();
    private final ArrayList<WriteStatus> writeStatuses = new ArrayList<>();
    private long mtsv;

    private FileInvalidationDataUploader(
        RootedPath rootedPath,
        RootedPath parentRootedPath,
        RootedPath realRootedPath,
        boolean exists,
        long initialMtsv) {
      this.rootedPath = rootedPath;
      this.parentRootedPath = parentRootedPath;
      this.realRootedPath = realRootedPath;
      this.exists = exists;
      this.mtsv = initialMtsv;
    }

    @Override
    public FileInvalidationDataInfo apply(Void unused) {
      String cacheKey = computeCacheKey(rootedPath.getRootRelativePath(), mtsv, FILE_KEY_DELIMITER);
      KeyBytesProvider keyBytes = getKeyBytes(cacheKey, data::setOverflowKey);
      byte[] dataBytes = data.build().toByteArray();
      writeStatuses.add(writer.put(keyBytes, dataBytes));
      return new FileInvalidationDataInfo(
          cacheKey, sparselyAggregateWriteStatuses(writeStatuses), exists, mtsv, realRootedPath);
    }

    /**
     * Adds information about the invalidation data of {@link #parentRootedPath}.
     *
     * <p>This is called at most once but might not be called at all if {@link #parentRootedPath}
     * refers to a constant path.
     */
    private void addParent(FileInvalidationDataInfo parent) {
      long parentMtsv = parent.mtsv();
      if (parentMtsv != LongVersionGetter.MINIMAL) {
        data.setParentMtsv(parentMtsv);
      }
      updateMtsvIfGreater(parentMtsv);
      writeStatuses.add(parent.writeStatus());
    }

    /**
     * Incorporates information about a symlink parent.
     *
     * <p>If a symlink is present, it's possible that combining the symlink with {@link
     * #parentRootedPath} points to a file who's parent directory hasn't been resolved. Resolution
     * of that parent directory results in {@code parentInfo}.
     *
     * <p>If the symlink points to a symlink, it's possible for that symlink to point to a file
     * having a yet another parent directory that has to be resolved again. In that case this method
     * would be called again.
     */
    private void addSymlinkParentInfo(FileInvalidationDataInfo parentInfo) {
      updateMtsvIfGreater(parentInfo.mtsv());
      writeStatuses.add(parentInfo.writeStatus());
    }

    private void updateMtsvIfGreater(long version) {
      if (version > mtsv) {
        mtsv = version;
      }
    }

    private Symlink.Builder addSymlinksBuilder() {
      return data.addSymlinksBuilder();
    }
  }

  /** Resolves the parent then resolves symlinks if {@code unresolvedLinkTarget} is non-null. */
  private ListenableFuture<Void> fullyResolvePath(
      @Nullable PathFragment unresolvedLinkTarget, FileInvalidationDataUploader uploader) {
    var pathResolver = new PathResolver(unresolvedLinkTarget, uploader);
    switch (registerDependency(FileValue.key(uploader.parentRootedPath))) {
      case FileDataInfo parentData:
        return pathResolver.apply(parentData);
      case FutureFileDataInfo futureParentData:
        return Futures.transformAsync(futureParentData, pathResolver, directExecutor());
    }
  }

  /**
   * Waits for {@link FileInvalidationDataUploader#parentRootedPath} to be resolved (signalled
   * through the {@link #apply} callback) and starts symlink resolution if there is a symlink.
   */
  class PathResolver implements AsyncFunction<FileDataInfo, Void> {
    /** Symlink's target path. */
    @Nullable // non-null if there is a symlink
    private final PathFragment unresolvedLinkTarget;

    private final FileInvalidationDataUploader uploader;

    private PathResolver(
        @Nullable PathFragment unresolvedLinkTarget, FileInvalidationDataUploader uploader) {
      this.unresolvedLinkTarget = unresolvedLinkTarget;
      this.uploader = uploader;
    }

    @Override
    public ListenableFuture<Void> apply(FileDataInfo parentData) {
      RootedPath realParentPath;
      switch (parentData) {
        case CONSTANT_FILE:
          // Assumes that BundledFileSystem does not symlink outside of BundledFileSystem.
          realParentPath = uploader.parentRootedPath;
          break;
        case FileInvalidationDataInfo parentReference:
          uploader.addParent(parentReference);
          // If the parent folder doesn't exist, unresolvedLinkTarget will be null.
          realParentPath = parentReference.realPath();
          break;
      }

      if (unresolvedLinkTarget == null) {
        return immediateVoidFuture(); // No symlink processing needed.
      }

      Path linkPath; // Real path to the symlink.
      if (realParentPath.equals(uploader.parentRootedPath)) {
        linkPath = uploader.rootedPath.asPath();
      } else {
        linkPath =
            realParentPath
                .asPath()
                .getRelative(uploader.rootedPath.getRootRelativePath().getBaseName());
      }
      return processSymlinks(realParentPath, linkPath, unresolvedLinkTarget, uploader);
    }
  }

  /**
   * Recursively processes symlinks.
   *
   * <p>Requires that there are no symlink cycles (though ancestor references are benign). This is
   * assumed to hold for builds that succeed.
   *
   * @param parentRootedPath the real parent of the symlink
   * @param linkPath the real path to the symlink
   * @param link the target path contents of the symlink
   * @param uploader original uploader instance for the path resolution encountering this symlink
   */
  private ListenableFuture<Void> processSymlinks(
      RootedPath parentRootedPath,
      Path linkPath,
      PathFragment link,
      FileInvalidationDataUploader uploader) {
    if (link.isAbsolute()) {
      if (isInTest()) {
        // Test environments may use absolute symlinks, which aren't allowed in production
        // environments with analysis caching. Skips further dependency resolution for those.
        return immediateVoidFuture();
      }
      throw new IllegalStateException(
          String.format("Absolute symlink not permitted: %s contained %s", linkPath, link));
    }
    Symlink.Builder symlinkData = uploader.addSymlinksBuilder().setContents(link.getPathString());
    PathFragment linkParent = parentRootedPath.getRootRelativePath();
    PathFragment unresolvedTarget = linkParent.getRelative(link);

    // Assumes that there are no external symlinks, e.g. ones that go above root.
    checkArgument(
        !unresolvedTarget.containsUplevelReferences(),
        "symlink link above root for %s : %s = (%s) + (%s)",
        parentRootedPath,
        unresolvedTarget,
        linkParent,
        link);

    try {
      // Includes the version of the link itself in the MTSV.
      uploader.updateMtsvIfGreater(getVersion(linkPath, /* exists= */ true));
    } catch (IOException e) {
      return immediateFailedFuture(e);
    }

    if (unresolvedTarget.isEmpty()) {
      // It was a symlink to root. It's unclear how this ever useful, but it's not illegal. No
      // resolution required.
      return immediateVoidFuture();
    }

    PathFragment unresolvedTargetParent = unresolvedTarget.getParentDirectory();

    if (linkParent.startsWith(unresolvedTargetParent)) {
      // Any ancestor directories of the fully resolved `linkParent` are already resolved so
      // there's no further ancestor resolution here.
      return processSymlinkTarget(
          toRootedPath(parentRootedPath.getRoot(), unresolvedTarget), uploader);
    }

    var parentProcessor = new SymlinkParentProcessor(parentRootedPath, link, uploader, symlinkData);

    // The parent path was changed by the link so it needs to be newly resolved.
    switch (checkNotNull(
        registerDependency(
            FileValue.key(toRootedPath(parentRootedPath.getRoot(), unresolvedTargetParent))),
        unresolvedTargetParent)) {
      case FileDataInfo data:
        return parentProcessor.apply(data);
      case FutureFileDataInfo future:
        return Futures.transformAsync(future, parentProcessor, directExecutor());
    }
  }

  private ListenableFuture<Void> processSymlinkTarget(
      RootedPath resolvedSymlinkPath, FileInvalidationDataUploader uploader) {
    InMemoryNodeEntry nodeEntry = graph.getIfPresent(resolvedSymlinkPath);
    if (nodeEntry == null) {
      return immediateFailedFuture(new MissingSkyframeEntryException(resolvedSymlinkPath));
    }
    var symlinkValue = (FileStateValue) nodeEntry.getValue();
    if (!symlinkValue.getType().equals(SYMLINK)) {
      // We've come full circle back to the initial, fully resolved, FileValue. So there's no
      // additional bookkeeping needed.
      return immediateVoidFuture();
    }
    return processSymlinks(
        resolvedSymlinkPath.getParentDirectory(),
        resolvedSymlinkPath.asPath(),
        symlinkValue.getSymlinkTarget(),
        uploader);
  }

  /**
   * Waits for information about a symlink's real parent, signalled through the {@link #apply}
   * callback.
   *
   * <p>Once the real parent is known, syntactically combines it with the symlink target path. Then,
   * continues resolving the new path, potentially triggering recursion if the target is another
   * symlink.
   */
  private final class SymlinkParentProcessor implements AsyncFunction<FileDataInfo, Void> {
    private final RootedPath parentPath;
    private final PathFragment link;
    private final FileInvalidationDataUploader uploader;
    private final Symlink.Builder symlinkData;

    private SymlinkParentProcessor(
        RootedPath parentPath,
        PathFragment link,
        FileInvalidationDataUploader uploader,
        Symlink.Builder symlinkData) {
      this.parentPath = parentPath;
      this.link = link;
      this.uploader = uploader;
      this.symlinkData = symlinkData;
    }

    @Override
    public ListenableFuture<Void> apply(FileDataInfo realParentData) {
      RootedPath resolvedParentRootedPath;
      switch (realParentData) {
        case CONSTANT_FILE:
          // Assumes that symlinks in BundledFileSystem do not escape.
          resolvedParentRootedPath = parentPath;
          break;
        case FileInvalidationDataInfo parentReference:
          uploader.addSymlinkParentInfo(parentReference);
          long parentMtsv = parentReference.mtsv();
          if (parentMtsv != LongVersionGetter.MINIMAL) {
            symlinkData.setParentMtsv(parentMtsv);
          }
          if (!parentReference.exists()) {
            // The parent folder doesn't exist so further resolution of the symlink is moot.
            return immediateVoidFuture();
          }
          resolvedParentRootedPath = parentReference.realPath();
          break;
      }
      return processSymlinkTarget(
          toRootedPath(
              resolvedParentRootedPath.getRoot(),
              resolvedParentRootedPath.getRootRelativePath().getRelative(link.getBaseName())),
          uploader);
    }
  }

  private ListingDataInfoOrFuture populateFutureListingDataInfo(FutureListingDataInfo future) {
    RootedPath rootedPath = future.key().argument();
    if (rootedPath.getRoot().getFileSystem() instanceof BundledFileSystem) {
      return future.completeWith(CONSTANT_LISTING); // This listing doesn't change.
    }
    var handler = new ListingFileHandler(rootedPath);
    switch (registerDependency(FileValue.key(rootedPath))) {
      case FileDataInfo info:
        return future.completeWith(handler.apply(info));
      case FutureFileDataInfo futureInfo:
        return future.completeWith(Futures.transformAsync(futureInfo, handler, directExecutor()));
    }
  }

  private class ListingFileHandler implements AsyncFunction<FileDataInfo, ListingDataInfo> {
    private final RootedPath rootedPath;

    private ListingFileHandler(RootedPath rootedPath) {
      this.rootedPath = rootedPath;
    }

    /**
     * Incorporates information from the file associated with the directory.
     *
     * <p>This code assumes that the directory exists, as does {@link
     * com.google.devtools.build.lib.skyframe.DirectoryListingValue#key}.
     */
    @Override
    public ListenableFuture<ListingDataInfo> apply(FileDataInfo info) {
      DirectoryListingInvalidationData.Builder data = DirectoryListingInvalidationData.newBuilder();
      var writeStatuses = new ArrayList<WriteStatus>();
      long fileMtsv;
      RootedPath realPath;
      switch (info) {
        case CONSTANT_FILE: // reached only for the root directory
          realPath = rootedPath;
          fileMtsv = LongVersionGetter.MINIMAL;
          break;
        case FileInvalidationDataInfo fileInfo:
          writeStatuses.add(fileInfo.writeStatus());
          fileMtsv = fileInfo.mtsv();
          if (fileMtsv != LongVersionGetter.MINIMAL) {
            data.setFileMtsv(fileMtsv);
          }
          realPath = fileInfo.realPath();
          break;
      }

      ListenableFuture<Long> dirMtsvFuture =
          Futures.submit(
              (Callable<Long>)
                  () -> {
                    return versionGetter.getDirectoryListingVersion(realPath.asPath());
                  },
              ForkJoinPool.commonPool());

      return Futures.transform(
          dirMtsvFuture,
          dirMtsv -> {
            long mtsv = max(dirMtsv, fileMtsv);

            String cacheKey =
                computeCacheKey(rootedPath.getRootRelativePath(), mtsv, DIRECTORY_KEY_DELIMITER);
            KeyBytesProvider keyBytes = getKeyBytes(cacheKey, data::setOverflowKey);
            byte[] dataBytes = data.build().toByteArray();
            writeStatuses.add(writer.put(keyBytes, dataBytes));
            return new ListingInvalidationDataInfo(
                cacheKey, sparselyAggregateWriteStatuses(writeStatuses));
          },
          directExecutor());
    }
  }

  NodeDataInfoOrFuture populateFutureNodeDataInfo(FutureNodeDataInfo future) {
    AbstractNestedFileOpNodes node = future.key();
    var dependencyHandler = new NodeDependencyHandler();

    // Loops through all node dependencies, registering them with the dependencyHandler. The
    // dependencyHandler triggers recursive registration, keeping track of immediate results and
    // any futures.
    for (int i = 0; i < node.analysisDependenciesCount(); i++) {
      switch (node.getAnalysisDependency(i)) {
        case FileKey fileKey:
          dependencyHandler.addFileKey(fileKey);
          break;
        case DirectoryListingKey listingKey:
          dependencyHandler.addListingKey(listingKey);
          break;
        case AbstractNestedFileOpNodes nestedKeys:
          dependencyHandler.addNodeKey(nestedKeys);
          break;
      }
    }

    switch (node) {
      case NestedFileOpNodes plainNodes:
        break;
      case NestedFileOpNodesWithSource withSource:
        dependencyHandler.setSourceFile(withSource.source());
        break;
    }

    var allFutures = dependencyHandler.getCombinedFutures();
    if (allFutures.isEmpty()) {
      NodeDataInfo result;
      try {
        result = dependencyHandler.call();
      } catch (ExecutionException | IOException e) {
        // Only thrown when calling Future.get, but none should be present if this is reached.
        throw new IllegalStateException("unexpected failure", e);
      }
      return future.completeWith(result);
    }
    return future.completeWith(
        Futures.whenAllComplete(allFutures).call(dependencyHandler, directExecutor()));
  }

  static OutputStream getCompressedOutputStream(OutputStream outputStream) throws IOException {
    // The default level and the fastest level (-7) results in 35% and 19% wall time overhead when
    // not using a threshold to compress, the default level provided a 2x better compression. Since
    // we do use a threshold and there is no wall time regression, we favor the better compression
    // ratio.
    return new ZstdOutputStream(outputStream);
  }

  /**
   * Accepts all the dependencies associated with a node, registers their serialization and waits
   * for processing to complete, signalled through the {@link #call} callback.
   *
   * <p>Once processing is complete and all keys are known, uploads the node value. {@link
   * #computeNodeBytes} defines the wire format of nodes.
   */
  class NodeDependencyHandler implements Callable<NodeDataInfo> {
    private final TreeSet<String> fileKeys = new TreeSet<>();
    private final TreeSet<String> listingKeys = new TreeSet<>();
    private final TreeMap<PackedFingerprint, NodeInvalidationDataInfo> nodeDependencies =
        new TreeMap<>();
    @Nullable private FileDataInfoOrFuture sourceFileOrFuture;

    private final ArrayList<WriteStatus> writeStatuses = new ArrayList<>();

    private final ArrayList<FutureFileDataInfo> futureFileDataInfo = new ArrayList<>();
    private final ArrayList<FutureListingDataInfo> futureListingDataInfo = new ArrayList<>();
    private final ArrayList<FutureNodeDataInfo> futureNodeDataInfo = new ArrayList<>();

    @Override
    public NodeDataInfo call() throws ExecutionException, IOException {
      for (FutureFileDataInfo futureInfo : futureFileDataInfo) {
        addFileInfo(Futures.getDone(futureInfo));
      }
      for (FutureListingDataInfo futureInfo : futureListingDataInfo) {
        addListingInfo(Futures.getDone(futureInfo));
      }
      for (FutureNodeDataInfo futureInfo : futureNodeDataInfo) {
        addNodeInfo(Futures.getDone(futureInfo));
      }
      @Nullable String sourceFileKey = getSourceFileKey();

      if (fileKeys.isEmpty() && listingKeys.isEmpty() && sourceFileKey == null) {
        if (nodeDependencies.isEmpty()) {
          return CONSTANT_NODE; // None of the dependencies are relevant to invalidation.
        }
        // There are multiple ways that result could become unary here, even if `node` always has at
        // least 2 children. The following may reduce child count.
        // 1. TreeSet deduplication.
        // 2. Constant references.
        // 3. NestedFileOpNodes with the same fingerprints.
        if (nodeDependencies.size() == 1) {
          // It ended up as a node wrapping another node. Discards the wrapper.
          //
          // TODO: b/364831651 - consider additional special casing for unary file or listing
          // dependencies.
          return nodeDependencies.values().iterator().next();
        }
      }

      byte[] nodeBytes = computeNodeBytes(nodeDependencies, fileKeys, listingKeys, sourceFileKey);
      byte[] maybeCompressedBytes = nodeBytes;
      if (nodeBytes.length >= COMPRESSION_NUM_BYTES_THRESHOLD) {
        maybeCompressedBytes = compressBytes(nodeBytes);
      }
      PackedFingerprint key = writer.fingerprint(maybeCompressedBytes);
      writeStatuses.add(writer.put(key, maybeCompressedBytes));
      return new NodeInvalidationDataInfo(key, sparselyAggregateWriteStatuses(writeStatuses));
    }

    private void addFileKey(FileKey fileKey) {
      switch (registerDependency(fileKey)) {
        case FileDataInfo info:
          addFileInfo(info);
          break;
        case FutureFileDataInfo futureInfo:
          futureFileDataInfo.add(futureInfo);
          break;
      }
    }

    private void addFileInfo(FileDataInfo info) {
      switch (info) {
        case CONSTANT_FILE:
          break;
        case FileInvalidationDataInfo fileInfo:
          fileKeys.add(fileInfo.cacheKey());
          writeStatuses.add(fileInfo.writeStatus());
          break;
      }
    }

    private void addListingKey(DirectoryListingKey listingKey) {
      switch (registerDependency(listingKey)) {
        case ListingDataInfo info:
          addListingInfo(info);
          break;
        case FutureListingDataInfo futureInfo:
          futureListingDataInfo.add(futureInfo);
          break;
      }
    }

    private void addListingInfo(ListingDataInfo info) {
      switch (info) {
        case CONSTANT_LISTING:
          break;
        case ListingInvalidationDataInfo listingInfo:
          listingKeys.add(listingInfo.cacheKey());
          writeStatuses.add(listingInfo.writeStatus());
          break;
      }
    }

    private void addNodeKey(AbstractNestedFileOpNodes nestedKeys) {
      switch (registerDependency(nestedKeys)) {
        case NodeDataInfo info:
          addNodeInfo(info);
          break;
        case FutureNodeDataInfo futureInfo:
          futureNodeDataInfo.add(futureInfo);
          break;
      }
    }

    private void addNodeInfo(NodeDataInfo info) {
      switch (info) {
        case CONSTANT_NODE:
          break;
        case NodeInvalidationDataInfo nodeInfo:
          nodeDependencies.put(nodeInfo.cacheKey(), nodeInfo);
          writeStatuses.add(nodeInfo.writeStatus());
          break;
      }
    }

    private void setSourceFile(FileKey sourceFile) {
      checkState(
          sourceFileOrFuture == null,
          "Attempting to set source file to %s, but it was already set to %s",
          sourceFile,
          sourceFileOrFuture);
      this.sourceFileOrFuture = registerDependency(sourceFile);
    }

    private ImmutableList<ListenableFuture<?>> getCombinedFutures() {
      var combined =
          ImmutableList.<ListenableFuture<?>>builder()
              .addAll(futureFileDataInfo)
              .addAll(futureListingDataInfo)
              .addAll(futureNodeDataInfo);
      switch (sourceFileOrFuture) {
        case null -> {}
        case FileDataInfo unusedSource -> {}
        case FutureFileDataInfo futureSource -> combined.add(futureSource);
      }
      return combined.build();
    }

    private byte[] compressBytes(byte[] nodeBytes) throws IOException {
      ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
      MagicBytes.writeMagicBytes(outputStream);
      try (OutputStream compressedBytesStream =
          FileDependencySerializer.getCompressedOutputStream(outputStream)) {
        compressedBytesStream.write(nodeBytes);
      }
      return outputStream.toByteArray();
    }

    @Nullable
    private String getSourceFileKey() throws ExecutionException {
      if (sourceFileOrFuture == null) {
        return null;
      }
      return switch (switch (sourceFileOrFuture) {
        case FileDataInfo sourceInfo -> sourceInfo;
        case FutureFileDataInfo futureSourceInfo -> Futures.getDone(futureSourceInfo);
      }) {
        case CONSTANT_FILE -> null;
        case FileInvalidationDataInfo fileInfo -> {
          writeStatuses.add(fileInfo.writeStatus());
          yield fileInfo.cacheKey();
        }
      };
    }
  }

  /**
   * Computes a canonical byte representation of the node.
   *
   * <p>Logically, a node is a set of string file or listing keys, as described at {@link
   * FileInvalidationData} and {@link DirectoryListingInvalidationData}, respectively, and a set of
   * {@link NestedFileOpNodes} fingerprints. Its byte representation is specified as follows.
   *
   * <ol>
   *   <li>The count of nested nodes, as a proto-encoded int.
   *   <li>The count of file keys, as a proto-encoded int.
   *   <li>The count of listing keys, as a proto-encoded int.
   *   <li>A proto-encoded boolean, true if a source file key is present and false otherwise.
   *   <li>Sorted and deduplicated, fingerprints of the {@link NestedFileOpNodes} byte
   *       representations.
   *   <li>Sorted and deduplicated, proto-encoded strings of the file keys.
   *   <li>Sorted and deduplicated, proto-encoded strings of the listing keys.
   *   <li>Proto-encoded string of the source file key, if applicable.
   * </ol>
   *
   * <p>More compact formats are possible, but this reduces the complexity of the deserializer.
   */
  @VisibleForTesting
  static byte[] computeNodeBytes(
      Map<PackedFingerprint, NodeInvalidationDataInfo> nodeDependencies,
      Set<String> fileKeys,
      Set<String> listingKeys,
      @Nullable String sourceFileKey) {
    try {
      var bytesOut = new ByteArrayOutputStream();
      var codedOut = CodedOutputStream.newInstance(bytesOut);
      codedOut.writeInt32NoTag(nodeDependencies.size());
      codedOut.writeInt32NoTag(fileKeys.size());
      codedOut.writeInt32NoTag(listingKeys.size());
      codedOut.writeBoolNoTag(sourceFileKey != null);
      for (PackedFingerprint fp : nodeDependencies.keySet()) {
        fp.writeTo(codedOut);
      }
      for (String key : fileKeys) {
        codedOut.writeStringNoTag(key);
      }
      for (String key : listingKeys) {
        codedOut.writeStringNoTag(key);
      }
      if (sourceFileKey != null) {
        codedOut.writeStringNoTag(sourceFileKey);
      }
      codedOut.flush();
      bytesOut.flush();
      return bytesOut.toByteArray();
    } catch (IOException e) {
      throw new AssertionError("Unexpected IOException from ByteArrayOutputStream", e);
    }
  }

  private long getVersion(RootedPath rootedPath, boolean exists) throws IOException {
    return getVersion(rootedPath.asPath(), exists);
  }

  private long getVersion(Path path, boolean exists) throws IOException {
    return exists
        ? versionGetter.getFilePathOrSymlinkVersion(path)
        : versionGetter.getNonexistentPathVersion(path);
  }

  private KeyBytesProvider getKeyBytes(String cacheKey, Consumer<String> overflowConsumer) {
    if (cacheKey.length() > MAX_KEY_LENGTH) {
      overflowConsumer.accept(cacheKey);
      return writer.fingerprint(cacheKey.getBytes(UTF_8));
    }
    return new StringKey(cacheKey);
  }
}
