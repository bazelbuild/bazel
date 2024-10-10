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
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.actions.FileStateType.SYMLINK;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.DIRECTORY_KEY_DELIMITER;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.FILE_KEY_DELIMITER;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.MAX_KEY_LENGTH;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.MTSV_SENTINEL;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.computeCacheKey;
import static com.google.devtools.build.lib.vfs.RootedPath.toRootedPath;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.BundledFileSystem;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.KeyBytesProvider;
import com.google.devtools.build.lib.skyframe.serialization.StringKey;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataReference.DirectoryInvalidationDataReference;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataReference.FileInvalidationDataReference;
import com.google.devtools.build.lib.skyframe.serialization.proto.DirectoryListingInvalidationData;
import com.google.devtools.build.lib.skyframe.serialization.proto.FileInvalidationData;
import com.google.devtools.build.lib.skyframe.serialization.proto.Symlink;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.Version;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Records {@link FileValue.Key} and {@link DirectoryListingValue.Key} invalidation information. */
final class FileDependencySerializer {

  private final VersionNumberExtractor versionExtractor;
  private final InMemoryGraph graph;
  private final FingerprintValueService fingerprintValueService;

  private final ConcurrentHashMap<FileValue.Key, FileInvalidationDataReference> fileReferences =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<DirectoryListingValue.Key, DirectoryInvalidationDataReference>
      directoryReferences = new ConcurrentHashMap<>();

  interface VersionNumberExtractor {
    long getVersionNumber(Version version);
  }

  FileDependencySerializer(
      VersionNumberExtractor versionExtractor,
      InMemoryGraph graph,
      FingerprintValueService fingerprintValueService) {
    this.versionExtractor = versionExtractor;
    this.graph = graph;
    this.fingerprintValueService = fingerprintValueService;
  }

  /**
   * Stores data about a {@code key} and its transitive dependencies in {@link
   * #fingerprintValueService} to be used for invalidation.
   *
   * <p>The resulting data can be embedded in reverse deps of {@code key} and used to invalidate
   * them by checking against a list of changed files.
   *
   * <p>See comments at {@link FileInvalidationData} for more details about the data being
   * persisted.
   */
  @Nullable // null if `key` isn't relevant to invalidation
  FileInvalidationDataReference registerDependency(FileValue.Key key) {
    RootedPath rootedPath = key.argument();
    if (rootedPath.getRoot().getFileSystem() instanceof BundledFileSystem) {
      return null; // This file doesn't change.
    }
    if (rootedPath.getRootRelativePath().isEmpty()) {
      return null; // Assumes the root folder doesn't change.
    }

    // Initialization of `reference` occurs in three steps.
    //
    // 1. Construction: This happens within `computeIfAbsent` and is context-independent, ensuring
    //    no extra allocations on a cache hit.
    //
    // 2. Essential Initialization: Performed outside `computeIfAbsent` in a synchronized block,
    //    this step provides context-dependent data immediately required by other consumers of
    //    `reference`. Being outside of `computeIfAbsent` allows sharing intermediate computation
    //    results with step 3 without additional allocations when there's a cache hit.
    //
    // 3. Future Completion: This is the most computationally intensive step, involving recursion.
    //    However, other consumers of `reference` only have an asynchronous dependency on its
    //    resolution, allowing them to proceed without waiting for this step to complete.
    FileInvalidationDataReference reference =
        fileReferences.computeIfAbsent(key, unused -> new FileInvalidationDataReference());
    // Uses double-checked locking to determine ownership of `reference`.
    if (reference.getCacheKey() != null) {
      return reference;
    }

    PathFragment path;
    FileValue value;
    synchronized (reference) {
      if (reference.getCacheKey() != null) {
        return reference; // Another thread took ownership.
      }

      path = rootedPath.getRootRelativePath();
      InMemoryNodeEntry node = checkNotNull(graph.getIfPresent(key), key);
      value = (FileValue) node.getValue();
      long mtsv = versionExtractor.getVersionNumber(node.getMaxTransitiveSourceVersion());
      reference.populate(
          mtsv, value.realRootedPath(rootedPath), computeCacheKey(path, mtsv, FILE_KEY_DELIMITER));
    }
    // If this is reached, this thread owns `reference` and must complete its future.
    boolean writeStatusSet = false;
    try {
      var data = FileInvalidationData.newBuilder();
      var writeStatuses = new ArrayList<ListenableFuture<Void>>();
      FileInvalidationDataReference parentReference =
          registerDependency(FileValue.key(rootedPath.getParentDirectory()));
      if (parentReference != null) {
        long mtsv = parentReference.getMtsv();
        if (mtsv != MTSV_SENTINEL) {
          data.setParentMtsv(parentReference.getMtsv());
        }
        writeStatuses.add(parentReference);
      }

      if (value.isSymlink()) {
        processSymlinks(
            parentReference == null
                ? rootedPath.getParentDirectory()
                : parentReference.getRealPath(),
            value.getUnresolvedLinkTarget(),
            data,
            writeStatuses);
      }

      KeyBytesProvider keyBytes = getKeyBytes(reference.getCacheKey(), data::setOverflowKey);
      writeStatuses.add(fingerprintValueService.put(keyBytes, data.build().toByteArray()));
      reference.setWriteStatus(
          writeStatuses.size() == 1
              ? writeStatuses.get(0)
              : Futures.whenAllSucceed(writeStatuses).call(() -> null, directExecutor()));
      writeStatusSet = true;
    } finally {
      if (!writeStatusSet) {
        reference.setUnexpectedlyUnsetError();
      }
    }
    return reference;
  }

  @Nullable // null if `key` isn't relevant to invalidation
  DirectoryInvalidationDataReference registerDependency(DirectoryListingValue.Key key) {
    RootedPath rootedPath = key.argument();
    if (rootedPath.getRoot().getFileSystem() instanceof BundledFileSystem) {
      return null; // This directory doesn't change.
    }

    DirectoryInvalidationDataReference reference =
        directoryReferences.computeIfAbsent(
            key, unused -> new DirectoryInvalidationDataReference());
    // Uses double-checked locking to determine ownership of `reference`.
    if (reference.getCacheKey() != null) {
      return reference;
    }

    synchronized (reference) {
      if (reference.getCacheKey() != null) {
        return reference; // Another thread took ownership.
      }

      InMemoryNodeEntry node = checkNotNull(graph.getIfPresent(key), key);
      long mtsv = versionExtractor.getVersionNumber(node.getMaxTransitiveSourceVersion());
      reference.populate(
          computeCacheKey(rootedPath.getRootRelativePath(), mtsv, DIRECTORY_KEY_DELIMITER));
    }
    // If this is reached, this thread owns `reference` and must complete it's future.
    boolean writeStatusSet = false;
    try {
      DirectoryListingInvalidationData.Builder data = DirectoryListingInvalidationData.newBuilder();
      FileInvalidationDataReference fileReference =
          registerDependency(FileValue.key(key.argument()));
      if (fileReference != null) {
        long mtsv = fileReference.getMtsv();
        if (mtsv != MTSV_SENTINEL) {
          data.setFileMtsv(fileReference.getMtsv());
        }
      }
      KeyBytesProvider keyBytes = getKeyBytes(reference.getCacheKey(), data::setOverflowKey);
      ListenableFuture<Void> writeStatus =
          fingerprintValueService.put(keyBytes, data.build().toByteArray());
      reference.setWriteStatus(
          fileReference == null
              ? writeStatus
              : Futures.whenAllSucceed(fileReference, writeStatus)
                  .call(() -> null, directExecutor()));
      writeStatusSet = true;
    } finally {
      if (!writeStatusSet) {
        reference.setUnexpectedlyUnsetError();
      }
    }
    return reference;
  }

  /**
   * Requires that there are no symlink cycles (though ancestor references are benign).
   *
   * <p>This is assumed to hold for builds that succeed.
   */
  private void processSymlinks(
      RootedPath parentRootedPath,
      PathFragment link,
      FileInvalidationData.Builder data,
      List<ListenableFuture<Void>> ancestorDeps) {
    while (true) {
      Symlink.Builder symlinkData = data.addSymlinksBuilder().setContents(link.getPathString());
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
      if (unresolvedTarget.isEmpty()) {
        // It was a symlink to root. It's unclear how this ever useful, but it's not illegal. No
        // resolution required.
        return;
      }
      PathFragment unresolvedTargetParent = unresolvedTarget.getParentDirectory();
      RootedPath symbolicPath = toRootedPath(parentRootedPath.getRoot(), unresolvedTarget);
      RootedPath resolvedSymlinkPath; // path pointed to by the symlink, after resolving parent
      if (linkParent.startsWith(unresolvedTargetParent)) {
        // Any ancestor directories of the fully resolved `linkParent` are already resolved so
        // there's no need for further resolution.
        resolvedSymlinkPath = symbolicPath;
      } else {
        // The parent path was changed by the link so it needs to be newly resolved.
        FileInvalidationDataReference parentReference =
            checkNotNull(
                registerDependency(
                    FileValue.key(
                        toRootedPath(parentRootedPath.getRoot(), unresolvedTargetParent))),
                unresolvedTargetParent);
        ancestorDeps.add(parentReference);
        long parentMtsv = parentReference.getMtsv();
        if (parentMtsv != MTSV_SENTINEL) {
          symlinkData.setParentMtsv(parentReference.getMtsv());
        }
        RootedPath resolvedParentRootedPath = parentReference.getRealPath();
        resolvedSymlinkPath =
            toRootedPath(
                resolvedParentRootedPath.getRoot(),
                resolvedParentRootedPath.getRootRelativePath().getRelative(link.getBaseName()));
      }

      var symlinkValue =
          (FileStateValue)
              checkNotNull(graph.getIfPresent(resolvedSymlinkPath), resolvedSymlinkPath).getValue();
      if (!symlinkValue.getType().equals(SYMLINK)) {
        return;
      }
      parentRootedPath = resolvedSymlinkPath.getParentDirectory();
      link = symlinkValue.getSymlinkTarget();
    }
  }

  private KeyBytesProvider getKeyBytes(String cacheKey, Consumer<String> overflowConsumer) {
    if (cacheKey.length() > MAX_KEY_LENGTH) {
      overflowConsumer.accept(cacheKey);
      return fingerprintValueService.fingerprint(cacheKey.getBytes(UTF_8));
    }
    return new StringKey(cacheKey);
  }
}
