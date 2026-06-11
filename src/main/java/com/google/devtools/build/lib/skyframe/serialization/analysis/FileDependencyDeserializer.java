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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.DIRECTORY_KEY_DELIMITER;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.FILE_KEY_DELIMITER;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.MAX_KEY_LENGTH;
import static com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencyKeySupport.computeCacheKey;
import static com.google.devtools.build.lib.vfs.PathFragment.SEPARATOR_CHAR;
import static com.google.protobuf.ExtensionRegistry.getEmptyRegistry;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.luben.zstd.RecyclingBufferPool;
import com.github.luben.zstd.ZstdInputStream;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.concurrent.SettableFutureKeyedValue;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.Fingerprinter;
import com.google.devtools.build.lib.skyframe.serialization.KeyBytesProvider;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.StringKey;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencies.AvailableFileDependencies;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FileDependencies.MissingFileDependencies;
import com.google.devtools.build.lib.skyframe.serialization.proto.DirectoryListingInvalidationData;
import com.google.devtools.build.lib.skyframe.serialization.proto.FileInvalidationData;
import com.google.devtools.build.lib.skyframe.serialization.proto.Symlink;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executor;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import javax.annotation.Nullable;

/**
 * Deserializes dependency information persisted by {@link FileDependencySerializer}.
 *
 * <p>Fetching a file dependency is a mostly linear asynchronous state machine that performs actions
 * then waits in an alternating manner.
 *
 * <ol>
 *   <li>Request the data for a given key.
 *   <li>{@link WaitForFileInvalidationData}.
 *   <li>Request the data for the parent directory (a recursive call).
 *   <li>{@link WaitForParent}.
 *   <li>Process any symlinks, resolving symlink parents as needed.
 *   <li>{@link WaitForSymlinkParent}.
 *   <li>Processing symlinks repeats for all the symlinks associated with an entry.
 * </ol>
 *
 * <p>A similar, but simpler state machine is used for directory listings.
 *
 * <ol>
 *   <li>Request the data for a given key.
 *   <li>{@link WaitForListingInvalidationData}.
 *   <li>Request the file data corresponding to the directory (delegating to {@link
 *       #getFileDependencies}).
 *   <li>{@link WaitForListingFileDependencies}.
 *   <li>Create and cache the {@link ListingDependencies} instance.
 * </ol>
 */
final class FileDependencyDeserializer {
  private static final OsPathPolicy OS = OsPathPolicy.getFilePathOs();

  /** Singleton representing the root file. */
  static final FileDependencies ROOT_FILE = FileDependencies.builder("").build();

  private final Executor executor;
  private final Fingerprinter fingerprinter;

  /**
   * A cache for {@link FileDependencies}, primarily for deduplication.
   *
   * <p>The cache keys are as described at {@link FileInvalidationData}. We can potentially strip
   * the version information here, but keeping the version enables a single {@link
   * FileDependencyDeserializer} instance to be shared across disparate builds.
   *
   * <p>While in-flight, the value has type {@link FutureFileDependencies}, which is replaced by
   * {@link FileDependencies} once the computation completes.
   *
   * <p>References to {@link FileDependencies} form DAGs where certain top-level entries are
   * retained by the {@code SkyValue}s that depend on them. When all such associated {@code
   * SkyValue}s are invalidated, the dependency information becomes eligible for GC.
   */
  private final DependencyMap<
          String, FileDependenciesOrFuture, FileDependencies, FutureFileDependencies>
      fileCache =
          new DependencyMap<>(
              Caffeine.newBuilder().weakValues().<String, FileDependenciesOrFuture>build().asMap(),
              FutureFileDependencies::new,
              FutureFileDependencies.class,
              FileDependencyDeserializer.this::populateFutureFileDependencies);

  /**
   * A cache for {@link ListingDependencies}, primarily for deduplication.
   *
   * <p>This follows the design of {@link #fileCache} but is for directory listings.
   */
  private final DependencyMap<
          String, ListingDependenciesOrFuture, ListingDependencies, FutureListingDependencies>
      listingCache =
          new DependencyMap<>(
              Caffeine.newBuilder()
                  .weakValues()
                  .<String, ListingDependenciesOrFuture>build()
                  .asMap(),
              FutureListingDependencies::new,
              FutureListingDependencies.class,
              FileDependencyDeserializer.this::populateFutureListingDependencies);

  private final DependencyMap<
          PackedFingerprint,
          NestedDependenciesOrFuture,
          NestedDependencies,
          FutureNestedDependencies>
      nestedCache =
          new DependencyMap<>(
              Caffeine.newBuilder()
                  .weakValues()
                  .<PackedFingerprint, NestedDependenciesOrFuture>build()
                  .asMap(),
              FutureNestedDependencies::new,
              FutureNestedDependencies.class,
              FileDependencyDeserializer.this::populateFutureNestedDependencies);

  FileDependencyDeserializer(Executor executor, Fingerprinter fingerprinter) {
    this.executor = executor;
    this.fingerprinter = fingerprinter;
  }

  sealed interface FileDependenciesOrFuture permits FileDependencies, FutureFileDependencies {}

  /**
   * The main purpose of this class is to act as a {@link ListenableFuture<FileDependencies>}.
   *
   * <p>Its specific type is explicitly visible to clients to allow them to cleanly distinguish it
   * as a permitted subtype of {@link FileDependenciesOrFuture}.
   */
  static final class FutureFileDependencies
      extends SettableFutureKeyedValue<FutureFileDependencies, String, FileDependencies>
      implements FileDependenciesOrFuture {
    private FutureFileDependencies(String key, BiConsumer<String, FileDependencies> consumer) {
      super(key, consumer);
    }
  }

  /**
   * Reconstitutes the set of file dependencies associated with {@code key}.
   *
   * <p>Performs lookups and parent resolution (recursively) and symlink resolution to obtain all
   * files associated with {@code key} represented as {@link FileDependencies}.
   *
   * @param key as described in {@link FileInvalidationData}.
   * @return either an immediate {@link FileDependencies} instance or effectively a {@link
   *     ListenableFuture<FileDependencies>} instance.
   */
  FileDependenciesOrFuture getFileDependencies(String key, FingerprintValueStore store) {
    return fileCache.getValueOrFuture(key, store);
  }

  private FileDependenciesOrFuture populateFutureFileDependencies(
      FutureFileDependencies ownedFuture, FingerprintValueStore store) {
    return fetchInvalidationData(
        this::getKeyBytes, WaitForFileInvalidationData::new, ownedFuture, store);
  }

  sealed interface ListingDependenciesOrFuture
      permits ListingDependencies, FutureListingDependencies {}

  /**
   * The main purpose of this class is to act as a {@link ListenableFuture<ListingDependencies>}.
   *
   * <p>Its specific type is explicitly visible to clients to allow them to cleanly distinguish it
   * as a permitted subtype of {@link ListingDependenciesOrFuture}.
   */
  static final class FutureListingDependencies
      extends SettableFutureKeyedValue<FutureListingDependencies, String, ListingDependencies>
      implements ListingDependenciesOrFuture {
    private FutureListingDependencies(
        String key, BiConsumer<String, ListingDependencies> consumer) {
      super(key, consumer);
    }
  }

  /**
   * Deserializes the resolved directory listing information associated with {@code key}.
   *
   * @param key should be as described at {@link DirectoryListingInvalidationData}.
   * @return either an immediate {@link ListingDependencies} instance or effectively a {@link
   *     ListenableFuture<ListingDependencies>} instance.
   */
  ListingDependenciesOrFuture getListingDependencies(String key, FingerprintValueStore store) {
    return listingCache.getValueOrFuture(key, store);
  }

  private ListingDependenciesOrFuture populateFutureListingDependencies(
      FutureListingDependencies ownedFuture, FingerprintValueStore store) {
    return fetchInvalidationData(
        this::getKeyBytes, WaitForListingInvalidationData::new, ownedFuture, store);
  }

  sealed interface NestedDependenciesOrFuture
      permits NestedDependencies, FutureNestedDependencies {}

  static final class FutureNestedDependencies
      extends SettableFutureKeyedValue<
          FutureNestedDependencies, PackedFingerprint, NestedDependencies>
      implements NestedDependenciesOrFuture {
    private FutureNestedDependencies(
        PackedFingerprint key, BiConsumer<PackedFingerprint, NestedDependencies> consumer) {
      super(key, consumer);
    }
  }

  /**
   * Retrieves the nested dependency information associated with {@code key}.
   *
   * <p>Like the other implementations, this can be thought of as a simple state machine. There's
   * one explicit state represented by {@link WaitForNestedNodeBytes}, which waits for the bytes
   * associated with {@code key}. There's a second implicit state that waits for child elements,
   * which may be files, listings or other nested nodes.
   *
   * @param key is a fingerprint of the byte representation described at {@link
   *     FileDependencySerializer#computeNodeBytes}.
   */
  NestedDependenciesOrFuture getNestedDependencies(
      PackedFingerprint key, FingerprintValueStore store) {
    return nestedCache.getValueOrFuture(key, store);
  }

  private NestedDependenciesOrFuture populateFutureNestedDependencies(
      FutureNestedDependencies ownedFuture, FingerprintValueStore store) {
    return fetchInvalidationData(
        Functions.identity(), WaitForNestedNodeBytes::new, ownedFuture, store);
  }

  // ---------- Begin FileDependencies deserialization implementation ----------

  private class WaitForFileInvalidationData implements AsyncFunction<byte[], FileDependencies> {
    private final String key;
    private final FingerprintValueStore store;

    private WaitForFileInvalidationData(String key, FingerprintValueStore store) {
      this.key = key;
      this.store = store;
    }

    @Override
    public ListenableFuture<FileDependencies> apply(byte[] bytes)
        throws InvalidProtocolBufferException {
      if (bytes == null) {
        return immediateFuture(FileDependencies.newMissingInstance());
      }

      var data = FileInvalidationData.parseFrom(bytes, getEmptyRegistry());
      if (data.hasOverflowKey() && !data.getOverflowKey().equals(key)) {
        return immediateFailedFuture(
            new SerializationException(
                String.format(
                    "Non-matching overflow key. This is possible if there is a key fingerprint"
                        + " collision. Expected %s got %s",
                    key, data)));
      }

      int pathBegin = key.indexOf(FILE_KEY_DELIMITER) + 1;
      int parentDirectoryEnd = key.lastIndexOf(SEPARATOR_CHAR);

      // `parentDirectoryEnd` is the index of the last `/`. This can be -1 if there is no `/` in
      // the key, or it can be less than `pathBegin` if the only `/`s are in the version part of
      // the key (e.g. "Ly/APA:WORKSPACE"). In either case, there is no parent directory to
      // resolve.
      if (parentDirectoryEnd < pathBegin) {
        checkState(
            !data.hasParentMtsv(), "no parent directory, but had parent MTSV %s, %s", key, data);
        return resolveParent(key, data, key.substring(pathBegin), /* parentKey= */ null, store);
      }

      String parentDirectory = key.substring(pathBegin, parentDirectoryEnd);
      String parentKey =
          computeCacheKey(
              parentDirectory,
              data.hasParentMtsv() ? data.getParentMtsv() : LongVersionGetter.MINIMAL,
              FILE_KEY_DELIMITER);
      String basename = key.substring(parentDirectoryEnd + 1);
      return resolveParent(key, data, basename, parentKey, store);
    }
  }

  private ListenableFuture<FileDependencies> resolveParent(
      String key,
      FileInvalidationData data,
      String basename,
      @Nullable String parentKey,
      FingerprintValueStore store) {
    var waitForParent = new WaitForParent(key, data, basename, store);

    if (parentKey == null) {
      return waitForParent.apply(/* parentOrMissing= */ null);
    }

    return switch (getFileDependencies(parentKey, store)) {
      case FileDependencies parent -> waitForParent.apply(parent);
      case FutureFileDependencies future ->
          Futures.transformAsync(future, waitForParent, directExecutor());
    };
  }

  private class WaitForParent implements AsyncFunction<FileDependencies, FileDependencies> {
    private final String key;
    private final FileInvalidationData data;
    private final String basename;
    private final FingerprintValueStore store;

    private WaitForParent(
        String key, FileInvalidationData data, String basename, FingerprintValueStore store) {
      this.key = key;
      this.data = data;
      this.basename = basename;
      this.store = store;
    }

    @Override
    public ListenableFuture<FileDependencies> apply(@Nullable FileDependencies parentOrMissing) {
      FileDependencies.Builder builder;
      String parentDirectory;
      switch (parentOrMissing) {
        case null -> {
          parentDirectory = null;
          builder = FileDependencies.builder(basename);
        }
        case AvailableFileDependencies parent -> {
          parentDirectory = parent.resolvedPath();
          builder =
              FileDependencies.builder(getRelative(parentDirectory, basename))
                  .addDependency(parent);
        }
        case MissingFileDependencies unused -> {
          return immediateFuture(FileDependencies.newMissingInstance());
        }
      }
      return processSymlinks(key, data, /* symlinkIndex= */ 0, parentDirectory, builder, store);
    }
  }

  /**
   * Processes any symlinks that my be present in {@code data}.
   *
   * @param key the main key that this symlink belongs to
   * @param parentDirectory the real directory containing the symlink
   */
  private ListenableFuture<FileDependencies> processSymlinks(
      String key,
      FileInvalidationData data,
      int symlinkIndex,
      @Nullable String parentDirectory, // null if root-level
      FileDependencies.Builder builder,
      FingerprintValueStore store) {
    if (symlinkIndex >= data.getSymlinksCount()) {
      return immediateFuture(builder.build());
    }

    Symlink link = data.getSymlinks(symlinkIndex);
    String linkContents = link.getContents();
    checkState(
        OS.getDriveStrLength(linkContents) == 0,
        "expected symlink contents to be a relative path: %s",
        data);
    // Combines the parent directory of the link with its contents and normalizes.
    String normalizedLinkTarget = getRelativeAndNormalize(parentDirectory, linkContents);
    String normalizedLinkParent = getParentDirectory(normalizedLinkTarget);

    if (!doesSymlinkParentNeedResolution(parentDirectory, normalizedLinkParent)) {
      checkState(
          !link.hasParentMtsv(),
          "no resolution needed for data=%s, symlinkIndex=%s, parentDirectory=%s,"
              + " normalizedLinkParent=%s but symlink had parent MTSV",
          data,
          symlinkIndex,
          parentDirectory,
          normalizedLinkParent);
      // Since `normalizedLinkParent` is already a real directory, `normalizedLinkTarget` is the
      // resolved symlink path.
      if (!normalizedLinkTarget.isEmpty()) { // Avoids adding root as a resolved path.
        builder.addPath(normalizedLinkTarget);
      }
      return processSymlinks(key, data, symlinkIndex + 1, normalizedLinkParent, builder, store);
    }

    String linkBasename = normalizedLinkTarget.substring(normalizedLinkParent.length() + 1);

    String newParentKey =
        computeCacheKey(
            normalizedLinkParent,
            link.hasParentMtsv() ? link.getParentMtsv() : LongVersionGetter.MINIMAL,
            FILE_KEY_DELIMITER);

    var waitForSymlinkParent =
        new WaitForSymlinkParent(key, data, symlinkIndex, linkBasename, builder, store);

    return switch (getFileDependencies(newParentKey, store)) {
      case FileDependencies resolvedParent -> waitForSymlinkParent.apply(resolvedParent);
      case FutureFileDependencies future ->
          Futures.transformAsync(future, waitForSymlinkParent, directExecutor());
    };
  }

  private class WaitForSymlinkParent implements AsyncFunction<FileDependencies, FileDependencies> {
    private final String key;
    private final FileInvalidationData data;
    private final int symlinkIndex;
    private final String linkBasename;
    private final FileDependencies.Builder builder;
    private final FingerprintValueStore store;

    private WaitForSymlinkParent(
        String key,
        FileInvalidationData data,
        int symlinkIndex,
        String linkBasename,
        FileDependencies.Builder builder,
        FingerprintValueStore store) {
      this.key = key;
      this.data = data;
      this.symlinkIndex = symlinkIndex;
      this.linkBasename = linkBasename;
      this.builder = builder;
      this.store = store;
    }

    @Override
    public ListenableFuture<FileDependencies> apply(FileDependencies parentOrMissing) {
      return switch (parentOrMissing) {
        case AvailableFileDependencies parent -> {
          String parentPath = parent.resolvedPath();
          builder.addPath(getRelative(parentPath, linkBasename)).addDependency(parent);
          yield processSymlinks(key, data, symlinkIndex + 1, parentPath, builder, store);
        }
        case MissingFileDependencies unused ->
            immediateFuture(FileDependencies.newMissingInstance());
      };
    }
  }

  private static String getRelative(@Nullable String parentDirectory, String basename) {
    if (parentDirectory == null) {
      return basename;
    }
    return parentDirectory + SEPARATOR_CHAR + basename;
  }

  private static String getRelativeAndNormalize(
      @Nullable String parentDirectory, String linkContents) {
    int normalizationLevel = OS.needsToNormalize(linkContents);
    return OS.normalize(getRelative(parentDirectory, linkContents), normalizationLevel);
  }

  @Nullable // null if `path` is at the root level
  private static String getParentDirectory(String path) {
    int lastSeparator = path.lastIndexOf(SEPARATOR_CHAR);
    if (lastSeparator == -1) { // no separator
      return null;
    }
    return path.substring(0, lastSeparator);
  }

  /**
   * Predicate specifying when a symlink parent directory needs further resolution.
   *
   * <p>A relative path specifier in symlink contents can modify the parent directory but it does
   * not always do so. For example, the symlink could point to a file in the same directory or the
   * symlink could point to a file in an ancestor directory. In both of these cases, the parent
   * directory is already fully resolved.
   *
   * @param previousParent the parent of the actual symlink itself. Null if the parent is actually
   *     the root directory.
   * @param newParent the parent directory after combining the symlink with {@code previousParent}.
   *     Null if the result is the root directory.
   */
  private static boolean doesSymlinkParentNeedResolution(
      @Nullable String previousParent, @Nullable String newParent) {
    if (newParent == null) {
      return false; // Already root level. No parent resolution needed.
    }
    if (previousParent == null) {
      return true; // No previousParent so resolution is needed.
    }
    // `newParent` is already a resolved path if it is the same as or an ancestor of the already
    // resolved `previousParent`.
    return !previousParent.startsWith(newParent);
  }

  // ---------- Begin ListingDependencies deserialization implementation ----------

  private class WaitForListingInvalidationData
      implements AsyncFunction<byte[], ListingDependencies> {
    private final String key;
    private final FingerprintValueStore store;

    private WaitForListingInvalidationData(String key, FingerprintValueStore store) {
      this.key = key;
      this.store = store;
    }

    @Override
    public ListenableFuture<ListingDependencies> apply(byte[] bytes)
        throws InvalidProtocolBufferException {
      if (bytes == null) {
        return immediateFuture(ListingDependencies.newMissingInstance());
      }

      var data = DirectoryListingInvalidationData.parseFrom(bytes, getEmptyRegistry());
      if (data.hasOverflowKey() && !data.getOverflowKey().equals(key)) {
        return immediateFailedFuture(
            new SerializationException(
                String.format(
                    "Non-matching overflow key. This is possible if there is a key fingerprint"
                        + " collision. Expected %s got %s",
                    key, data)));
      }

      int pathBegin = key.indexOf(DIRECTORY_KEY_DELIMITER) + 1;

      String path = key.substring(pathBegin);
      if (path.isEmpty()) {
        return immediateFuture(ListingDependencies.from(ROOT_FILE));
      }

      String fileKey =
          computeCacheKey(
              path,
              data.hasFileMtsv() ? data.getFileMtsv() : LongVersionGetter.MINIMAL,
              FILE_KEY_DELIMITER);
      return switch (getFileDependencies(fileKey, store)) {
        case FileDependencies dependencies ->
            immediateFuture(ListingDependencies.from(dependencies));
        case FutureFileDependencies future ->
            Futures.transform(future, ListingDependencies::from, directExecutor());
      };
    }
  }

  // ---------- Begin NestedDependencies deserialization implementation ----------

  private class WaitForNestedNodeBytes implements AsyncFunction<byte[], NestedDependencies> {
    private final FingerprintValueStore store;

    private WaitForNestedNodeBytes(PackedFingerprint unused, FingerprintValueStore store) {
      this.store = store;
    }

    /**
     * Parses the {@code bytes} to create a {@link NestedDependencies} instance.
     *
     * <p>Refer to comment at {@link FileDependencySerializer#computeNodeBytes} for the data format.
     * Uses delegation for children, which might not be completely resolved.
     */
    @Override
    public ListenableFuture<NestedDependencies> apply(byte[] bytes) {
      if (bytes == null) {
        return immediateFuture(NestedDependencies.newMissingInstance());
      }

      try {
        boolean usesZstdCompression = MagicBytes.hasMagicBytes(bytes);
        InputStream inputStream;
        if (usesZstdCompression) {
          ByteArrayInputStream byteArrayInputStream =
              new ByteArrayInputStream(bytes, 2, bytes.length - 2);
          inputStream = new ZstdInputStream(byteArrayInputStream, RecyclingBufferPool.INSTANCE);
        } else {
          inputStream = new ByteArrayInputStream(bytes);
        }
        try (inputStream) {
          CodedInputStream codedIn = CodedInputStream.newInstance(inputStream);
          int nestedCount = codedIn.readInt32();
          int fileCount = codedIn.readInt32();
          int listingCount = codedIn.readInt32();
          int sourceCount = codedIn.readInt32();

          var elements = new FileSystemDependencies[nestedCount + fileCount + listingCount];
          var sources =
              sourceCount > 0
                  ? new FileDependencies[sourceCount]
                  : NestedDependencies.EMPTY_SOURCES;
          var countdown = new PendingElementCountdown(elements, sources);

          for (int i = 0; i < nestedCount; i++) {
            var key = PackedFingerprint.readFrom(codedIn);
            switch (getNestedDependencies(key, store)) {
              case NestedDependencies dependencies -> elements[i] = dependencies;
              case FutureNestedDependencies future -> {
                countdown.registerPendingElement();
                Futures.addCallback(future, new WaitingForElement(i, countdown), directExecutor());
              }
            }
          }

          int nestedAndFileCount = nestedCount + fileCount;
          for (int i = nestedCount; i < nestedAndFileCount; i++) {
            String key = codedIn.readString();
            switch (getFileDependencies(key, store)) {
              case FileDependencies dependencies -> elements[i] = dependencies;
              case FutureFileDependencies future -> {
                countdown.registerPendingElement();
                Futures.addCallback(future, new WaitingForElement(i, countdown), directExecutor());
              }
            }
          }

          int total = nestedAndFileCount + listingCount;
          for (int i = nestedAndFileCount; i < total; i++) {
            String key = codedIn.readString();
            switch (getListingDependencies(key, store)) {
              case ListingDependencies dependencies -> elements[i] = dependencies;
              case FutureListingDependencies future -> {
                countdown.registerPendingElement();
                Futures.addCallback(future, new WaitingForElement(i, countdown), directExecutor());
              }
            }
          }

          for (int i = 0; i < sourceCount; i++) {
            String key = codedIn.readString();
            switch (getFileDependencies(key, store)) {
              case FileDependencies dependencies -> sources[i] = dependencies;
              case FutureFileDependencies future -> {
                countdown.registerPendingElement();
                Futures.addCallback(future, new WaitingForSource(i, countdown), directExecutor());
              }
            }
          }
          countdown.notifyInitializationDone();
          return countdown;
        }
      } catch (IOException e) {
        return immediateFailedFuture(
            new SerializationException("Error deserializing nested node", e));
      }
    }
  }

  /**
   * A future that keeps track of the count of elements that still need to be set.
   *
   * <p>This future completes once all the elements are set.
   */
  private static class PendingElementCountdown extends QuiescingFuture<NestedDependencies> {
    private final FileSystemDependencies[] elements;
    private final FileDependencies[] sources;

    private PendingElementCountdown(FileSystemDependencies[] elements, FileDependencies[] sources) {
      super(directExecutor());
      this.elements = elements;
      this.sources = sources;
    }

    private void registerPendingElement() {
      increment();
    }

    private void notifyInitializationDone() {
      decrement();
    }

    private void setPendingElement(int index, FileSystemDependencies value) {
      elements[index] = value;
      decrement();
    }

    private void setSource(int index, FileDependencies value) {
      sources[index] = value;
      decrement();
    }

    private void notifyFailure(Throwable e) {
      notifyException(e);
    }

    @Override
    protected NestedDependencies getValue() {
      return NestedDependencies.from(elements, sources);
    }
  }

  /**
   * Callback that populates the element at {@link #index} upon success.
   *
   * <p>Performs required bookkeeping for {@link PendingElementCountdown}.
   */
  private static class WaitingForElement implements FutureCallback<FileSystemDependencies> {
    private final int index;
    private final PendingElementCountdown countdown;

    private WaitingForElement(int index, PendingElementCountdown countdown) {
      this.index = index;
      this.countdown = countdown;
    }

    @Override
    public void onSuccess(FileSystemDependencies dependencies) {
      countdown.setPendingElement(index, dependencies);
    }

    @Override
    public void onFailure(Throwable t) {
      countdown.notifyFailure(t);
    }
  }

  private static final class WaitingForSource implements FutureCallback<FileDependencies> {
    private final int index;
    private final PendingElementCountdown countdown;

    private WaitingForSource(int index, PendingElementCountdown countdown) {
      this.index = index;
      this.countdown = countdown;
    }

    @Override
    public void onSuccess(FileDependencies dependencies) {
      countdown.setSource(index, dependencies);
    }

    @Override
    public void onFailure(Throwable t) {
      countdown.notifyFailure(t);
    }
  }

  // ---------- Begin shared helpers ----------

  private <KeyT, T, FutureT extends SettableFutureKeyedValue<FutureT, KeyT, T>>
      FutureT fetchInvalidationData(
          Function<KeyT, ? extends KeyBytesProvider> keyConverter,
          BiFunction<KeyT, FingerprintValueStore, AsyncFunction<byte[], T>> waitFactory,
          FutureT ownedFuture,
          FingerprintValueStore store) {
    KeyT key = ownedFuture.key();
    ListenableFuture<byte[]> futureBytes;
    try {
      futureBytes = store.get(keyConverter.apply(key));
    } catch (IOException e) {
      return ownedFuture.failWith(e);
    }

    return ownedFuture.completeWith(
        Futures.transformAsync(futureBytes, waitFactory.apply(key, store), executor));
  }

  private KeyBytesProvider getKeyBytes(String cacheKey) {
    if (cacheKey.length() > MAX_KEY_LENGTH) {
      return fingerprinter.fingerprint(cacheKey.getBytes(UTF_8));
    }
    return new StringKey(cacheKey);
  }

  private static final class DependencyMap<
          KeyT,
          ValueOrFutureT,
          ValueT extends ValueOrFutureT,
          FutureT extends SettableFutureKeyedValue<FutureT, KeyT, ValueT>>
      extends AbstractValueOrFutureMap<KeyT, ValueOrFutureT, ValueT, FutureT> {
    private final BiFunction<FutureT, FingerprintValueStore, ValueOrFutureT> populator;

    private DependencyMap(
        ConcurrentMap<KeyT, ValueOrFutureT> map,
        BiFunction<KeyT, BiConsumer<KeyT, ValueT>, ValueOrFutureT> valueOrFutureFactory,
        Class<FutureT> futureType,
        BiFunction<FutureT, FingerprintValueStore, ValueOrFutureT> populator) {
      super(map, valueOrFutureFactory, futureType);
      this.populator = populator;
    }

    ValueOrFutureT getValueOrFuture(KeyT key, FingerprintValueStore store) {
      ValueOrFutureT result = getOrCreateValueForSubclasses(key);
      if (futureType().isInstance(result)) {
        FutureT future = futureType().cast(result);
        if (future.tryTakeOwnership()) {
          try {
            return populator.apply(future, store);
          } finally {
            future.verifyComplete();
          }
        }
      }
      return result;
    }
  }
}
