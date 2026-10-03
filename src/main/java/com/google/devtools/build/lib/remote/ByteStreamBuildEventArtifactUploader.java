// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.DigestUtil.isOldStyleDigestFunction;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.Utils.grpcAwareErrorMessage;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext.CachePolicy;
import com.google.devtools.build.lib.remote.options.RemoteBuildEventUploadMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.XattrProvider;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.schedulers.Schedulers;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A {@link BuildEventArtifactUploader} backed by {@link CombinedCache}. */
class ByteStreamBuildEventArtifactUploader extends AbstractReferenceCounted
    implements BuildEventArtifactUploader {
  private static final Pattern TEST_LOG_PATTERN = Pattern.compile(".*/bazel-out/[^/]*/testlogs/.*");
  private static final Pattern BUILD_LOG_PATTERN =
      Pattern.compile(".*/bazel-out/_tmp/actions/std(err|out)-.*");
  private static final long MAX_URI_CACHE_SIZE = 4_000_000;

  private final Executor executor;
  private final ExtendedEventHandler reporter;
  private final boolean verboseFailures;
  private final CombinedCache combinedCache;
  private final String buildRequestId;
  private final String commandId;
  private final String remoteInstanceName;
  private final String remoteBytestreamUriPrefix;

  private final AtomicBoolean shutdown = new AtomicBoolean();
  private final Scheduler scheduler;

  private final XattrProvider xattrProvider;
  private final RemoteBuildEventUploadMode remoteBuildEventUploadMode;
  private final int maximumOpenFiles;

  /**
   * The {@code bytestream://} URIs of files known to be present in the remote cache, keyed by
   * path and digest.
   *
   * <p>Every {@code TargetCompleteEvent} references the complete transitive set of important
   * outputs of its target, so a file shared by many targets is passed to {@link #upload} once per
   * target. This cache makes sure such a file is queried and uploaded at most once per command,
   * and that the {@link PathConverter}s of all pending events share a single URI string for it.
   *
   * <p>Values are held weakly: an entry lives exactly as long as some pending event still
   * references its URI, so the cache retains nothing beyond what the pending events already do.
   * The size bound only guards against pathological cases.
   */
  private final Cache<UriCacheKey, String> uriCache =
      Caffeine.newBuilder().maximumSize(MAX_URI_CACHE_SIZE).weakValues().build();

  /**
   * Canonical instances of the path fragments that the {@link PathConverter}s of pending events
   * use as keys. Every event resolves its paths through its own (possibly per-action) file system,
   * so without interning each pending event would retain its own copy of the fragment and path
   * string of every file it references. Weak, so it retains nothing beyond what the converters do.
   */
  private final Interner<PathFragment> pathFragmentInterner = BlazeInterners.newWeakInterner();

  ByteStreamBuildEventArtifactUploader(
      Executor executor,
      ExtendedEventHandler reporter,
      boolean verboseFailures,
      CombinedCache combinedCache,
      String remoteInstanceName,
      String remoteBytestreamUriPrefix,
      String buildRequestId,
      String commandId,
      XattrProvider xattrProvider,
      RemoteBuildEventUploadMode remoteBuildEventUploadMode,
      int maximumOpenFiles) {
    this.executor = executor;
    this.reporter = reporter;
    this.verboseFailures = verboseFailures;
    this.combinedCache = combinedCache;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteInstanceName = remoteInstanceName;
    this.remoteBytestreamUriPrefix = remoteBytestreamUriPrefix;
    this.scheduler = Schedulers.from(executor);
    this.xattrProvider = xattrProvider;
    this.remoteBuildEventUploadMode = remoteBuildEventUploadMode;
    this.maximumOpenFiles = maximumOpenFiles;
  }

  /** Returns {@code true} if Bazel knows that the file is stored on a remote system. */
  private static boolean isRemoteFile(Path file) throws IOException {
    return file.getFileSystem() instanceof RemoteActionFileSystem
        && ((RemoteActionFileSystem) file.getFileSystem()).isRemote(file);
  }

  private static final class PathMetadata {

    private final Path path;
    private final Digest digest;
    private final boolean directory;
    private final boolean symlink;
    private final boolean remote;
    private final boolean isBuildToolLog;
    private final boolean specialFile;
    private final DigestFunction.Value digestFunction;

    /** The URI an earlier event established for this file, if any. Implies {@link #remote}. */
    @Nullable private final String uri;

    PathMetadata(
        Path path,
        Digest digest,
        boolean directory,
        boolean symlink,
        boolean remote,
        boolean isBuildToolLog,
        boolean specialFile,
        DigestFunction.Value digestFunction) {
      this(
          path,
          digest,
          directory,
          symlink,
          remote,
          isBuildToolLog,
          specialFile,
          digestFunction,
          /* uri= */ null);
    }

    private PathMetadata(
        Path path,
        Digest digest,
        boolean directory,
        boolean symlink,
        boolean remote,
        boolean isBuildToolLog,
        boolean specialFile,
        DigestFunction.Value digestFunction,
        @Nullable String uri) {
      this.path = path;
      this.digest = digest;
      this.directory = directory;
      this.symlink = symlink;
      this.remote = remote;
      this.isBuildToolLog = isBuildToolLog;
      this.specialFile = specialFile;
      this.digestFunction = digestFunction;
      this.uri = uri;
    }

    /** Returns a copy for a file that is known to be present remotely under the given URI. */
    PathMetadata withRemoteUri(String uri) {
      return new PathMetadata(
          path,
          digest,
          directory,
          symlink,
          /* remote= */ true,
          isBuildToolLog,
          specialFile,
          digestFunction,
          uri);
    }

    public Path getPath() {
      return path;
    }

    public Digest getDigest() {
      return digest;
    }

    public boolean isDirectory() {
      return directory;
    }

    public boolean isSymlink() {
      return symlink;
    }

    public boolean isRemote() {
      return remote;
    }

    public boolean isBuildToolLog() {
      return isBuildToolLog;
    }

    boolean isSpecialFile() {
      return specialFile;
    }

    public DigestFunction.Value getDigestFunction() {
      return digestFunction;
    }

    @Nullable
    String getUri() {
      return uri;
    }
  }

  /** Key of {@link #uriCache}: within a command, a file is identified by its path and content. */
  private record UriCacheKey(PathFragment path, Digest digest) {}

  @Nullable
  private UriCacheKey uriCacheKey(PathMetadata file) {
    // Build tool logs are written while the build runs and are not shared between events.
    if (file.getDigest() == null || file.isBuildToolLog()) {
      return null;
    }
    return new UriCacheKey(
        pathFragmentInterner.intern(file.getPath().asFragment()), file.getDigest());
  }

  /**
   * Returns {@code file} marked as remote if an earlier event already established that it is
   * present in the remote cache, so that it is neither queried nor uploaded again.
   */
  private PathMetadata withCachedUri(PathMetadata file) {
    UriCacheKey key = uriCacheKey(file);
    if (key == null) {
      return file;
    }
    String uri = uriCache.getIfPresent(key);
    return uri != null ? file.withRemoteUri(uri) : file;
  }

  /**
   * Returns the {@code bytestream://} URI of a file, sharing a single {@link String} among all
   * events of this command that reference the file where possible.
   */
  private String bytestreamUri(String remoteServerInstanceName, PathMetadata file) {
    if (file.getUri() != null) {
      return file.getUri();
    }
    // A cached URI marks the file as present remotely for later events, so only remember files
    // that are known to be present or that this mode never uploads anyway. In particular, a file
    // whose upload failed (which in MINIMAL mode still gets a bytestream:// URI) must be retried by
    // the next event that references it.
    UriCacheKey key = file.isRemote() || !isUploadCandidate(file) ? uriCacheKey(file) : null;
    if (key == null) {
      return computeBytestreamUri(
          remoteServerInstanceName, file.getDigest(), file.getDigestFunction());
    }
    return uriCache.get(
        key,
        unused ->
            computeBytestreamUri(
                remoteServerInstanceName, file.getDigest(), file.getDigestFunction()));
  }

  private static String computeBytestreamUri(
      String remoteServerInstanceName, Digest digest, DigestFunction.Value digestFunction) {
    StringBuilder uri =
        new StringBuilder("bytestream://").append(remoteServerInstanceName).append("/blobs/");
    if (!isOldStyleDigestFunction(digestFunction)) {
      uri.append(Ascii.toLowerCase(digestFunction.getValueDescriptor().getName())).append('/');
    }
    return uri.append(digest.getHash()).append('/').append(digest.getSizeBytes()).toString();
  }

  /**
   * Collects metadata for {@code file}. Depending on the underlying filesystem used this method
   * might do I/O.
   */
  private PathMetadata readPathMetadata(Path path, LocalFile file) throws IOException {
    DigestUtil digestUtil = new DigestUtil(xattrProvider, path.getFileSystem().getDigestFunction());
    DigestFunction.Value digestFunction = digestUtil.getDigestFunction();
    boolean isBuildToolLog =
        file.type == LocalFileType.LOG || file.type == LocalFileType.PERFORMANCE_LOG;
    FileArtifactValue metadata = file.artifactMetadata;
    if (metadata != null) {
      switch (metadata.getType()) {
        case DIRECTORY -> {
          return new PathMetadata(
              path,
              /* digest= */ null,
              /* directory= */ true,
              /* symlink= */ false,
              /* remote= */ false,
              /* isBuildToolLog= */ false,
              /* specialFile= */ false,
              digestFunction);
        }
        case SYMLINK -> {
          return new PathMetadata(
              path,
              /* digest= */ null,
              /* directory= */ false,
              /* symlink= */ true,
              /* remote= */ false,
              /* isBuildToolLog= */ false,
              /* specialFile= */ false,
              digestFunction);
        }
        case REGULAR_FILE -> {
          if (metadata.getDigest() != null) {
            return new PathMetadata(
                path,
                DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize()),
                /* directory= */ false,
                /* symlink= */ false,
                /* remote= */ metadata.isRemote(),
                isBuildToolLog,
                /* specialFile= */ false,
                digestFunction);
          }
        }
        default -> {}
      }
    }

    if (file.type == LocalFileType.OUTPUT_DIRECTORY
        || ((file.type == LocalFileType.SUCCESSFUL_TEST_OUTPUT
                || file.type == LocalFileType.FAILED_TEST_OUTPUT)
            && path.isDirectory())) {
      return new PathMetadata(
          path,
          /* digest= */ null,
          /* directory= */ true,
          /* symlink= */ false,
          /* remote= */ false,
          /* isBuildToolLog= */ false,
          /* specialFile= */ false,
          digestFunction);
    }
    if (file.type == LocalFileType.OUTPUT_SYMLINK) {
      return new PathMetadata(
          path,
          /* digest= */ null,
          /* directory= */ false,
          /* symlink= */ true,
          /* remote= */ false,
          /* isBuildToolLog= */ false,
          /* specialFile= */ false,
          digestFunction);
    }
    if (path.isSpecialFile()) {
      return new PathMetadata(
          path,
          /* digest= */ null,
          /* directory= */ false,
          /* symlink= */ false,
          /* remote= */ false,
          /* isBuildToolLog= */ false,
          /* specialFile= */ true,
          digestFunction);
    }

    Digest digest = digestUtil.compute(path);
    return new PathMetadata(
        path,
        digest,
        /* directory= */ false,
        /* symlink= */ false,
        isRemoteFile(path),
        isBuildToolLog,
        /* specialFile= */ false,
        digestFunction);
  }

  private static void processQueryResult(
      ImmutableSet<Digest> missingDigests,
      List<PathMetadata> filesToQuery,
      List<PathMetadata> knownRemotePaths) {
    for (PathMetadata file : filesToQuery) {
      if (missingDigests.contains(file.getDigest())) {
        knownRemotePaths.add(file);
      } else {
        PathMetadata remotePathMetadata =
            new PathMetadata(
                file.getPath(),
                file.getDigest(),
                file.isDirectory(),
                file.isSymlink(),
                /* remote= */ true,
                file.isBuildToolLog(),
                file.isSpecialFile(),
                file.getDigestFunction());
        knownRemotePaths.add(remotePathMetadata);
      }
    }
  }

  private boolean shouldUpload(PathMetadata path) {
    return !path.isRemote() && isUploadCandidate(path);
  }

  /** Returns whether the file would be uploaded if it were not already present remotely. */
  private boolean isUploadCandidate(PathMetadata path) {
    boolean result =
        path.getDigest() != null
            && !path.isDirectory()
            && !path.isSymlink()
            && !path.isSpecialFile();

    if (remoteBuildEventUploadMode == RemoteBuildEventUploadMode.MINIMAL) {
      result = result && (path.isBuildToolLog() || isBuildOrTestLog(path));
    }

    return result;
  }

  private boolean isBuildOrTestLog(PathMetadata path) {
    return TEST_LOG_PATTERN.matcher(path.getPath().getPathString()).matches()
        || BUILD_LOG_PATTERN.matcher(path.getPath().getPathString()).matches();
  }

  private Single<List<PathMetadata>> queryCombinedCache(
      CombinedCache combinedCache, RemoteActionExecutionContext context, List<PathMetadata> paths) {
    List<PathMetadata> knownPaths = new ArrayList<>(paths.size());
    List<PathMetadata> filesToQuery = new ArrayList<>();
    Set<Digest> digestsToQuery = new HashSet<>();
    for (PathMetadata path : paths) {
      if (shouldUpload(path)) {
        filesToQuery.add(path);
        digestsToQuery.add(path.getDigest());
      } else {
        knownPaths.add(path);
      }
    }

    if (digestsToQuery.isEmpty()) {
      return Single.just(knownPaths);
    }
    return toSingle(() -> combinedCache.findMissingDigests(context, digestsToQuery), executor)
        .onErrorResumeNext(
            error -> {
              reportUploadError(error, null, null);
              // Assuming all digests are missing if failed to query
              return Single.just(ImmutableSet.copyOf(digestsToQuery));
            })
        .map(
            missingDigests -> {
              processQueryResult(missingDigests, filesToQuery, knownPaths);
              return knownPaths;
            });
  }

  private void reportUploadError(Throwable error, Path path, Digest digest) {
    if (error instanceof CancellationException) {
      return;
    }

    String errorMessage = "Uploading BEP referenced local file";
    if (path != null) {
      errorMessage += " " + path;
    }
    if (digest != null) {
      errorMessage += " " + digest;
    }
    errorMessage += ": " + grpcAwareErrorMessage(error, verboseFailures);

    reporter.handle(Event.warn(errorMessage));
  }

  private Single<List<PathMetadata>> uploadLocalFiles(
      CombinedCache combinedCache, RemoteActionExecutionContext context, List<PathMetadata> paths) {
    // Limits the concurrency of in-flight file uploads per batch to prevent opening too many
    // files simultaneously (governed by --bep_maximum_open_remote_upload_files).
    int maxConcurrency = maximumOpenFiles > 0 ? maximumOpenFiles : Integer.MAX_VALUE;
    return Flowable.fromIterable(paths)
        .flatMapSingle(
            path -> {
              if (!shouldUpload(path)) {
                return Single.just(path);
              }

              return toCompletable(
                      () -> combinedCache.uploadFile(context, path.getDigest(), path.getPath()),
                      executor)
                  .toSingle(
                      () ->
                          new PathMetadata(
                              path.getPath(),
                              path.getDigest(),
                              path.isDirectory(),
                              path.isSymlink(),
                              // set remote to true so the PathConverter will use bytestream://
                              // scheme to convert the URI for this file
                              /* remote= */ true,
                              path.isBuildToolLog(),
                              path.isSpecialFile(),
                              path.getDigestFunction()))
                  .onErrorResumeNext(
                      error -> {
                        reportUploadError(error, path.getPath(), path.getDigest());
                        return Single.just(path);
                      });
            },
            /* delayErrors= */ false,
            maxConcurrency)
        .collect(Collectors.toList());
  }

  private Single<String> getRemoteServerInstanceName(CombinedCache combinedCache) {
    if (!Strings.isNullOrEmpty(remoteBytestreamUriPrefix)) {
      return Single.just(remoteBytestreamUriPrefix);
    }

    return toSingle(combinedCache::getRemoteAuthority, directExecutor())
        .map(
            a -> {
              if (!Strings.isNullOrEmpty(remoteInstanceName)) {
                return a + "/" + remoteInstanceName;
              }
              return a;
            });
  }

  private Single<PathConverter> doUpload(Map<Path, LocalFile> files) {
    if (files.isEmpty()) {
      return Single.just(PathConverter.NO_CONVERSION);
    }

    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(buildRequestId, commandId, "bes-upload");
    RemoteActionExecutionContext context =
        RemoteActionExecutionContext.create(metadata)
            .withWriteCachePolicy(CachePolicy.REMOTE_CACHE_ONLY);

    return Single.using(
        combinedCache::retain,
        combinedCache ->
            Flowable.fromIterable(files.entrySet())
                .map(
                    entry -> {
                      Path path = entry.getKey();
                      LocalFile file = entry.getValue();
                      try {
                        return withCachedUri(readPathMetadata(path, file));
                      } catch (IOException e) {
                        reportUploadError(e, path, null);
                        return new PathMetadata(
                            path,
                            /* digest= */ null,
                            /* directory= */ false,
                            /* symlink= */ false,
                            /* remote= */ false,
                            /* isBuildToolLog= */ false,
                            /* specialFile= */ false,
                            DigestFunction.Value.SHA256);
                      }
                    })
                .collect(Collectors.toList())
                .flatMap(paths -> queryCombinedCache(combinedCache, context, paths))
                .flatMap(paths -> uploadLocalFiles(combinedCache, context, paths))
                .flatMap(
                    paths ->
                        getRemoteServerInstanceName(combinedCache)
                            .map(
                                remoteServerInstanceName ->
                                    new PathConverterImpl(
                                        paths,
                                        remoteBuildEventUploadMode,
                                        file -> bytestreamUri(remoteServerInstanceName, file),
                                        pathFragmentInterner))),
        CombinedCache::release);
  }

  @Override
  public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
    return toListenableFuture(doUpload(files).subscribeOn(scheduler));
  }

  @Override
  public boolean mayBeSlow() {
    return true;
  }

  @Override
  protected void deallocate() {
    if (shutdown.getAndSet(true)) {
      return;
    }
    combinedCache.release();
  }

  @Override
  public ReferenceCounted touch(Object o) {
    return this;
  }

  /**
   * A {@link PathConverter} that only retains the final URI of every file referenced by a build
   * event.
   *
   * <p>A build event stays pending, and keeps its converter alive, until the BES upload has caught
   * up with it. With {@code --bes_upload_mode=fully_async} on large builds this can be a very long
   * time, so the converter must not retain the {@link PathMetadata} (and thus {@link Digest},
   * {@link DigestFunction} and {@link Path}) of every file it can convert. Instead, the URI is
   * computed once, up front.
   *
   * <p>Paths are keyed by their {@link PathFragment} rather than the {@link Path} itself: a {@link
   * Path} references its {@link com.google.devtools.build.lib.vfs.FileSystem}, which for outputs
   * of remotely executed actions is a per-action {@link RemoteActionFileSystem} that would
   * otherwise be kept alive by the pending event. Callers only ever look up paths that the event
   * declared as referenced local files, so the fragment alone identifies the file.
   *
   * <p>Instances are immutable: {@link #apply} may be called from the BES upload thread while
   * uploads for later events are still running.
   */
  private static final class PathConverterImpl implements PathConverter {
    /** Files reported with a {@code bytestream://} URI, mapped to that URI. */
    private final ImmutableMap<PathFragment, String> pathToUri;

    /** Files that were not uploaded and are reported with a {@code file://} URI. */
    private final ImmutableSet<PathFragment> localPaths;

    /** Files that are omitted from the BEP (e.g. directories, symlinks or failed reads). */
    private final ImmutableSet<PathFragment> skippedPaths;

    PathConverterImpl(
        List<PathMetadata> uploads,
        RemoteBuildEventUploadMode remoteBuildEventUploadMode,
        Function<PathMetadata, String> bytestreamUri,
        Interner<PathFragment> pathFragmentInterner) {
      Preconditions.checkNotNull(uploads);
      ImmutableMap.Builder<PathFragment, String> pathToUri =
          ImmutableMap.builderWithExpectedSize(uploads.size());
      ImmutableSet.Builder<PathFragment> localPaths = ImmutableSet.builder();
      ImmutableSet.Builder<PathFragment> skippedPaths = ImmutableSet.builder();
      for (PathMetadata metadata : uploads) {
        PathFragment path = pathFragmentInterner.intern(metadata.getPath().asFragment());
        Digest digest = metadata.getDigest();
        if (digest != null) {
          // Always use bytestream:// in MINIMAL mode
          if (remoteBuildEventUploadMode == RemoteBuildEventUploadMode.MINIMAL
              || metadata.isRemote()) {
            pathToUri.put(path, bytestreamUri.apply(metadata));
          } else {
            localPaths.add(path);
          }
        } else if (metadata.isSpecialFile()) {
          localPaths.add(path);
        } else {
          skippedPaths.add(path);
        }
      }
      // The same file may legitimately be referenced through paths on different file systems.
      this.pathToUri = pathToUri.buildKeepingLast();
      this.localPaths = localPaths.build();
      this.skippedPaths = skippedPaths.build();
    }

    @Override
    @Nullable
    public String apply(Path path) {
      Preconditions.checkNotNull(path);
      PathFragment fragment = path.asFragment();

      if (localPaths.contains(fragment)) {
        return "file://" + path.getPathString();
      }

      String uri = pathToUri.get(fragment);
      if (uri == null) {
        if (skippedPaths.contains(fragment)) {
          return null;
        }
        // It's a programming error to reference a file that has not been uploaded.
        throw new IllegalStateException(
            String.format("Illegal file reference: '%s'", path.getPathString()));
      }
      return uri;
    }
  }
}
