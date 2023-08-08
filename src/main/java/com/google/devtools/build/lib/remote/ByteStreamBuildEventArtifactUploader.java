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

import static com.google.devtools.build.lib.remote.util.DigestUtil.isOldStyleDigestFunction;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.Utils.grpcAwareErrorMessage;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildtool.buildevent.ProfilerStartedEvent;
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
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A {@link BuildEventArtifactUploader} backed by {@link RemoteCache}. */
class ByteStreamBuildEventArtifactUploader extends AbstractReferenceCounted
    implements BuildEventArtifactUploader {
  private static final Pattern TEST_LOG_PATTERN = Pattern.compile(".*/bazel-out/[^/]*/testlogs/.*");
  private static final Pattern BUILD_LOG_PATTERN =
      Pattern.compile(".*/bazel-out/_tmp/actions/std(err|out)-.*");

  private final Executor executor;
  private final ExtendedEventHandler reporter;
  private final boolean verboseFailures;
  private final RemoteCache remoteCache;
  private final String buildRequestId;
  private final String commandId;
  private final String remoteServerInstanceName;

  private final AtomicBoolean shutdown = new AtomicBoolean();
  private final Scheduler scheduler;

  private final Set<PathFragment> omittedFiles = Sets.newConcurrentHashSet();
  private final Set<PathFragment> omittedTreeRoots = Sets.newConcurrentHashSet();
  private final XattrProvider xattrProvider;
  private final RemoteBuildEventUploadMode remoteBuildEventUploadMode;

  @Nullable private Path profilePath;

  ByteStreamBuildEventArtifactUploader(
      Executor executor,
      ExtendedEventHandler reporter,
      boolean verboseFailures,
      RemoteCache remoteCache,
      String remoteServerInstanceName,
      String buildRequestId,
      String commandId,
      XattrProvider xattrProvider,
      RemoteBuildEventUploadMode remoteBuildEventUploadMode) {
    this.executor = executor;
    this.reporter = reporter;
    this.verboseFailures = verboseFailures;
    this.remoteCache = remoteCache;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteServerInstanceName = remoteServerInstanceName;
    this.scheduler = Schedulers.from(executor);
    this.xattrProvider = xattrProvider;
    this.remoteBuildEventUploadMode = remoteBuildEventUploadMode;
  }

  public void omitFile(Path file) {
    Preconditions.checkState(
        remoteBuildEventUploadMode != RemoteBuildEventUploadMode.MINIMAL,
        "Cannot omit file in MINIMAL mode");
    omittedFiles.add(file.asFragment());
  }

  public void omitTree(Path treeRoot) {
    Preconditions.checkState(
        remoteBuildEventUploadMode != RemoteBuildEventUploadMode.MINIMAL,
        "Cannot omit tree in MINIMAL mode");
    omittedTreeRoots.add(treeRoot.asFragment());
  }

  /** Returns {@code true} if Bazel knows that the file is stored on a remote system. */
  private static boolean isRemoteFile(Path file) {
    return file.getFileSystem() instanceof RemoteActionFileSystem
        && ((RemoteActionFileSystem) file.getFileSystem()).isRemote(file);
  }

  private static final class PathMetadata {

    private final Path path;
    private final Digest digest;
    private final boolean directory;
    private final boolean symlink;
    private final boolean remote;
    private final boolean omitted;
    private final DigestFunction.Value digestFunction;

    PathMetadata(
        Path path,
        Digest digest,
        boolean directory,
        boolean symlink,
        boolean remote,
        boolean omitted,
        DigestFunction.Value digestFunction) {
      this.path = path;
      this.digest = digest;
      this.directory = directory;
      this.symlink = symlink;
      this.remote = remote;
      this.omitted = omitted;
      this.digestFunction = digestFunction;
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

    public boolean isOmitted() {
      return omitted;
    }

    public DigestFunction.Value getDigestFunction() {
      return digestFunction;
    }
  }

  /**
   * Collects metadata for {@code file}. Depending on the underlying filesystem used this method
   * might do I/O.
   */
  private PathMetadata readPathMetadata(Path path, LocalFile file) throws IOException {
    DigestUtil digestUtil = new DigestUtil(xattrProvider, path.getFileSystem().getDigestFunction());

    if (file.type == LocalFileType.OUTPUT_DIRECTORY
        || (file.type == LocalFileType.OUTPUT && path.isDirectory())) {
      return new PathMetadata(
          path,
          /* digest= */ null,
          /* directory= */ true,
          /* symlink= */ false,
          /* remote= */ false,
          /* omitted= */ false,
          /* digestFunction= */ digestUtil.getDigestFunction());
    }
    if (file.type == LocalFileType.OUTPUT_SYMLINK) {
      return new PathMetadata(
          path,
          /* digest= */ null,
          /* directory= */ false,
          /* symlink= */ true,
          /* remote= */ false,
          /* omitted= */ false,
          /* digestFunction= */ digestUtil.getDigestFunction());
    }

    PathFragment filePathFragment = path.asFragment();
    boolean omitted = false;
    if (omittedFiles.contains(filePathFragment)) {
      omitted = true;
    } else {
      for (PathFragment treeRoot : omittedTreeRoots) {
        if (path.startsWith(treeRoot)) {
          omittedFiles.add(filePathFragment);
          omitted = true;
        }
      }
    }

    Digest digest = digestUtil.compute(path);
    return new PathMetadata(
        path,
        digest,
        /* directory= */ false,
        /* symlink= */ false,
        isRemoteFile(path),
        omitted,
        digestUtil.getDigestFunction());
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
                file.isOmitted(),
                file.getDigestFunction());
        knownRemotePaths.add(remotePathMetadata);
      }
    }
  }

  private boolean shouldUpload(PathMetadata path) {
    boolean result =
        path.getDigest() != null
            && !path.isRemote()
            && !path.isDirectory()
            && !path.isSymlink()
            && !path.isOmitted();

    if (remoteBuildEventUploadMode == RemoteBuildEventUploadMode.MINIMAL) {
      result = result && (isLog(path) || isProfile(path));
    }

    return result;
  }

  private boolean isLog(PathMetadata path) {
    return TEST_LOG_PATTERN.matcher(path.getPath().getPathString()).matches()
        || BUILD_LOG_PATTERN.matcher(path.getPath().getPathString()).matches();
  }

  private boolean isProfile(PathMetadata path) {
    return path.getPath().equals(profilePath);
  }

  private Single<List<PathMetadata>> queryRemoteCache(
      RemoteCache remoteCache, RemoteActionExecutionContext context, List<PathMetadata> paths) {
    List<PathMetadata> knownRemotePaths = new ArrayList<>(paths.size());
    List<PathMetadata> filesToQuery = new ArrayList<>();
    Set<Digest> digestsToQuery = new HashSet<>();
    for (PathMetadata path : paths) {
      // Query remote cache for files even if omitted from uploading
      if (shouldUpload(path) || path.isOmitted()) {
        filesToQuery.add(path);
        digestsToQuery.add(path.getDigest());
      } else {
        knownRemotePaths.add(path);
      }
    }

    if (digestsToQuery.isEmpty()) {
      return Single.just(knownRemotePaths);
    }
    return toSingle(() -> remoteCache.findMissingDigests(context, digestsToQuery), executor)
        .onErrorResumeNext(
            error -> {
              reportUploadError(error, null, null);
              // Assuming all digests are missing if failed to query
              return Single.just(ImmutableSet.copyOf(digestsToQuery));
            })
        .map(
            missingDigests -> {
              processQueryResult(missingDigests, filesToQuery, knownRemotePaths);
              return knownRemotePaths;
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
      RemoteCache remoteCache, RemoteActionExecutionContext context, List<PathMetadata> paths) {
    return Flowable.fromIterable(paths)
        .flatMapSingle(
            path -> {
              if (!shouldUpload(path)) {
                return Single.just(path);
              }

              return toCompletable(
                      () -> remoteCache.uploadFile(context, path.getDigest(), path.getPath()),
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
                              path.isOmitted(),
                              path.getDigestFunction()))
                  .onErrorResumeNext(
                      error -> {
                        reportUploadError(error, path.getPath(), path.getDigest());
                        return Single.just(path);
                      });
            })
        .collect(Collectors.toList());
  }

  private Single<PathConverter> doUpload(Map<Path, LocalFile> files) {
    if (files.isEmpty()) {
      return Single.just(PathConverter.NO_CONVERSION);
    }

    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(buildRequestId, commandId, "bes-upload", null);
    RemoteActionExecutionContext context =
        RemoteActionExecutionContext.create(metadata)
            .withWriteCachePolicy(CachePolicy.REMOTE_CACHE_ONLY);

    return Single.using(
        remoteCache::retain,
        remoteCache ->
            Flowable.fromIterable(files.entrySet())
                .map(
                    entry -> {
                      Path path = entry.getKey();
                      LocalFile file = entry.getValue();
                      try {
                        return readPathMetadata(path, file);
                      } catch (IOException e) {
                        reportUploadError(e, path, null);
                        return new PathMetadata(
                            path,
                            /* digest= */ null,
                            /* directory= */ false,
                            /* symlink= */ false,
                            /* remote= */ false,
                            /* omitted= */ false,
                            DigestFunction.Value.SHA256);
                      }
                    })
                .collect(Collectors.toList())
                .flatMap(paths -> queryRemoteCache(remoteCache, context, paths))
                .flatMap(paths -> uploadLocalFiles(remoteCache, context, paths))
                .map(
                    paths ->
                        new PathConverterImpl(
                            remoteServerInstanceName, paths, remoteBuildEventUploadMode)),
        RemoteCache::release);
  }

  @Subscribe
  public void onProfilerStartedEvent(ProfilerStartedEvent event) {
    profilePath = event.getProfilePath();
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
    remoteCache.release();
  }

  @Override
  public ReferenceCounted touch(Object o) {
    return this;
  }

  private static class PathConverterImpl implements PathConverter {

    private final String remoteServerInstanceName;
    private final Map<Path, PathMetadata> pathToMetadata;
    private final Set<Path> skippedPaths;
    private final Set<Path> localPaths;

    PathConverterImpl(
        String remoteServerInstanceName,
        List<PathMetadata> uploads,
        RemoteBuildEventUploadMode remoteBuildEventUploadMode) {
      Preconditions.checkNotNull(uploads);
      this.remoteServerInstanceName = remoteServerInstanceName;
      pathToMetadata = Maps.newHashMapWithExpectedSize(uploads.size());
      ImmutableSet.Builder<Path> skippedPaths = ImmutableSet.builder();
      ImmutableSet.Builder<Path> localPaths = ImmutableSet.builder();
      for (PathMetadata metadata : uploads) {
        Path path = metadata.getPath();
        Digest digest = metadata.getDigest();
        if (digest != null) {
          // Always use bytestream:// in MINIMAL mode
          if (remoteBuildEventUploadMode == RemoteBuildEventUploadMode.MINIMAL) {
            pathToMetadata.put(path, metadata);
          } else if (metadata.isRemote()) {
            pathToMetadata.put(path, metadata);
          } else {
            localPaths.add(path);
          }
        } else {
          skippedPaths.add(path);
        }
      }
      this.skippedPaths = skippedPaths.build();
      this.localPaths = localPaths.build();
    }

    @Override
    @Nullable
    public String apply(Path path) {
      Preconditions.checkNotNull(path);

      if (localPaths.contains(path)) {
        return String.format("file://%s", path.getPathString());
      }

      PathMetadata metadata = pathToMetadata.get(path);
      if (metadata == null) {
        if (skippedPaths.contains(path)) {
          return null;
        }
        // It's a programming error to reference a file that has not been uploaded.
        throw new IllegalStateException(
            String.format("Illegal file reference: '%s'", path.getPathString()));
      }

      Digest digest = metadata.getDigest();
      DigestFunction.Value digestFunction = metadata.getDigestFunction();
      String out;
      if (isOldStyleDigestFunction(digestFunction)) {
        out =
            String.format(
                "bytestream://%s/blobs/%s/%d",
                remoteServerInstanceName, digest.getHash(), digest.getSizeBytes());
      } else {
        out =
            String.format(
                "bytestream://%s/blobs/%s/%s/%d",
                remoteServerInstanceName,
                Ascii.toLowerCase(digestFunction.getValueDescriptor().getName()),
                digest.getHash(),
                digest.getSizeBytes());
      }
      return out;
    }
  }
}
