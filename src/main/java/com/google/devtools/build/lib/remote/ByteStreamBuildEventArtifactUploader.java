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

import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.Utils.grpcAwareErrorMessage;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext.Step;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.XattrProvider;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.schedulers.Schedulers;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/** A {@link BuildEventArtifactUploader} backed by {@link RemoteCache}. */
class ByteStreamBuildEventArtifactUploader extends AbstractReferenceCounted
    implements BuildEventArtifactUploader {

  private final Executor executor;
  private final ExtendedEventHandler reporter;
  private final boolean verboseFailures;
  private final RemoteCache remoteCache;
  private final String buildRequestId;
  private final String commandId;
  private final String remoteServerInstanceName;

  private final AtomicBoolean shutdown = new AtomicBoolean();
  private final Scheduler scheduler;

  private final Set<Path> omittedFiles = Sets.newConcurrentHashSet();
  private final Set<Path> omittedTreeRoots = Sets.newConcurrentHashSet();
  private final XattrProvider xattrProvider;

  ByteStreamBuildEventArtifactUploader(
      Executor executor,
      ExtendedEventHandler reporter,
      boolean verboseFailures,
      RemoteCache remoteCache,
      String remoteServerInstanceName,
      String buildRequestId,
      String commandId,
      XattrProvider xattrProvider) {
    this.executor = executor;
    this.reporter = reporter;
    this.verboseFailures = verboseFailures;
    this.remoteCache = remoteCache;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteServerInstanceName = remoteServerInstanceName;
    this.scheduler = Schedulers.from(executor);
    this.xattrProvider = xattrProvider;
  }

  public void omitFile(Path file) {
    omittedFiles.add(file);
  }

  public void omitTree(Path treeRoot) {
    omittedTreeRoots.add(treeRoot);
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
    private final boolean remote;

    PathMetadata(Path path, Digest digest, boolean directory, boolean remote) {
      this.path = path;
      this.digest = digest;
      this.directory = directory;
      this.remote = remote;
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

    public boolean isRemote() {
      return remote;
    }
  }

  /**
   * Collects metadata for {@code file}. Depending on the underlying filesystem used this method
   * might do I/O.
   */
  private PathMetadata readPathMetadata(Path file) throws IOException {
    if (file.isDirectory()) {
      return new PathMetadata(file, /* digest= */ null, /* directory= */ true, /* remote= */ false);
    }
    if (omittedFiles.contains(file)) {
      return new PathMetadata(file, /*digest=*/ null, /*directory=*/ false, /*remote=*/ false);
    }

    for (Path treeRoot : omittedTreeRoots) {
      if (file.startsWith(treeRoot)) {
        omittedFiles.add(file);
        return new PathMetadata(file, /*digest=*/ null, /*directory=*/ false, /*remote=*/ false);
      }
    }

    DigestUtil digestUtil = new DigestUtil(xattrProvider, file.getFileSystem().getDigestFunction());
    Digest digest = digestUtil.compute(file);
    return new PathMetadata(file, digest, /* directory= */ false, isRemoteFile(file));
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
                file.getPath(), file.getDigest(), file.isDirectory(), /* remote= */ true);
        knownRemotePaths.add(remotePathMetadata);
      }
    }
  }

  private static boolean shouldUpload(PathMetadata path) {
    return path.getDigest() != null && !path.isRemote() && !path.isDirectory();
  }

  private Single<List<PathMetadata>> queryRemoteCache(
      RemoteCache remoteCache, RemoteActionExecutionContext context, List<PathMetadata> paths) {
    List<PathMetadata> knownRemotePaths = new ArrayList<>(paths.size());
    List<PathMetadata> filesToQuery = new ArrayList<>();
    Set<Digest> digestsToQuery = new HashSet<>();
    for (PathMetadata path : paths) {
      if (shouldUpload(path)) {
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
              reporterUploadError(error);
              // Assuming all digests are missing if failed to query
              return Single.just(ImmutableSet.copyOf(digestsToQuery));
            })
        .map(
            missingDigests -> {
              processQueryResult(missingDigests, filesToQuery, knownRemotePaths);
              return knownRemotePaths;
            });
  }

  private void reporterUploadError(Throwable error) {
    if (error instanceof CancellationException) {
      return;
    }

    String errorMessage =
        "Uploading BEP referenced local files: " + grpcAwareErrorMessage(error, verboseFailures);

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
                  .toSingleDefault(path)
                  .onErrorResumeNext(
                      error -> {
                        reporterUploadError(error);
                        return Single.just(
                            new PathMetadata(
                                path.getPath(),
                                /*digest=*/ null,
                                path.isDirectory(),
                                path.isRemote()));
                      });
            })
        .collect(Collectors.toList());
  }

  private Single<PathConverter> upload(Set<Path> files) {
    if (files.isEmpty()) {
      return Single.just(PathConverter.NO_CONVERSION);
    }

    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(buildRequestId, commandId, "bes-upload", null);
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(metadata);
    context.setStep(Step.UPLOAD_BES_FILES);

    return Single.using(
        remoteCache::retain,
        remoteCache ->
            Flowable.fromIterable(files)
                .map(
                    file -> {
                      try {
                        return readPathMetadata(file);
                      } catch (IOException e) {
                        reporterUploadError(e);
                        return new PathMetadata(
                            file, /*digest=*/ null, /*directory=*/ false, /*remote=*/ false);
                      }
                    })
                .collect(Collectors.toList())
                .flatMap(paths -> queryRemoteCache(remoteCache, context, paths))
                .flatMap(paths -> uploadLocalFiles(remoteCache, context, paths))
                .map(paths -> new PathConverterImpl(remoteServerInstanceName, paths, omittedFiles)),
        RemoteCache::release);
  }

  @Override
  public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
    return toListenableFuture(upload(files.keySet()).subscribeOn(scheduler));
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
    private final Map<Path, Digest> pathToDigest;
    private final Set<Path> skippedPaths;
    private final Set<Path> localPaths;

    PathConverterImpl(
        String remoteServerInstanceName, List<PathMetadata> uploads, Set<Path> localPaths) {
      Preconditions.checkNotNull(uploads);
      this.remoteServerInstanceName = remoteServerInstanceName;
      pathToDigest = new HashMap<>(uploads.size());
      ImmutableSet.Builder<Path> skippedPaths = ImmutableSet.builder();
      for (PathMetadata pair : uploads) {
        Path path = pair.getPath();
        Digest digest = pair.getDigest();
        if (digest != null) {
          pathToDigest.put(path, digest);
        } else {
          skippedPaths.add(path);
        }
      }
      this.skippedPaths = skippedPaths.build();
      this.localPaths = localPaths;
    }

    @Override
    public String apply(Path path) {
      Preconditions.checkNotNull(path);

      if (localPaths.contains(path)) {
        return String.format("file://%s", path.getPathString());
      }

      Digest digest = pathToDigest.get(path);
      if (digest == null) {
        if (skippedPaths.contains(path)) {
          return null;
        }
        // It's a programming error to reference a file that has not been uploaded.
        throw new IllegalStateException(
            String.format("Illegal file reference: '%s'", path.getPathString()));
      }
      return String.format(
          "bytestream://%s/blobs/%s/%d",
          remoteServerInstanceName, digest.getHash(), digest.getSizeBytes());
    }
  }
}
