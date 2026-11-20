// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote.worker;

import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.devtools.build.lib.remote.BazelOutputServiceGrpc;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.BatchStatRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.BatchStatResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.CleanRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.CleanResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.FinalizeArtifactsRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.FinalizeArtifactsResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.FinalizeBuildRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.FinalizeBuildResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StageArtifactsRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StageArtifactsResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StartBuildRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StartBuildResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceREv2Proto.FileArtifactLocator;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.Options;
import com.google.protobuf.Any;
import io.grpc.Status;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

/** An output-service daemon backed by ordinary files and the integration test worker's CAS. */
public final class TestOutputService extends BazelOutputServiceGrpc.BazelOutputServiceImplBase {
  private final Path root;
  private final OnDiskBlobStoreCache cache;
  private final DigestUtil digestUtil;
  private final ConcurrentHashMap<String, Path> builds = new ConcurrentHashMap<>();

  private TestOutputService(Path root, OnDiskBlobStoreCache cache, DigestUtil digestUtil) {
    this.root = root;
    this.cache = cache;
    this.digestUtil = digestUtil;
  }

  private static Path child(Path root, String relativePath) throws IOException {
    var fragment = PathFragment.create(unicodeToInternal(relativePath));
    if (fragment.isAbsolute() || fragment.containsUplevelReferences()) {
      throw new IOException("Expected a relative output path: " + relativePath);
    }
    return root.getRelative(fragment);
  }

  private Path output(String buildId, String relativePath) throws IOException {
    var outputRoot = builds.get(buildId);
    if (outputRoot == null) {
      throw new IOException("Unknown build: " + buildId);
    }
    return child(outputRoot, relativePath);
  }

  private synchronized void record(String method, String path) throws IOException {
    FileSystemUtils.appendLinesAs(root.getChild("requests.log"), UTF_8, method + " " + path);
  }

  private static <T> void respond(StreamObserver<T> observer, Callable<T> request) {
    try {
      observer.onNext(request.call());
      observer.onCompleted();
    } catch (Exception e) {
      observer.onError(
          Status.INTERNAL.withDescription(e.toString()).withCause(e).asRuntimeException());
    }
  }

  @Override
  public void startBuild(StartBuildRequest request, StreamObserver<StartBuildResponse> observer) {
    respond(
        observer,
        () -> {
          if (!root.asFragment().equals(PathFragment.create(request.getOutputPathPrefix()))) {
            throw new IOException(
                "Unexpected output path prefix: " + request.getOutputPathPrefix());
          }
          var outputRoot = child(root, request.getOutputBaseId());
          outputRoot.createDirectoryAndParents();
          builds.put(request.getBuildId(), outputRoot);
          record("StartBuild", request.getBuildId());
          return StartBuildResponse.newBuilder()
              .setOutputPathSuffix(request.getOutputBaseId())
              .build();
        });
  }

  @Override
  public void clean(CleanRequest request, StreamObserver<CleanResponse> observer) {
    respond(
        observer,
        () -> {
          child(root, request.getOutputBaseId()).deleteTree();
          record("Clean", request.getOutputBaseId());
          return CleanResponse.getDefaultInstance();
        });
  }

  @Override
  public void stageArtifacts(
      StageArtifactsRequest request, StreamObserver<StageArtifactsResponse> observer) {
    respond(
        observer,
        () -> {
          var response = StageArtifactsResponse.newBuilder();
          var context = RemoteActionExecutionContext.create(RequestMetadata.getDefaultInstance());
          for (var artifact : request.getArtifactsList()) {
            var path = output(request.getBuildId(), artifact.getPath());
            var digest = artifact.getLocator().unpack(FileArtifactLocator.class).getDigest();
            path.getParentDirectory().createDirectoryAndParents();
            path.delete();
            getFromFuture(cache.downloadFile(context, path, digest));
            path.setExecutable(true);
            response.addResponses(StageArtifactsResponse.Response.getDefaultInstance());
            record("StageArtifacts", artifact.getPath());
          }
          return response.build();
        });
  }

  @Override
  public void finalizeArtifacts(
      FinalizeArtifactsRequest request, StreamObserver<FinalizeArtifactsResponse> observer) {
    respond(
        observer,
        () -> {
          for (var artifact : request.getArtifactsList()) {
            var path = output(request.getBuildId(), artifact.getPath());
            var expected = artifact.getLocator().unpack(FileArtifactLocator.class).getDigest();
            if (!digestUtil.compute(path, path.stat()).equals(expected)) {
              throw new IOException("Incorrect digest for finalized output: " + artifact.getPath());
            }
            record("FinalizeArtifacts", artifact.getPath());
          }
          return FinalizeArtifactsResponse.getDefaultInstance();
        });
  }

  @Override
  public void finalizeBuild(
      FinalizeBuildRequest request, StreamObserver<FinalizeBuildResponse> observer) {
    respond(
        observer,
        () -> {
          builds.remove(request.getBuildId());
          record("FinalizeBuild", request.getBuildId());
          return FinalizeBuildResponse.getDefaultInstance();
        });
  }

  @Override
  public void batchStat(BatchStatRequest request, StreamObserver<BatchStatResponse> observer) {
    respond(
        observer,
        () -> {
          var response = BatchStatResponse.newBuilder();
          for (var relativePath : request.getPathsList()) {
            var path = output(request.getBuildId(), relativePath);
            var stat = path.statIfFound(Symlinks.NOFOLLOW);
            var entry = response.addResponsesBuilder();
            if (stat != null) {
              var result = entry.getStatBuilder();
              if (stat.isSymbolicLink()) {
                result.getSymlinkBuilder().setTarget(path.readSymbolicLink().getPathString());
              } else if (stat.isDirectory()) {
                result.getDirectoryBuilder();
              } else if (stat.isFile()) {
                result
                    .getFileBuilder()
                    .setLocator(
                        Any.pack(
                            FileArtifactLocator.newBuilder()
                                .setDigest(digestUtil.compute(path, stat))
                                .build()));
              }
            }
            record("BatchStat", relativePath);
          }
          return response.build();
        });
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("Usage: TestOutputService <output-root> <worker-cas>");
    }
    var fs = RemoteWorker.getFileSystem();
    var root = fs.getPath(unicodeToInternal(args[0]));
    root.createDirectoryAndParents();
    var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, fs.getDigestFunction());
    var cache =
        new OnDiskBlobStoreCache(
            fs.getPath(unicodeToInternal(args[1])),
            digestUtil,
            Options.getDefaults(RemoteWorkerOptions.class));
    var server =
        NettyServerBuilder.forPort(0)
            .addService(new TestOutputService(root, cache, digestUtil))
            .build()
            .start();
    var portFile = root.getChild("port.tmp");
    FileSystemUtils.writeContent(portFile, UTF_8, Integer.toString(server.getPort()));
    portFile.renameTo(root.getChild("port"));
    server.awaitTermination();
  }
}
