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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.hash.Hashing.md5;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.DigestFunction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.events.EventHandler;
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
import com.google.devtools.build.lib.remote.BazelOutputServiceREv2Proto.StartBuildArgs;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.FileMetadata;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.XattrProvider;
import com.google.devtools.build.lib.vfs.XattrProvider.DelegatingXattrProvider;
import com.google.protobuf.Any;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Output service implementation for the remote build with local output service daemon. */
public class BazelOutputService implements OutputService {

  private final String outputBaseId;
  private final Supplier<Path> execRootSupplier;
  private final Supplier<Path> outputPathSupplier;
  private final DigestFunction.Value digestFunction;
  private final RemoteOptions remoteOptions;
  private final boolean verboseFailures;
  private final RemoteRetrier retrier;
  private final ReferenceCountedChannel channel;

  @Nullable private String buildId;
  @Nullable private PathFragment outputPathTarget;

  public BazelOutputService(
      Path outputBase,
      Supplier<Path> execRootSupplier,
      Supplier<Path> outputPathSupplier,
      DigestFunction.Value digestFunction,
      RemoteOptions remoteOptions,
      boolean verboseFailures,
      RemoteRetrier retrier,
      ReferenceCountedChannel channel) {
    this.outputBaseId = DigestUtil.hashCodeToString(md5().hashString(outputBase.toString(), UTF_8));
    this.execRootSupplier = execRootSupplier;
    this.outputPathSupplier = outputPathSupplier;
    this.digestFunction = digestFunction;
    this.remoteOptions = remoteOptions;
    this.verboseFailures = verboseFailures;
    this.retrier = retrier;
    this.channel = channel;
  }

  public void shutdown() {
    channel.release();
  }

  @Override
  public String getFilesSystemName() {
    return "BazelOutputService";
  }

  private void prepareOutputPath(Path outputPath, PathFragment target) throws AbruptExitException {
    // Plant a symlink at bazel-out pointing to the target returned from the remote output service.
    try {
      if (!outputPath.isSymbolicLink()) {
        outputPath.deleteTree();
      }
      FileSystemUtils.ensureSymbolicLink(outputPath, target);
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format("Failed to plant output path symlink: %s", e.getMessage()))
                  .setExecution(
                      Execution.newBuilder().setCode(Code.LOCAL_OUTPUT_DIRECTORY_SYMLINK_FAILURE))
                  .build()),
          e);
    }
  }

  private PathFragment constructOutputPathTarget(
      PathFragment outputPathPrefix, StartBuildResponse response) throws AbruptExitException {
    var outputPathSuffix = PathFragment.create(response.getOutputPathSuffix());
    if (outputPathPrefix.isEmpty() && !outputPathSuffix.isAbsolute()) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "Expect StartBuildResponse.output_path_suffix to be an absolute path"
                              + " (because StartBuildRequest.output_path_prefix is empty), got %s.",
                          outputPathSuffix.isEmpty()
                              ? "an empty string"
                              : response.getOutputPathSuffix()))
                  .setExecution(Execution.newBuilder().setCode(Code.EXECUTION_UNKNOWN))
                  .build()));
    } else if (outputPathSuffix.isAbsolute()) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "Expect StartBuildResponse.output_path_suffix to be a relative path, got"
                              + " %s.",
                          response.getOutputPathSuffix()))
                  .setExecution(Execution.newBuilder().setCode(Code.EXECUTION_UNKNOWN))
                  .build()));
    } else if (outputPathSuffix.containsUplevelReferences()) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "Expect normalized StartBuildResponse.output_path_suffix to not contain"
                              + " uplevel references, got %s.",
                          outputPathSuffix))
                  .setExecution(Execution.newBuilder().setCode(Code.EXECUTION_UNKNOWN))
                  .build()));
    }

    var outputPathTarget = outputPathPrefix.getRelative(outputPathSuffix);
    checkState(outputPathTarget.isAbsolute());
    return outputPathTarget;
  }

  @Override
  public ModifiedFileSet startBuild(
      EventHandler eventHandler, UUID buildId, boolean finalizeActions)
      throws AbruptExitException, InterruptedException {
    checkState(this.buildId == null, "this.buildId must be null");
    this.buildId = buildId.toString();
    var outputPathPrefix = PathFragment.create(remoteOptions.remoteOutputServiceOutputPathPrefix);
    if (!outputPathPrefix.isEmpty() && !outputPathPrefix.isAbsolute()) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "--experimental_remote_output_service_path_prefix must be an absolute"
                              + " path, got '%s'",
                          outputPathPrefix))
                  .setExecution(Execution.newBuilder().setCode(Code.EXECUTION_UNKNOWN))
                  .build()));
    }
    var outputPath = outputPathSupplier.get();

    // Notify the remote output service that the build is about to start. The remote output service
    // will return the directory in which it wants us to let the build take place.
    var request =
        StartBuildRequest.newBuilder()
            .setVersion(1)
            .setOutputBaseId(outputBaseId)
            .setBuildId(this.buildId)
            .setArgs(
                Any.pack(
                    StartBuildArgs.newBuilder()
                        .setRemoteCache(remoteOptions.remoteCache)
                        .setInstanceName(remoteOptions.remoteInstanceName)
                        .setDigestFunction(digestFunction)
                        .build()))
            .setOutputPathPrefix(outputPathPrefix.toString())
            .putOutputPathAliases(outputPath.toString(), ".")
            .build();

    StartBuildResponse response;
    try {
      response = startBuild(request);
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "StartBuild failed: %s", Utils.grpcAwareErrorMessage(e, verboseFailures)))
                  .setExecution(Execution.newBuilder().setCode(Code.EXECUTION_UNKNOWN))
                  .build()));
    }

    checkState(outputPathTarget == null, "outputPathTarget must be null");
    outputPathTarget = constructOutputPathTarget(outputPathPrefix, response);
    prepareOutputPath(outputPath, outputPathTarget);

    if (finalizeActions) {
      // TODO(chiwang): Handle StartBuildResponse.initial_output_path_contents
    }

    return ModifiedFileSet.EVERYTHING_MODIFIED;
  }

  private StartBuildResponse startBuild(StartBuildRequest request)
      throws IOException, InterruptedException {
    return retrier.execute(
        () ->
            channel.withChannelBlocking(
                channel -> {
                  try {
                    return BazelOutputServiceGrpc.newBlockingStub(channel).startBuild(request);
                  } catch (StatusRuntimeException e) {
                    throw new IOException(e);
                  }
                }));
  }

  protected void stageArtifacts(List<FileMetadata> files) throws IOException, InterruptedException {
    var outputPath = outputPathSupplier.get();
    var request = StageArtifactsRequest.newBuilder();
    request.setBuildId(buildId);
    for (var file : files) {
      request.addArtifacts(
          StageArtifactsRequest.Artifact.newBuilder()
              .setPath(file.path().relativeTo(outputPath).toString())
              .setLocator(
                  Any.pack(FileArtifactLocator.newBuilder().setDigest(file.digest()).build()))
              .build());
    }
    var response = stageArtifacts(request.build());
    if (response.getResponsesCount() != files.size()) {
      throw new IOException(
          String.format(
              "StageArtifacts failed: expect %s responses from StageArtifactsResponse, got %s",
              files.size(), response.getResponsesCount()));
    }

    for (var i = 0; i < files.size(); ++i) {
      var fileResponse = response.getResponses(i);
      if (fileResponse.getStatus().getCode() != Status.Code.OK.value()) {
        throw new IOException(
            String.format(
                "Failed to stage %s, code: %s",
                files.get(i).path().relativeTo(outputPath), fileResponse.getStatus()));
      }
    }
  }

  private StageArtifactsResponse stageArtifacts(StageArtifactsRequest request)
      throws IOException, InterruptedException {
    return retrier.execute(
        () ->
            channel.withChannelBlocking(
                channel -> {
                  try {
                    return BazelOutputServiceGrpc.newBlockingStub(channel).stageArtifacts(request);
                  } catch (StatusRuntimeException e) {
                    throw new IOException(e);
                  }
                }));
  }

  @Override
  public void finalizeBuild(boolean buildSuccessful)
      throws AbruptExitException, InterruptedException {
    var request =
        FinalizeBuildRequest.newBuilder()
            .setBuildId(checkNotNull(buildId))
            .setBuildSuccessful(buildSuccessful)
            .build();
    try {
      var unused = finalizeBuild(request);
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format(
                          "FinalizeBuild failed: %s",
                          Utils.grpcAwareErrorMessage(e, verboseFailures)))
                  .setExecution(Execution.newBuilder().setCode(Code.EXECUTION_UNKNOWN))
                  .build()));
    } finally {
      this.buildId = null;
      this.outputPathTarget = null;
    }
  }

  private FinalizeBuildResponse finalizeBuild(FinalizeBuildRequest request)
      throws IOException, InterruptedException {
    return retrier.execute(
        () ->
            channel.withChannelBlocking(
                channel -> {
                  try {
                    return BazelOutputServiceGrpc.newBlockingStub(channel).finalizeBuild(request);
                  } catch (StatusRuntimeException e) {
                    throw new IOException(e);
                  }
                }));
  }

  @Override
  public void finalizeAction(Action action, OutputMetadataStore outputMetadataStore)
      throws IOException, EnvironmentalExecException, InterruptedException {
    var execRoot = execRootSupplier.get();
    var outputPath = outputPathSupplier.get();

    var request = FinalizeArtifactsRequest.newBuilder();
    request.setBuildId(buildId);
    for (var output : action.getOutputs()) {
      if (outputMetadataStore.artifactOmitted(output)) {
        continue;
      }

      if (output.isTreeArtifact()) {
        // TODO(chiwang): Use TreeArtifactLocator
        var children = outputMetadataStore.getTreeArtifactChildren((SpecialArtifact) output);
        for (var child : children) {
          addArtifact(outputMetadataStore, execRoot, outputPath, request, child);
        }
      } else {
        addArtifact(outputMetadataStore, execRoot, outputPath, request, output);
      }
    }

    var unused = finalizeArtifacts(request.build());
  }

  private FinalizeArtifactsResponse finalizeArtifacts(FinalizeArtifactsRequest request)
      throws IOException, InterruptedException {
    return retrier.execute(
        () ->
            channel.withChannelBlocking(
                channel -> {
                  try {
                    return BazelOutputServiceGrpc.newBlockingStub(channel)
                        .finalizeArtifacts(request);
                  } catch (StatusRuntimeException e) {
                    throw new IOException(e);
                  }
                }));
  }

  private void addArtifact(
      OutputMetadataStore outputMetadataStore,
      Path execRoot,
      Path outputPath,
      FinalizeArtifactsRequest.Builder builder,
      Artifact output)
      throws IOException, InterruptedException {
    checkState(!output.isTreeArtifact());
    var metadata = outputMetadataStore.getOutputMetadata(output);
    if (metadata.getType().isFile()) {
      var digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
      var path = execRoot.getRelative(output.getExecPath()).relativeTo(outputPath).toString();
      builder.addArtifacts(
          FinalizeArtifactsRequest.Artifact.newBuilder()
              .setPath(path)
              .setLocator(Any.pack(FileArtifactLocator.newBuilder().setDigest(digest).build()))
              .build());
    }
  }

  @Override
  public BatchStat getBatchStatter() {
    // TODO(chiwang): implement this
    return null;
  }

  @Override
  public boolean canCreateSymlinkTree() {
    return false;
  }

  @Override
  public void createSymlinkTree(
      Map<PathFragment, PathFragment> symlinks, PathFragment symlinkTreeRoot) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clean() throws ExecException, InterruptedException {
    var request = CleanRequest.newBuilder().setOutputBaseId(outputBaseId).build();
    try {
      var unused = clean(request);
    } catch (IOException e) {
      throw new EnvironmentalExecException(e, Code.UNEXPECTED_EXCEPTION);
    }
  }

  private CleanResponse clean(CleanRequest request) throws IOException, InterruptedException {
    return retrier.execute(
        () ->
            channel.withChannelBlocking(
                channel -> {
                  try {
                    return BazelOutputServiceGrpc.newBlockingStub(channel).clean(request);
                  } catch (StatusRuntimeException e) {
                    throw new IOException(e);
                  }
                }));
  }

  private BatchStatResponse batchStat(BatchStatRequest request)
      throws IOException, InterruptedException {
    return retrier.execute(
        () ->
            channel.withChannelBlocking(
                channel -> {
                  try {
                    return BazelOutputServiceGrpc.newBlockingStub(channel).batchStat(request);
                  } catch (StatusRuntimeException e) {
                    throw new IOException(e);
                  }
                }));
  }

  @Override
  public XattrProvider getXattrProvider(XattrProvider delegate) {
    return new DelegatingXattrProvider(delegate) {
      @Nullable
      @Override
      public byte[] getFastDigest(Path path) throws IOException {
        var outputPath = outputPathSupplier.get();
        var buildId = checkNotNull(BazelOutputService.this.buildId);
        var outputPathTarget = checkNotNull(BazelOutputService.this.outputPathTarget);

        String pathString = null;
        if (path.startsWith(outputPath)) {
          pathString = path.relativeTo(outputPath).toString();
        } else if (path.startsWith(outputPathTarget)) {
          pathString = path.asFragment().relativeTo(outputPathTarget).toString();
        }
        if (pathString == null) {
          return super.getFastDigest(path);
        }

        var request =
            BatchStatRequest.newBuilder().setBuildId(buildId).addPaths(pathString).build();
        BatchStatResponse response;
        try {
          response = batchStat(request);
        } catch (InterruptedException e) {
          throw new IOException(e);
        }

        if (response.getResponsesCount() != 1) {
          throw new IOException(
              String.format(
                  "BatchStat failed: expect 1 response, got %s", response.getResponsesCount()));
        }

        var statResponse = response.getResponses(0);
        if (!statResponse.hasStat()) {
          throw new FileNotFoundException(path.getPathString());
        }

        var stat = statResponse.getStat();
        if (stat.hasFile()) {
          var file = stat.getFile();
          if (file.hasLocator()) {
            var locator = file.getLocator().unpack(FileArtifactLocator.class);
            return DigestUtil.toBinaryDigest(locator.getDigest());
          }
        }

        return null;
      }
    };
  }
}
