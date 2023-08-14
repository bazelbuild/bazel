// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import build.bazel.remote.execution.v2.Platform;
import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.XattrProvider;
import com.google.protobuf.util.Durations;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A logging utility for spawns.
 */
public class SpawnLogContext implements ActionContext {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Path execRoot;
  private final MessageOutputStream executionLog;
  @Nullable private final ExecutionOptions executionOptions;
  @Nullable private final RemoteOptions remoteOptions;
  private final XattrProvider xattrProvider;

  public SpawnLogContext(
      Path execRoot,
      MessageOutputStream executionLog,
      @Nullable ExecutionOptions executionOptions,
      @Nullable RemoteOptions remoteOptions,
      XattrProvider xattrProvider) {
    this.execRoot = execRoot;
    this.executionLog = executionLog;
    this.executionOptions = executionOptions;
    this.remoteOptions = remoteOptions;
    this.xattrProvider = xattrProvider;
  }

  /** Log the executed spawn to the output stream. */
  public void logSpawn(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      SortedMap<PathFragment, ActionInput> inputMap,
      Duration timeout,
      SpawnResult result)
      throws IOException, ExecException {
    SortedMap<Path, ActionInput> existingOutputs = listExistingOutputs(spawn);
    SpawnExec.Builder builder = SpawnExec.newBuilder();
    builder.addAllCommandArgs(spawn.getArguments());

    Map<String, String> env = spawn.getEnvironment();
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(env.keySet());
    for (String var : variables) {
      builder.addEnvironmentVariablesBuilder().setName(var).setValue(env.get(var));
    }

    try (SilentCloseable c = Profiler.instance().profile("logSpawn/inputs")) {
      for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
        ActionInput input = e.getValue();
        if (input instanceof VirtualActionInput.EmptyActionInput) {
          continue;
        }
        Path inputPath = execRoot.getRelative(input.getExecPathString());
        if (inputPath.isDirectory()) {
          listDirectoryContents(inputPath, builder::addInputs, inputMetadataProvider);
        } else {
          Digest digest = computeDigest(input, null, inputMetadataProvider, xattrProvider);
          builder.addInputsBuilder().setPath(input.getExecPathString()).setDigest(digest);
        }
      }
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Error computing spawn inputs");
    }
    try (SilentCloseable c = Profiler.instance().profile("logSpawn/outputs")) {
      ArrayList<String> outputPaths = new ArrayList<>();
      for (ActionInput output : spawn.getOutputFiles()) {
        outputPaths.add(output.getExecPathString());
      }
      Collections.sort(outputPaths);
      builder.addAllListedOutputs(outputPaths);
      for (Map.Entry<Path, ActionInput> e : existingOutputs.entrySet()) {
        Path path = e.getKey();
        if (path.isDirectory()) {
          listDirectoryContents(path, builder::addActualOutputs, inputMetadataProvider);
        } else {
          File.Builder outputBuilder = builder.addActualOutputsBuilder();
          outputBuilder.setPath(path.relativeTo(execRoot).toString());
          try {
            outputBuilder.setDigest(
                computeDigest(e.getValue(), path, inputMetadataProvider, xattrProvider));
          } catch (IOException ex) {
            logger.atWarning().withCause(ex).log("Error computing spawn event output properties");
          }
        }
      }
    }
    builder.setRemotable(Spawns.mayBeExecutedRemotely(spawn));

    Platform execPlatform = PlatformUtils.getPlatformProto(spawn, remoteOptions);
    if (execPlatform != null) {
      builder.setPlatform(buildPlatform(execPlatform));
    }
    if (result.status() != SpawnResult.Status.SUCCESS) {
      builder.setStatus(result.status().toString());
    }
    if (!timeout.isZero()) {
      builder.setTimeoutMillis(timeout.toMillis());
    }
    builder.setCacheable(Spawns.mayBeCached(spawn));
    builder.setRemoteCacheable(Spawns.mayBeCachedRemotely(spawn));
    builder.setExitCode(result.exitCode());
    builder.setRemoteCacheHit(result.isCacheHit());
    builder.setRunner(result.getRunnerName());
    if (result.getDigest() != null) {
      builder
          .getDigestBuilder()
          .setHash(result.getDigest().getHash())
          .setSizeBytes(result.getDigest().getSizeBytes());
    }

    String progressMessage = spawn.getResourceOwner().getProgressMessage();
    if (progressMessage != null) {
      builder.setProgressMessage(progressMessage);
    }
    builder.setMnemonic(spawn.getMnemonic());
    builder.setWalltime(millisToProto(result.getMetrics().executionWallTimeInMs()));

    if (spawn.getTargetLabel() != null) {
      builder.setTargetLabel(spawn.getTargetLabel());
    }

    if (executionOptions != null && executionOptions.executionLogSpawnMetrics) {
      SpawnMetrics metrics = result.getMetrics();
      Protos.SpawnMetrics.Builder metricsBuilder = builder.getMetricsBuilder();
      if (metrics.totalTimeInMs() != 0L) {
        metricsBuilder.setTotalTime(millisToProto(metrics.totalTimeInMs()));
      }
      if (metrics.parseTimeInMs() != 0L) {
        metricsBuilder.setParseTime(millisToProto(metrics.parseTimeInMs()));
      }
      if (metrics.networkTimeInMs() != 0L) {
        metricsBuilder.setNetworkTime(millisToProto(metrics.networkTimeInMs()));
      }
      if (metrics.fetchTimeInMs() != 0L) {
        metricsBuilder.setFetchTime(millisToProto(metrics.fetchTimeInMs()));
      }
      if (metrics.queueTimeInMs() != 0L) {
        metricsBuilder.setQueueTime(millisToProto(metrics.queueTimeInMs()));
      }
      if (metrics.setupTimeInMs() != 0L) {
        metricsBuilder.setSetupTime(millisToProto(metrics.setupTimeInMs()));
      }
      if (metrics.uploadTimeInMs() != 0L) {
        metricsBuilder.setUploadTime(millisToProto(metrics.uploadTimeInMs()));
      }
      if (metrics.executionWallTimeInMs() != 0L) {
        metricsBuilder.setExecutionWallTime(millisToProto(metrics.executionWallTimeInMs()));
      }
      if (metrics.processOutputsTimeInMs() != 0L) {
        metricsBuilder.setProcessOutputsTime(millisToProto(metrics.processOutputsTimeInMs()));
      }
      if (metrics.retryTimeInMs() != 0L) {
        metricsBuilder.setRetryTime(millisToProto(metrics.retryTimeInMs()));
      }
      metricsBuilder.setInputBytes(metrics.inputBytes());
      metricsBuilder.setInputFiles(metrics.inputFiles());
      metricsBuilder.setMemoryEstimateBytes(metrics.memoryEstimate());
      metricsBuilder.setInputBytesLimit(metrics.inputBytesLimit());
      metricsBuilder.setInputFilesLimit(metrics.inputFilesLimit());
      metricsBuilder.setOutputBytesLimit(metrics.outputBytesLimit());
      metricsBuilder.setOutputFilesLimit(metrics.outputFilesLimit());
      metricsBuilder.setMemoryBytesLimit(metrics.memoryLimit());
      if (metrics.timeLimitInMs() != 0L) {
        metricsBuilder.setTimeLimit(millisToProto(metrics.timeLimitInMs()));
      }
    }

    try (SilentCloseable c = Profiler.instance().profile("logSpawn/write")) {
      executionLog.write(builder.build());
    }
  }

  private static com.google.protobuf.Duration millisToProto(int t) {
    return Durations.fromMillis(t);
  }

  public void close() throws IOException {
    executionLog.close();
  }

  private static Protos.Platform buildPlatform(Platform platform) {
    Protos.Platform.Builder platformBuilder = Protos.Platform.newBuilder();
    for (Platform.Property p : platform.getPropertiesList()) {
      platformBuilder.addPropertiesBuilder().setName(p.getName()).setValue(p.getValue());
    }
    return platformBuilder.build();
  }

  private SortedMap<Path, ActionInput> listExistingOutputs(Spawn spawn) {
    TreeMap<Path, ActionInput> result = new TreeMap<>();
    for (ActionInput output : spawn.getOutputFiles()) {
      Path outputPath = execRoot.getRelative(output.getExecPathString());
      // TODO(olaola): once symlink API proposal is implemented, report symlinks here.
      if (outputPath.exists()) {
        result.put(outputPath, output);
      }
    }
    return result;
  }

  private void listDirectoryContents(
      Path path, Consumer<File> addFile, InputMetadataProvider inputMetadataProvider) {
    try {
      // TODO(olaola): once symlink API proposal is implemented, report symlinks here.
      List<Dirent> sortedDirent = new ArrayList<>(path.readdir(Symlinks.NOFOLLOW));
      sortedDirent.sort(Comparator.comparing(Dirent::getName));
      for (Dirent dirent : sortedDirent) {
        String name = dirent.getName();
        Path child = path.getRelative(name);
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          listDirectoryContents(child, addFile, inputMetadataProvider);
        } else {
          String pathString;
          if (child.startsWith(execRoot)) {
            pathString = child.relativeTo(execRoot).toString();
          } else {
            pathString = child.toString();
          }
          addFile.accept(
              File.newBuilder()
                  .setPath(pathString)
                  .setDigest(computeDigest(null, child, inputMetadataProvider, xattrProvider))
                  .build());
        }
      }
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Error computing spawn event file properties");
    }
  }

  /**
   * Computes the digest of the given ActionInput or corresponding path. Will try to access the
   * Metadata cache first, if it is available, and fall back to digesting the contents manually.
   */
  private Digest computeDigest(
      @Nullable ActionInput input,
      @Nullable Path path,
      InputMetadataProvider inputMetadataProvider,
      XattrProvider xattrProvider)
      throws IOException {
    Preconditions.checkArgument(input != null || path != null);
    DigestHashFunction hashFunction = execRoot.getFileSystem().getDigestFunction();
    Digest.Builder digest = Digest.newBuilder().setHashFunctionName(hashFunction.toString());
    if (input != null) {
      if (input instanceof VirtualActionInput) {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        ((VirtualActionInput) input).writeTo(buffer);
        byte[] blob = buffer.toByteArray();
        return digest
            .setHash(hashFunction.getHashFunction().hashBytes(blob).toString())
            .setSizeBytes(blob.length)
            .build();
      }
      // Try to access the cached metadata, otherwise fall back to local computation.
      try {
        FileArtifactValue metadata = inputMetadataProvider.getInputMetadata(input);
        if (metadata != null) {
          byte[] hash = metadata.getDigest();
          if (hash != null) {
            return digest
                .setHash(HashCode.fromBytes(hash).toString())
                .setSizeBytes(metadata.getSize())
                .build();
          }
        }
      } catch (IOException | IllegalStateException e) {
        // Pass through to local computation.
      }
    }
    if (path == null) {
      path = execRoot.getRelative(input.getExecPath());
    }
    // Compute digest manually.
    long fileSize = path.getFileSize();
    return digest
        .setHash(
            HashCode.fromBytes(
                    DigestUtils.getDigestWithManualFallback(path, fileSize, xattrProvider))
                .toString())
        .setSizeBytes(fileSize)
        .build();
  }

  public boolean shouldSort() {
    return executionOptions.executionLogSort;
  }
}
