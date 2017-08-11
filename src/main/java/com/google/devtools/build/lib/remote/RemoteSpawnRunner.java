// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.Platform;
import io.grpc.Status.Code;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeSet;
import javax.annotation.Nullable;

/** A client for the remote execution service. */
@ThreadSafe
class RemoteSpawnRunner implements SpawnRunner {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;
  private final SpawnRunner fallbackRunner;
  private final boolean verboseFailures;

  @Nullable private final RemoteActionCache remoteCache;
  @Nullable private final GrpcRemoteExecutor remoteExecutor;

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions options,
      SpawnRunner fallbackRunner,
      boolean verboseFailures,
      @Nullable RemoteActionCache remoteCache,
      @Nullable GrpcRemoteExecutor remoteExecutor) {
    this.execRoot = execRoot;
    this.options = options;
    this.platform = options.parseRemotePlatformOverride();
    this.fallbackRunner = fallbackRunner;
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;
    this.verboseFailures = verboseFailures;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, InterruptedException, IOException {
    if (!spawn.isRemotable() || remoteCache == null) {
      return fallbackRunner.exec(spawn, policy);
    }

    policy.report(ProgressStatus.EXECUTING, "remote");
    // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
    ActionInputFileCache inputFileCache = policy.getActionInputFileCache();
    TreeNodeRepository repository = new TreeNodeRepository(execRoot, inputFileCache);
    SortedMap<PathFragment, ActionInput> inputMap = policy.getInputMapping();
    TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
    repository.computeMerkleDigests(inputRoot);
    Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
    Action action =
        buildAction(
            spawn.getOutputFiles(),
            Digests.computeDigest(command),
            repository.getMerkleDigest(inputRoot),
            platform,
            policy.getTimeout());

    // Look up action cache, and reuse the action output if it is found.
    ActionKey actionKey = Digests.computeActionKey(action);
    boolean acceptCachedResult = options.remoteAcceptCached && Spawns.mayBeCached(spawn);
    boolean uploadLocalResults = options.remoteUploadLocalResults;

    try {
      // Try to lookup the action in the action cache.
      ActionResult cachedResult =
          acceptCachedResult
              ? remoteCache.getCachedActionResult(actionKey)
              : null;
      if (cachedResult != null) {
        if (cachedResult.getExitCode() != 0) {
          // The remote cache must never serve a failed action.
          throw new EnvironmentalExecException("The remote cache is in an invalid state as it"
              + " served a failed action. Hash of the action: " + actionKey.getDigest());
        }
        try {
          return downloadRemoteResults(cachedResult, policy.getFileOutErr());
        } catch (CacheNotFoundException e) {
          // Intentionally left empty. No cache hit, so we fall through to local or
          // remote execution.
        }
      }
    } catch (IOException e) {
      return execLocallyOrFail(spawn, policy, inputMap, actionKey,
          options.remoteUploadLocalResults, e);
    }

    if (remoteExecutor == null) {
      // Remote execution is disabled and so execute the spawn on the local machine.
      return execLocally(spawn, policy, inputMap, uploadLocalResults, remoteCache, actionKey);
    }

    try {
      // Upload the command and all the inputs into the remote cache.
      remoteCache.ensureInputsPresent(repository, execRoot, inputRoot, command);
    } catch (IOException e) {
      return execLocallyOrFail(spawn, policy, inputMap, actionKey,
          options.remoteUploadLocalResults, e);
    }

    final ActionResult result;
    try {
      result = executeRemotely(action, inputMap.size(), acceptCachedResult);
    } catch (IOException e) {
      return execLocallyOrFail(spawn, policy, inputMap, actionKey, uploadLocalResults, e);
    }

    boolean executionFailed = result.getExitCode() != 0;
    if (options.remoteLocalFallback && executionFailed) {
      return execLocally(spawn, policy, inputMap, options.remoteUploadLocalResults,
          remoteCache, actionKey);
    }

    try {
      return downloadRemoteResults(result, policy.getFileOutErr());
    } catch (IOException e) {
      return execLocallyOrFail(spawn, policy, inputMap, actionKey,
          options.remoteUploadLocalResults, e);
    }
  }

  private SpawnResult downloadRemoteResults(ActionResult result, FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    remoteCache.download(result, execRoot, outErr);
    return new SpawnResult.Builder()
        .setStatus(Status.SUCCESS)  // Even if the action failed with non-zero exit code.
        .setExitCode(result.getExitCode())
        .build();
  }

  private ActionResult executeRemotely(Action action, int numInputFiles,
      boolean acceptCachedResult) throws IOException, InterruptedException {
    // TODO(olaola): set BuildInfo and input total bytes as well.
    ExecuteRequest.Builder request =
        ExecuteRequest.newBuilder()
            .setInstanceName(options.remoteInstanceName)
            .setAction(action)
            .setTotalInputFileCount(numInputFiles)
            .setSkipCacheLookup(!acceptCachedResult);
    ExecuteResponse reply = remoteExecutor.executeRemotely(request.build());
    return reply.getResult();
  }

  private SpawnResult execLocallyOrFail(Spawn spawn, SpawnExecutionPolicy policy,
      SortedMap<PathFragment, ActionInput> inputMap, ActionKey actionKey,
      boolean uploadLocalResults, IOException cause)
      throws ExecException, InterruptedException, IOException {
    if (options.remoteLocalFallback) {
      return execLocally(spawn, policy, inputMap, uploadLocalResults,
          remoteCache, actionKey);
    }
    throw new EnvironmentalExecException(errorMessage(cause), cause, true);
  }

  private String errorMessage(IOException e) {
    String message = "";
    if (e instanceof RetryException
        && ((RetryException) e).causedByStatusCode(Code.UNAVAILABLE)) {
      message = "The remote executor/cache is unavailable";
    } else if (e instanceof CacheNotFoundException) {
      message = "Failed to download from remote cache";
    } else {
      message = "Error in remote cache/executor";
    }
    // TODO(olaola): reuse the ErrorMessage class for these errors.
    if (verboseFailures) {
      message += "\n" + Throwables.getStackTraceAsString(e);
    }
    return message;
  }

  static Action buildAction(
      Collection<? extends ActionInput> outputs,
      Digest command,
      Digest inputRoot,
      Platform platform,
      Duration timeout) {
    Action.Builder action = Action.newBuilder();
    action.setCommandDigest(command);
    action.setInputRootDigest(inputRoot);
    ArrayList<String> outputPaths = new ArrayList<>();
    for (ActionInput output : outputs) {
      outputPaths.add(output.getExecPathString());
    }
    Collections.sort(outputPaths);
    // TODO: output directories should be handled here, when they are supported.
    action.addAllOutputFiles(outputPaths);
    if (platform != null) {
      action.setPlatform(platform);
    }
    if (!timeout.isZero()) {
      action.setTimeout(com.google.protobuf.Duration.newBuilder().setSeconds(timeout.getSeconds()));
    }
    return action.build();
  }

  static Command buildCommand(List<String> arguments, ImmutableMap<String, String> env) {
    Command.Builder command = Command.newBuilder();
    command.addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(env.keySet());
    for (String var : variables) {
      command.addEnvironmentVariablesBuilder().setName(var).setValue(env.get(var));
    }
    return command.build();
  }

  private Map<Path, Long> getInputCtimes(SortedMap<PathFragment, ActionInput> inputMap) {
    HashMap<Path, Long>  ctimes = new HashMap<>();
    for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
      ActionInput input = e.getValue();
      if (input == SpawnInputExpander.EMPTY_FILE || input instanceof VirtualActionInput) {
        continue;
      }
      Path path = execRoot.getRelative(input.getExecPathString());
      try {
        ctimes.put(path, path.stat().getLastChangeTime());
      } catch (IOException ex) {
        // Put a token value indicating an exception; this is used so that if the exception
        // is raised both before and after the execution, it is ignored, but if it is raised only
        // one of the times, it triggers a remote cache upload skip.
        ctimes.put(path, -1L);
      }
    }
    return ctimes;
  }

  /**
   * Execute a {@link Spawn} locally, using {@link #fallbackRunner}.
   *
   * <p>If possible also upload the {@link SpawnResult} to a remote cache.
   */
  private SpawnResult execLocally(
      Spawn spawn,
      SpawnExecutionPolicy policy,
      SortedMap<PathFragment, ActionInput> inputMap,
      boolean uploadToCache,
      @Nullable RemoteActionCache remoteCache,
      @Nullable ActionKey actionKey) throws ExecException, IOException, InterruptedException {
    if (uploadToCache && Spawns.mayBeCached(spawn) && remoteCache != null && actionKey != null) {
      return execLocallyAndUpload(spawn, policy, inputMap, remoteCache, actionKey);
    }
    return fallbackRunner.exec(spawn, policy);
  }

  @VisibleForTesting
  SpawnResult execLocallyAndUpload(
      Spawn spawn,
      SpawnExecutionPolicy policy,
      SortedMap<PathFragment, ActionInput> inputMap,
      RemoteActionCache remoteCache,
      ActionKey actionKey) throws ExecException, IOException, InterruptedException {
    Map<Path, Long> ctimesBefore = getInputCtimes(inputMap);
    SpawnResult result = fallbackRunner.exec(spawn, policy);
    if (!Status.SUCCESS.equals(result.status()) || result.exitCode() != 0) {
      // Don't upload failed actions.
      return result;
    }
    Map<Path, Long> ctimesAfter = getInputCtimes(inputMap);
    for (Map.Entry<Path, Long> e : ctimesBefore.entrySet()) {
      // Skip uploading to remote cache, because an input was modified during execution.
      if (!ctimesAfter.get(e.getKey()).equals(e.getValue())) {
        return result;
      }
    }
    List<Path> outputFiles = listExistingOutputFiles(execRoot, spawn);
    remoteCache.upload(actionKey, execRoot, outputFiles, policy.getFileOutErr());
    return result;
  }

  static List<Path> listExistingOutputFiles(Path execRoot, Spawn spawn) {
    ArrayList<Path> outputFiles = new ArrayList<>();
    for (ActionInput output : spawn.getOutputFiles()) {
      Path outputPath = execRoot.getRelative(output.getExecPathString());
      // TODO(ulfjack): Store the actual list of output files in SpawnResult and use that instead
      // of statting the files here again.
      if (outputPath.exists()) {
        outputFiles.add(outputPath);
      }
    }
    return outputFiles;
  }
}
