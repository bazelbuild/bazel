// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Strategy that uses a distributed cache for sharing action input and output files.
 * Optionally this strategy also support offloading the work to a remote worker.
 */
@ExecutionStrategy(
  name = {"remote"},
  contextType = SpawnActionContext.class
)
final class RemoteSpawnStrategy implements SpawnActionContext {
  private final Path execRoot;
  private final StandaloneSpawnStrategy standaloneStrategy;
  private final RemoteActionCache remoteActionCache;
  private final RemoteWorkExecutor remoteWorkExecutor;

  RemoteSpawnStrategy(
      Map<String, String> clientEnv,
      Path execRoot,
      RemoteOptions options,
      boolean verboseFailures,
      RemoteActionCache actionCache,
      RemoteWorkExecutor workExecutor) {
    this.execRoot = execRoot;
    this.standaloneStrategy = new StandaloneSpawnStrategy(execRoot, verboseFailures);
    this.remoteActionCache = actionCache;
    this.remoteWorkExecutor = workExecutor;
  }

  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException {
    if (!spawn.isRemotable()) {
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    Executor executor = actionExecutionContext.getExecutor();
    ActionMetadata actionMetadata = spawn.getResourceOwner();
    ActionInputFileCache inputFileCache = actionExecutionContext.getActionInputFileCache();
    EventHandler eventHandler = executor.getEventHandler();

    // Compute a hash code to uniquely identify the action plus the action inputs.
    Hasher hasher = Hashing.sha256().newHasher();

    // TODO(alpha): The action key is usually computed using the path to the tool and the
    // arguments. It does not take into account the content / version of the system tool (e.g. gcc).
    // Either I put information about the system tools in the hash or assume tools are always
    // checked in.
    Preconditions.checkNotNull(actionMetadata.getKey());
    hasher.putString(actionMetadata.getKey(), Charset.defaultCharset());

    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(
            spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());
    for (ActionInput input : inputs) {
      hasher.putString(input.getExecPathString(), Charset.defaultCharset());
      try {
        // TODO(alpha): The digest from ActionInputFileCache is used to detect local file
        // changes. It might not be sufficient to identify the input file globally in the
        // remote action cache. Consider upgrading this to a better hash algorithm with
        // less collision.
        hasher.putBytes(inputFileCache.getDigest(input).toByteArray());
      } catch (IOException e) {
        throw new UserExecException("Failed to get digest for input.", e);
      }
    }

    // Save the action output if found in the remote action cache.
    String actionOutputKey = hasher.hash().toString();

    // Timeout for running the remote spawn.
    int timeout = 120;
    String timeoutStr = spawn.getExecutionInfo().get("timeout");
    if (timeoutStr != null) {
      try {
        timeout = Integer.parseInt(timeoutStr);
      } catch (NumberFormatException e) {
        throw new UserExecException("could not parse timeout: ", e);
      }
    }

    try {
      // Look up action cache using |actionOutputKey|. Reuse the action output if it is found.
      if (writeActionOutput(spawn.getMnemonic(), actionOutputKey, eventHandler, true)) {
        return;
      }

      FileOutErr outErr = actionExecutionContext.getFileOutErr();
      if (executeWorkRemotely(
          inputFileCache,
          spawn.getMnemonic(),
          actionOutputKey,
          spawn.getArguments(),
          inputs,
          spawn.getEnvironment(),
          spawn.getOutputFiles(),
          timeout,
          eventHandler,
          outErr)) {
        return;
      }

      // If nothing works then run spawn locally.
      standaloneStrategy.exec(spawn, actionExecutionContext);
      if (remoteActionCache != null) {
        remoteActionCache.putActionOutput(actionOutputKey, spawn.getOutputFiles());
      }
    } catch (IOException e) {
      throw new UserExecException("Unexpected IO error.", e);
    } catch (UnsupportedOperationException e) {
      eventHandler.handle(
          Event.warn(spawn.getMnemonic() + " unsupported operation for action cache (" + e + ")"));
    }
  }

  /**
   * Submit work to execute remotely.
   *
   * @return True in case the action succeeded and all expected action outputs are found.
   */
  private boolean executeWorkRemotely(
      ActionInputFileCache actionCache,
      String mnemonic,
      String actionOutputKey,
      List<String> arguments,
      List<ActionInput> inputs,
      ImmutableMap<String, String> environment,
      Collection<? extends ActionInput> outputs,
      int timeout,
      EventHandler eventHandler,
      FileOutErr outErr)
      throws IOException {
    if (remoteWorkExecutor == null) {
      return false;
    }
    try {
      ListenableFuture<RemoteWorkExecutor.Response> future =
          remoteWorkExecutor.submit(
              execRoot,
              actionCache,
              actionOutputKey,
              arguments,
              inputs,
              environment,
              outputs,
              timeout);
      RemoteWorkExecutor.Response response = future.get(timeout, TimeUnit.SECONDS);
      if (!response.success()) {
        String exception = "";
        if (!response.getException().isEmpty()) {
          exception = " (" + response.getException() + ")";
        }
        eventHandler.handle(
            Event.warn(
                mnemonic + " failed to execute work remotely" + exception + ", running locally"));
        return false;
      }
      if (response.getOut() != null) {
        outErr.printOut(response.getOut());
      }
      if (response.getErr() != null) {
        outErr.printErr(response.getErr());
      }
    } catch (ExecutionException e) {
      eventHandler.handle(
          Event.warn(mnemonic + " failed to execute work remotely (" + e + "), running locally"));
      return false;
    } catch (TimeoutException e) {
      eventHandler.handle(
          Event.warn(mnemonic + " timed out executing work remotely (" + e + "), running locally"));
      return false;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      eventHandler.handle(Event.warn(mnemonic + " remote work interrupted (" + e + ")"));
      return false;
    } catch (WorkTooLargeException e) {
      eventHandler.handle(Event.warn(mnemonic + " cannot be run remotely (" + e + ")"));
      return false;
    }
    return writeActionOutput(mnemonic, actionOutputKey, eventHandler, false);
  }

  /**
   * Saves the action output from cache. Returns true if all action outputs are found.
   */
  private boolean writeActionOutput(
      String mnemonic,
      String actionOutputKey,
      EventHandler eventHandler,
      boolean ignoreCacheNotFound)
      throws IOException {
    if (remoteActionCache == null) {
      return false;
    }
    try {
      remoteActionCache.writeActionOutput(actionOutputKey, execRoot);
      Event.info(mnemonic + " reuse action outputs from cache");
      return true;
    } catch (CacheNotFoundException e) {
      if (!ignoreCacheNotFound) {
        eventHandler.handle(
            Event.warn(mnemonic + " some cache entries cannot be found (" + e + ")"));
      }
    }
    return false;
  }

  @Override
  public boolean willExecuteRemotely(boolean remotable) {
    // Returning true here just helps to estimate the cost of this computation is zero.
    return remotable;
  }
}
