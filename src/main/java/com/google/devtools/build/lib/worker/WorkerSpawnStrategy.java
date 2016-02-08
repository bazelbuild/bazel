// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ChangedFilesMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A spawn action context that launches Spawns the first time they are used in a persistent mode and
 * then shards work over all the processes.
 */
@ExecutionStrategy(name = { "worker" }, contextType = SpawnActionContext.class)
final class WorkerSpawnStrategy implements SpawnActionContext {
  public static final String REASON_HEURISTIC =
      "Not using worker strategy, because incrementalHeuristic says no";
  public static final String REASON_NO_FLAGFILE =
      "Not using worker strategy, because last argument does not contain a @flagfile";
  public static final String REASON_NO_TOOLS =
      "Not using worker strategy, because the action has no tools";

  private final Path execRoot;
  private final WorkerPool workers;
  private final IncrementalHeuristic incrementalHeuristic;
  private final StandaloneSpawnStrategy standaloneStrategy;
  private final WorkerOptions options;
  private final boolean verboseFailures;
  private final int maxRetries;

  public WorkerSpawnStrategy(
      BlazeDirectories blazeDirs,
      OptionsClassProvider optionsProvider,
      EventBus eventBus,
      WorkerPool workers,
      boolean verboseFailures,
      int maxRetries) {
    Preconditions.checkNotNull(optionsProvider);
    this.options = optionsProvider.getOptions(WorkerOptions.class);
    this.incrementalHeuristic = new IncrementalHeuristic(options.workerMaxChangedFiles);
    eventBus.register(incrementalHeuristic);
    this.workers = Preconditions.checkNotNull(workers);
    this.standaloneStrategy = new StandaloneSpawnStrategy(blazeDirs.getExecRoot(), verboseFailures);
    this.execRoot = blazeDirs.getExecRoot();
    this.verboseFailures = verboseFailures;
    this.maxRetries = maxRetries;
  }

  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    EventHandler eventHandler = executor.getEventHandler();

    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(
          Label.print(spawn.getOwner().getLabel())
              + " ["
              + spawn.getResourceOwner().prettyPrint()
              + "]",
          spawn.asShellCommand(executor.getExecRoot()));
    }

    if (!incrementalHeuristic.shouldUseWorkers()) {
      if (options.workerVerbose) {
        eventHandler.handle(Event.info(REASON_HEURISTIC));
      }
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    // We assume that the spawn to be executed always gets a @flagfile argument, which contains the
    // flags related to the work itself (as opposed to start-up options for the executed tool).
    // Thus, we can extract the last element from its args (which will be the @flagfile), expand it
    // and put that into the WorkRequest instead.
    if (!Iterables.getLast(spawn.getArguments()).startsWith("@")) {
      if (options.workerVerbose) {
        eventHandler.handle(Event.info(REASON_NO_FLAGFILE));
      }
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    if (Iterables.isEmpty(spawn.getToolFiles())) {
      if (options.workerVerbose) {
        eventHandler.handle(Event.info(REASON_NO_TOOLS));
      }
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    executor
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "worker"));

    FileOutErr outErr = actionExecutionContext.getFileOutErr();

    ImmutableList<String> args = ImmutableList.<String>builder()
        .addAll(spawn.getArguments().subList(0, spawn.getArguments().size() - 1))
        .add("--persistent_worker")
        .build();
    ImmutableMap<String, String> env = spawn.getEnvironment();

    try {
      ActionInputFileCache inputFileCache = actionExecutionContext.getActionInputFileCache();

      HashCode workerFilesHash = combineActionInputHashes(spawn.getToolFiles(), inputFileCache);
      WorkerKey key = new WorkerKey(args, env, execRoot, spawn.getMnemonic(), workerFilesHash);

      WorkRequest.Builder requestBuilder = WorkRequest.newBuilder();
      expandArgument(requestBuilder, Iterables.getLast(spawn.getArguments()));

      List<ActionInput> inputs =
          ActionInputHelper.expandMiddlemen(
              spawn.getInputFiles(), actionExecutionContext.getMiddlemanExpander());

      for (ActionInput input : inputs) {
        ByteString digest = inputFileCache.getDigest(input);
        if (digest == null) {
          digest = ByteString.EMPTY;
        }

        requestBuilder
            .addInputsBuilder()
            .setPath(input.getExecPathString())
            .setDigest(digest)
            .build();
      }

      WorkResponse response = execInWorker(eventHandler, key, requestBuilder.build(), maxRetries);

      outErr.getErrorStream().write(response.getOutputBytes().toByteArray());

      if (response.getExitCode() != 0) {
        throw new UserExecException(
            String.format(
                "Worker process sent response with exit code: %d.", response.getExitCode()));
      }
    } catch (IOException e) {
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, spawn.getArguments(), env, execRoot.getPathString());
      throw new UserExecException(message, e);
    }
  }

  /**
   * Recursively expands arguments by replacing @filename args with the contents of the referenced
   * files. The @ itself can be escaped with @@.
   *
   * @param requestBuilder the WorkRequest.Builder that the arguments should be added to.
   * @param arg the argument to expand.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  private void expandArgument(WorkRequest.Builder requestBuilder, String arg) throws IOException {
    if (arg.startsWith("@") && !arg.startsWith("@@")) {
      for (String line : Files.readAllLines(
          Paths.get(execRoot.getRelative(arg.substring(1)).getPathString()), UTF_8)) {
        if (line.length() > 0) {
          expandArgument(requestBuilder, line);
        }
      }
    } else {
      requestBuilder.addArguments(arg);
    }
  }

  private HashCode combineActionInputHashes(
      Iterable<? extends ActionInput> toolFiles, ActionInputFileCache actionInputFileCache)
      throws IOException {
    Hasher hasher = Hashing.sha256().newHasher();
    for (ActionInput tool : toolFiles) {
      hasher.putString(tool.getExecPathString(), Charset.defaultCharset());
      hasher.putBytes(actionInputFileCache.getDigest(tool).toByteArray());
    }
    return hasher.hash();
  }

  private WorkResponse execInWorker(
      EventHandler eventHandler, WorkerKey key, WorkRequest request, int retriesLeft)
      throws IOException, InterruptedException, UserExecException {
    Worker worker = null;
    WorkResponse response = null;

    try {
      worker = workers.borrowObject(key);
      request.writeDelimitedTo(worker.getOutputStream());
      worker.getOutputStream().flush();

      response = WorkResponse.parseDelimitedFrom(worker.getInputStream());

      if (response == null) {
        throw new UserExecException(
            "Worker process did not return a correct WorkResponse. This is probably caused by a "
                + "bug in the worker, writing unexpected other data to stdout.");
      }
    } catch (IOException | InterruptedException e) {
      if (e instanceof InterruptedException) {
        // The user pressed Ctrl-C. Get out here quick.
        retriesLeft = 0;
      }

      if (worker != null) {
        workers.invalidateObject(key, worker);
        worker = null;
      }

      if (retriesLeft > 0) {
        // The worker process failed, but we still have some retries left. Let's retry with a fresh
        // worker.
        eventHandler.handle(
            Event.warn(
                key.getMnemonic()
                    + " worker failed ("
                    + e
                    + "), invalidating and retrying with new worker..."));
        return execInWorker(eventHandler, key, request, retriesLeft - 1);
      } else {
        throw e;
      }
    } finally {
      if (worker != null) {
        workers.returnObject(key, worker);
      }
    }
    return response;
  }

  @Override
  public String toString() {
    return "worker";
  }

  @Override
  public boolean isRemotable(String mnemonic, boolean remotable) {
    return false;
  }

  /**
   * For installation with remote execution, non-incremental builds may be slowed down by the
   * persistent worker system. To avoid this we only use workers for builds where few files
   * changed.
   */
  @ThreadSafety.ThreadSafe
  private static class IncrementalHeuristic {
    private final AtomicBoolean fewFilesChanged = new AtomicBoolean(false);
    private int limit = 0;

    public boolean shouldUseWorkers() {
      return limit == 0 || fewFilesChanged.get();
    }

    IncrementalHeuristic(int limit) {
      this.limit = limit;
    }

    @Subscribe
    public void changedFiles(ChangedFilesMessage msg) {
      fewFilesChanged.set(msg.getChangedFiles().size() <= limit);
    }
  }
}
