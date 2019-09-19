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
package com.google.devtools.build.lib.dynamic;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Throwables;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext.StopConcurrentSpawns;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * A spawn strategy that speeds up incremental builds while not slowing down full builds.
 *
 * <p>This strategy tries to run spawn actions on the local and remote machine at the same time, and
 * picks the spawn action that completes first. This gives the benefits of remote execution on full
 * builds, and local execution on incremental builds.
 *
 * <p>One might ask, why we don't run spawns on the workstation all the time and just "spill over"
 * actions to remote execution when there are no local resources available. This would work, except
 * that the cost of transferring action inputs and outputs from the local machine to and from remote
 * executors over the network is way too high - there is no point in executing an action locally and
 * save 0.5s of time, when it then takes us 5 seconds to upload the results to remote executors for
 * another action that's scheduled to run there.
 */
@ExecutionStrategy(
    name = {"dynamic", "dynamic_worker"},
    contextType = SpawnActionContext.class)
public class DynamicSpawnStrategy implements SpawnActionContext {
  private static final Logger logger = Logger.getLogger(DynamicSpawnStrategy.class.getName());

  private final ListeningExecutorService executorService;
  private final DynamicExecutionOptions options;
  private final Function<Spawn, ExecutionPolicy> getExecutionPolicy;

  /**
   * Set to true by the first action that completes remotely. Until that happens, all local actions
   * are delayed by the amount given in {@link DynamicExecutionOptions#localExecutionDelay}.
   *
   * <p>This is a rather simple approach to make it possible to score a cache hit on remote
   * execution before even trying to start the action locally. This saves resources that would
   * otherwise be wasted by continuously starting and immediately killing local processes. One
   * possibility for improvement would be to establish a reporting mechanism from strategies back to
   * here, where we delay starting locally until the remote strategy tells us that the action isn't
   * a cache hit.
   */
  private final AtomicBoolean delayLocalExecution = new AtomicBoolean(false);

  private Map<String, List<SandboxedSpawnActionContext>> localStrategiesByMnemonic;
  private Map<String, List<SandboxedSpawnActionContext>> remoteStrategiesByMnemonic;

  /**
   * Constructs a {@code DynamicSpawnStrategy}.
   *
   * @param executorService an {@link ExecutorService} that will be used to run Spawn actions.
   */
  public DynamicSpawnStrategy(
      ExecutorService executorService,
      DynamicExecutionOptions options,
      Function<Spawn, ExecutionPolicy> getExecutionPolicy) {
    this.executorService = MoreExecutors.listeningDecorator(executorService);
    this.options = options;
    this.getExecutionPolicy = getExecutionPolicy;
  }

  /**
   * Searches for a sandboxed spawn strategy with the given name.
   *
   * @param usedContexts the action contexts used during the build
   * @param name the name of the spawn strategy we are interested in
   * @return a sandboxed spawn strategy
   * @throws ExecutorInitException if the spawn strategy does not exist, or if it exists but is not
   *     sandboxed
   */
  private static SandboxedSpawnActionContext findStrategy(
      Iterable<ActionContext> usedContexts, String name) throws ExecutorInitException {
    for (ActionContext context : usedContexts) {
      ExecutionStrategy strategy = context.getClass().getAnnotation(ExecutionStrategy.class);
      if (strategy != null && Arrays.asList(strategy.name()).contains(name)) {
        if (!(context instanceof SandboxedSpawnActionContext)) {
          throw new ExecutorInitException("Requested strategy " + name + " exists but does not "
              + "support sandboxing");
        }
        return (SandboxedSpawnActionContext) context;
      }
    }
    throw new ExecutorInitException("Requested strategy " + name + " does not exist");
  }

  private static Map<String, List<SandboxedSpawnActionContext>> buildStrategiesMap(
      Iterable<ActionContext> usedContexts, List<Map.Entry<String, List<String>>> optionVals)
      throws ExecutorInitException {
    Map<String, List<SandboxedSpawnActionContext>> strategiesByMnemonic = new HashMap<>();
    for (Map.Entry<String, List<String>> entry : optionVals) {
      List<SandboxedSpawnActionContext> strategies = Lists.newArrayList();
      if (!entry.getValue().isEmpty()) {
        for (String element : entry.getValue()) {
          strategies.add(findStrategy(usedContexts, element));
        }
        strategiesByMnemonic.put(entry.getKey(), strategies);
      }
    }
    return strategiesByMnemonic;
  }

  @Override
  public void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {
    localStrategiesByMnemonic =
        buildStrategiesMap(usedContexts, DynamicExecutionModule.localStrategiesByMnemonic);
    remoteStrategiesByMnemonic =
        buildStrategiesMap(usedContexts, DynamicExecutionModule.remoteStrategiesByMnemonic);
  }

  /**
   * Cancels and waits for a branch (a spawn execution) to terminate.
   *
   * <p>This is intended to be used as the body of the {@link StopConcurrentSpawns} lambda passed to
   * the spawn runners.
   *
   * @param branch the future running the spawn
   * @param allow whether we are allowed to cancel the branch or not. This exists to prevent the
   *     case where each parallel branch wants to cancel each other at the same time, in which case
   *     we want to keep the result of one of them.
   * @param done semaphore that is expected to receive a permit once the future terminates (after
   *     {@link InterruptedException} bubbles up through its call stack)
   * @throws InterruptedException if we get interrupted for any reason trying to cancel the future
   */
  private static void stopBranch(Future<List<SpawnResult>> branch, Semaphore allow, Semaphore done)
      throws InterruptedException {
    // In theory, this should just be allow.acquire(), but doing so can lead to deadlocks. Note that
    // cancelling a future sets its cancellation bit but does not necessarily set its interrupted
    // bit, in which case an allow.acquire() will not throw InterruptedException. Similarly, if we
    // have any subtle bug related to the propagation of the interrupt bit within the branch we are
    // trying to stop, we'd hit this same condition.
    if (!allow.tryAcquire()) {
      throw new InterruptedException();
    }

    branch.cancel(true);
    done.acquire();
  }

  /**
   * Waits for a branch (a spawn execution) to complete.
   *
   * @param branch the future running the spawn
   * @return the spawn result if the execution terminated successfully, or null if the branch was
   *     cancelled
   * @throws ExecException the execution error of the spawn if it failed
   * @throws InterruptedException if we get interrupted while waiting for completion
   */
  @Nullable
  private static List<SpawnResult> waitBranch(Future<List<SpawnResult>> branch)
      throws ExecException, InterruptedException {
    try {
      return branch.get();
    } catch (CancellationException e) {
      return null;
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof ExecException) {
        throw (ExecException) cause;
      } else if (cause instanceof InterruptedException) {
        // If the branch was interrupted, it might be due to a user interrupt or due to our request
        // for cancellation. Assume the latter here because if this was actually a user interrupt,
        // our own get() would have been interrupted as well. It makes no sense to propagate the
        // interrupt status across threads.
        return null;
      } else {
        // Even though we cannot enforce this in the future's signature (but we do in Branch#call),
        // we only expect the exception types we validated above. Still, unchecked exceptions could
        // propagate, so just let them bubble up.
        Throwables.throwIfUnchecked(cause);
        throw new AssertionError("Unexpected exception type from strategy.exec()");
      }
    } catch (InterruptedException e) {
      branch.cancel(true);
      throw e;
    }
  }

  /**
   * Waits for the two branches of a spawn's execution to complete.
   *
   * <p>This guarantees that the two branches are stopped both on successful termination and on an
   * exception.
   *
   * @param branch1 the future running one side of the spawn (e.g. local). This future must cancel
   *     {@code branch2} at some point during its successful execution to guarantee termination. If
   *     we encounter an execution error, or if we are interrupted, then we handle such cancellation
   *     here.
   * @param branch2 the future running the other side of the spawn (e.g. remote). Same restrictions
   *     apply as in {@code branch1}, but in the symmetric direction.
   * @return the result of the branch that terminates first
   * @throws ExecException the execution error of the spawn that terminated first
   * @throws InterruptedException if we get interrupted while waiting for completion
   */
  private static List<SpawnResult> waitBranches(
      Future<List<SpawnResult>> branch1, Future<List<SpawnResult>> branch2)
      throws ExecException, InterruptedException {
    List<SpawnResult> result1;
    try {
      result1 = waitBranch(branch1);
    } catch (ExecException | InterruptedException | RuntimeException e) {
      branch2.cancel(true);
      throw e;
    }

    List<SpawnResult> result2 = waitBranch(branch2);

    if (result2 != null && result1 != null) {
      throw new AssertionError("One branch did not cancel the other one");
    } else if (result2 != null) {
      return result2;
    } else if (result1 != null) {
      return result1;
    } else {
      throw new AssertionError(
          "No branch completed, which probably means interrupts were not propagated correctly");
    }
  }

  @Override
  public List<SpawnResult> exec(
      final Spawn spawn, final ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    ExecutionPolicy executionPolicy = getExecutionPolicy.apply(spawn);
    if (executionPolicy.canRunLocallyOnly()) {
      return runLocally(spawn, actionExecutionContext, null);
    }
    if (executionPolicy.canRunRemotelyOnly()) {
      return runRemotely(spawn, actionExecutionContext, null);
    }

    // Semaphores to track termination of each branch. These are necessary to wait for the branch to
    // finish its own cleanup (e.g. terminating subprocesses) once it has been cancelled.
    Semaphore localDone = new Semaphore(0);
    Semaphore remoteDone = new Semaphore(0);

    Semaphore allowCancel = new Semaphore(1);
    SettableFuture<List<SpawnResult>> remoteBranch = SettableFuture.create();

    ListenableFuture<List<SpawnResult>> localBranch =
        executorService.submit(
            new Branch("local", actionExecutionContext) {
              @Override
              List<SpawnResult> callImpl(ActionExecutionContext context)
                  throws InterruptedException, ExecException {
                if (delayLocalExecution.get()) {
                  Thread.sleep(options.localExecutionDelay);
                }
                return runLocally(
                    spawn, context, () -> stopBranch(remoteBranch, allowCancel, remoteDone));
              }
            });
    localBranch.addListener(
        () -> {
          localDone.release();
          try {
            if (!localBranch.isCancelled()) {
              remoteBranch.cancel(true);
            }
          } catch (Exception e) {
            // Ignore. We should only get here on an interrupt, in which case the local branch
            // should have been cancelled already.
          }
        },
        MoreExecutors.directExecutor());

    remoteBranch.setFuture(
        executorService.submit(
            new Branch("remote", actionExecutionContext) {
              @Override
              public List<SpawnResult> callImpl(ActionExecutionContext context)
                  throws InterruptedException, ExecException {
                List<SpawnResult> spawnResults =
                    runRemotely(
                        spawn, context, () -> stopBranch(localBranch, allowCancel, localDone));
                delayLocalExecution.set(true);
                return spawnResults;
              }
            }));
    remoteBranch.addListener(
        () -> {
          remoteDone.release();
          if (!remoteBranch.isCancelled()) {
            localBranch.cancel(true);
          }
        },
        MoreExecutors.directExecutor());

    try {
      return waitBranches(localBranch, remoteBranch);
    } finally {
      checkState(localBranch.isDone());
      checkState(remoteBranch.isDone());
    }
  }

  private static List<SandboxedSpawnActionContext> getValidStrategies(
      Map<String, List<SandboxedSpawnActionContext>> strategiesByMnemonic, Spawn spawn) {
    List<SandboxedSpawnActionContext> validStrategies = Lists.newArrayList();
    if (strategiesByMnemonic.get(spawn.getMnemonic()) != null) {
      validStrategies.addAll(strategiesByMnemonic.get(spawn.getMnemonic()));
    }
    if (strategiesByMnemonic.get("") != null) {
      validStrategies.addAll(strategiesByMnemonic.get(""));
    }
    return validStrategies;
  }

  @Override
  public boolean canExec(Spawn spawn) {
    for (SandboxedSpawnActionContext strategy :
        getValidStrategies(localStrategiesByMnemonic, spawn)) {
      if (strategy.canExec(spawn)) {
        return true;
      }
    }
    for (SandboxedSpawnActionContext strategy :
        getValidStrategies(remoteStrategiesByMnemonic, spawn)) {
      if (strategy.canExec(spawn)) {
        return true;
      }
    }
    return false;
  }

  private static FileOutErr getSuffixedFileOutErr(FileOutErr fileOutErr, String suffix) {
    Path outDir = checkNotNull(fileOutErr.getOutputPath().getParentDirectory());
    String outBaseName = fileOutErr.getOutputPath().getBaseName();
    Path errDir = checkNotNull(fileOutErr.getErrorPath().getParentDirectory());
    String errBaseName = fileOutErr.getErrorPath().getBaseName();
    return new FileOutErr(
        outDir.getChild(outBaseName + suffix), errDir.getChild(errBaseName + suffix));
  }

  private List<SpawnResult> runLocally(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      @Nullable StopConcurrentSpawns stopConcurrentSpawns)
      throws ExecException, InterruptedException {
    for (SandboxedSpawnActionContext strategy :
        getValidStrategies(localStrategiesByMnemonic, spawn)) {
      return strategy.exec(spawn, actionExecutionContext, stopConcurrentSpawns);
    }
    throw new RuntimeException(
        "executorCreated not yet called or no default dynamic_local_strategy set");
  }

  private List<SpawnResult> runRemotely(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      @Nullable StopConcurrentSpawns stopConcurrentSpawns)
      throws ExecException, InterruptedException {
    for (SandboxedSpawnActionContext strategy :
        getValidStrategies(remoteStrategiesByMnemonic, spawn)) {
      return strategy.exec(spawn, actionExecutionContext, stopConcurrentSpawns);
    }
    throw new RuntimeException(
        "executorCreated not yet called or no default dynamic_remote_strategy set");
  }

  /**
   * Wraps the execution of a function that is supposed to execute a spawn via a strategy and only
   * updates the stdout/stderr files if this spawn succeeds.
   */
  private abstract static class Branch implements Callable<List<SpawnResult>> {
    private final String name;
    private final ActionExecutionContext context;

    /**
     * Creates a new branch of dynamic execution.
     *
     * @param name a name to describe what this branch represents (e.g. {@code remote}). Used to
     *     qualify temporary files.
     * @param context the action execution context given to the dynamic strategy, used to obtain the
     *     final location of the stdout/stderr
     */
    Branch(String name, ActionExecutionContext context) {
      this.name = name;
      this.context = context;
    }

    /**
     * Moves a set of stdout/stderr files over another one. Errors during the move are logged and
     * swallowed.
     *
     * @param from the source location
     * @param to the target location
     */
    private static void moveFileOutErr(FileOutErr from, FileOutErr to) {
      try {
        if (from.getOutputPath().exists()) {
          Files.move(from.getOutputPath().getPathFile(), to.getOutputPath().getPathFile());
        }
        if (from.getErrorPath().exists()) {
          Files.move(from.getErrorPath().getPathFile(), to.getErrorPath().getPathFile());
        }
      } catch (IOException e) {
        logger.log(Level.WARNING, "Could not move action logs from execution", e);
      }
    }

    /**
     * Hook to execute a spawn using an arbitrary strategy.
     *
     * @param context the action execution context where the spawn can write its stdout/stderr. The
     *     location of these files is specific to this branch.
     * @return the spawn results if execution was successful
     * @throws InterruptedException if the branch was cancelled or an interrupt was caught
     * @throws ExecException if the spawn execution fails
     */
    abstract List<SpawnResult> callImpl(ActionExecutionContext context)
        throws InterruptedException, ExecException;

    /**
     * Executes the {@link #callImpl} hook and handles stdout/stderr.
     *
     * @return the spawn results if execution was successful
     * @throws InterruptedException if the branch was cancelled or an interrupt was caught
     * @throws ExecException if the spawn execution fails
     */
    @Override
    public final List<SpawnResult> call() throws InterruptedException, ExecException {
      FileOutErr fileOutErr = getSuffixedFileOutErr(context.getFileOutErr(), "." + name);

      List<SpawnResult> results = null;
      ExecException exception = null;
      try {
        results = callImpl(context.withFileOutErr(fileOutErr));
      } catch (ExecException e) {
        exception = e;
      } finally {
        try {
          fileOutErr.close();
        } catch (IOException ignored) {
          // Nothing we can do here.
        }
      }

      moveFileOutErr(fileOutErr, context.getFileOutErr());

      if (exception != null) {
        throw exception;
      } else {
        checkNotNull(results);
        return results;
      }
    }
  }
}
