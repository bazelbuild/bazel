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
import com.google.common.collect.ImmutableList;
import com.google.common.io.Files;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext.StopConcurrentSpawns;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
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
   * Cancels and waits for a branch (a spawn execution) to terminate.
   *
   * <p>This is intended to be used as the body of the {@link StopConcurrentSpawns} lambda passed to
   * the spawn runners. Each strategy may call this at most once.
   *
   * @param branch the future running the spawn
   * @param branchDone semaphore that is expected to receive a permit once the future terminates
   *     (after {@link InterruptedException} bubbles up through its call stack)
   * @param cancellingStrategy identifier of the strategy that is performing the cancellation. Used
   *     to prevent cross-cancellations and to sanity-check that the same strategy doesn't issue the
   *     cancellation twice.
   * @param strategyThatCancelled name of the first strategy that executed this method, or a null
   *     reference if this is the first time this method is called. If not null, we expect the value
   *     referenced by this to be different than {@code cancellingStrategy}, or else we have a bug.
   * @throws InterruptedException if we get interrupted for any reason trying to cancel the future
   * @throws DynamicInterruptedException if we lost a race against another strategy trying to cancel
   *     us
   */
  private static void stopBranch(
      Future<ImmutableList<SpawnResult>> branch,
      Semaphore branchDone,
      String cancellingStrategy,
      AtomicReference<String> strategyThatCancelled)
      throws InterruptedException {
    // This multi-step, unlocked access to "strategyThatCancelled" is valid because, for a given
    // value of "cancellingStrategy", we do not expect concurrent calls to this method. (If there
    // are, we are in big trouble.)
    String current = strategyThatCancelled.get();
    if (current != null && current.equals(cancellingStrategy)) {
      throw new AssertionError("stopBranch called more than once by " + cancellingStrategy);
    } else {
      // Protect against the two branches from cancelling each other. The first branch to set the
      // reference to its own identifier wins and is allowed to issue the cancellation; the other
      // branch just has to give up execution.
      if (strategyThatCancelled.compareAndSet(null, cancellingStrategy)) {
        boolean cancelled = branch.cancel(true);
        checkState(cancelled, "Failed to cancel other branch from %s", cancellingStrategy);
        branchDone.acquire();
      } else {
        throw new DynamicInterruptedException(
            "Execution stopped because other strategy finished first");
      }
    }
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
  private static ImmutableList<SpawnResult> waitBranch(Future<ImmutableList<SpawnResult>> branch)
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
  private static ImmutableList<SpawnResult> waitBranches(
      Future<ImmutableList<SpawnResult>> branch1, Future<ImmutableList<SpawnResult>> branch2)
      throws ExecException, InterruptedException {
    ImmutableList<SpawnResult> result1;
    try {
      result1 = waitBranch(branch1);
    } catch (ExecException | InterruptedException | RuntimeException e) {
      branch2.cancel(true);
      throw e;
    }

    ImmutableList<SpawnResult> result2 = waitBranch(branch2);

    if (result2 != null && result1 != null) {
      throw new AssertionError("One branch did not cancel the other one");
    } else if (result2 != null) {
      return result2;
    } else if (result1 != null) {
      return result1;
    } else {
      throw new AssertionError("No branch completed, which might mean they cancelled each other");
    }
  }

  @Override
  public ImmutableList<SpawnResult> exec(
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

    AtomicReference<String> strategyThatCancelled = new AtomicReference<>(null);
    SettableFuture<ImmutableList<SpawnResult>> remoteBranch = SettableFuture.create();

    AtomicBoolean localCanReportDone = new AtomicBoolean(false);
    AtomicBoolean remoteCanReportDone = new AtomicBoolean(false);

    ListenableFuture<ImmutableList<SpawnResult>> localBranch =
        executorService.submit(
            new Branch("local", actionExecutionContext) {
              @Override
              ImmutableList<SpawnResult> callImpl(ActionExecutionContext context)
                  throws InterruptedException, ExecException {
                try {
                  if (!localCanReportDone.compareAndSet(false, true)) {
                    // If we ever get here, it's because we were cancelled and the listener ran
                    // first. Just make sure that's the case.
                    checkState(Thread.interrupted());
                    throw new InterruptedException();
                  }
                  if (delayLocalExecution.get()) {
                    Thread.sleep(options.localExecutionDelay);
                  }
                  return runLocally(
                      spawn,
                      context,
                      () -> stopBranch(remoteBranch, remoteDone, "local", strategyThatCancelled));
                } finally {
                  localDone.release();
                }
              }
            });
    localBranch.addListener(
        () -> {
          if (localCanReportDone.compareAndSet(false, true)) {
            localDone.release();
          }
          if (!localBranch.isCancelled()) {
            remoteBranch.cancel(true);
          }
        },
        MoreExecutors.directExecutor());

    remoteBranch.setFuture(
        executorService.submit(
            new Branch("remote", actionExecutionContext) {
              @Override
              public ImmutableList<SpawnResult> callImpl(ActionExecutionContext context)
                  throws InterruptedException, ExecException {
                try {
                  if (!remoteCanReportDone.compareAndSet(false, true)) {
                    // If we ever get here, it's because we were cancelled and the listener ran
                    // first. Just make sure that's the case.
                    checkState(Thread.interrupted());
                    throw new InterruptedException();
                  }
                  ImmutableList<SpawnResult> spawnResults =
                      runRemotely(
                          spawn,
                          context,
                          () ->
                              stopBranch(localBranch, localDone, "remote", strategyThatCancelled));
                  delayLocalExecution.set(true);
                  return spawnResults;
                } finally {
                  remoteDone.release();
                }
              }
            }));
    remoteBranch.addListener(
        () -> {
          if (remoteCanReportDone.compareAndSet(false, true)) {
            remoteDone.release();
          }
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

  @Override
  public boolean canExec(Spawn spawn, ActionContextRegistry actionContextRegistry) {
    DynamicStrategyRegistry dynamicStrategyRegistry =
        actionContextRegistry.getContext(DynamicStrategyRegistry.class);
    for (SandboxedSpawnActionContext strategy :
        dynamicStrategyRegistry.getDynamicSpawnActionContexts(
            spawn, DynamicStrategyRegistry.DynamicMode.LOCAL)) {
      if (strategy.canExec(spawn, actionContextRegistry)) {
        return true;
      }
    }
    for (SandboxedSpawnActionContext strategy :
        dynamicStrategyRegistry.getDynamicSpawnActionContexts(
            spawn, DynamicStrategyRegistry.DynamicMode.REMOTE)) {
      if (strategy.canExec(spawn, actionContextRegistry)) {
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

  private static ImmutableList<SpawnResult> runLocally(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      @Nullable StopConcurrentSpawns stopConcurrentSpawns)
      throws ExecException, InterruptedException {
    DynamicStrategyRegistry dynamicStrategyRegistry =
        actionExecutionContext.getContext(DynamicStrategyRegistry.class);

    for (SandboxedSpawnActionContext strategy :
        dynamicStrategyRegistry.getDynamicSpawnActionContexts(
            spawn, DynamicStrategyRegistry.DynamicMode.LOCAL)) {
      return strategy.exec(spawn, actionExecutionContext, stopConcurrentSpawns);
    }
    throw new RuntimeException(
        "executorCreated not yet called or no default dynamic_local_strategy set");
  }

  private static ImmutableList<SpawnResult> runRemotely(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      @Nullable StopConcurrentSpawns stopConcurrentSpawns)
      throws ExecException, InterruptedException {
    DynamicStrategyRegistry dynamicStrategyRegistry =
        actionExecutionContext.getContext(DynamicStrategyRegistry.class);

    for (SandboxedSpawnActionContext strategy :
        dynamicStrategyRegistry.getDynamicSpawnActionContexts(
            spawn, DynamicStrategyRegistry.DynamicMode.REMOTE)) {
      return strategy.exec(spawn, actionExecutionContext, stopConcurrentSpawns);
    }
    throw new RuntimeException(
        "executorCreated not yet called or no default dynamic_remote_strategy set");
  }

  /**
   * Wraps the execution of a function that is supposed to execute a spawn via a strategy and only
   * updates the stdout/stderr files if this spawn succeeds.
   */
  private abstract static class Branch implements Callable<ImmutableList<SpawnResult>> {
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
    abstract ImmutableList<SpawnResult> callImpl(ActionExecutionContext context)
        throws InterruptedException, ExecException;

    /**
     * Executes the {@link #callImpl} hook and handles stdout/stderr.
     *
     * @return the spawn results if execution was successful
     * @throws InterruptedException if the branch was cancelled or an interrupt was caught
     * @throws ExecException if the spawn execution fails
     */
    @Override
    public final ImmutableList<SpawnResult> call() throws InterruptedException, ExecException {
      FileOutErr fileOutErr = getSuffixedFileOutErr(context.getFileOutErr(), "." + name);

      ImmutableList<SpawnResult> results = null;
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
