// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry.DynamicMode;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.concurrent.CancellableTask;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * Wraps the execution of a function that is supposed to execute a spawn via a strategy and only
 * updates the stdout/stderr files if this spawn succeeds.
 */
abstract class Branch implements Callable<ImmutableList<SpawnResult>> {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * The execution of this branch. Cancelling it before it starts prevents it from ever running,
   * and after cancellation the other branch can wait until it no longer executes and thus has
   * finished its own cleanup (e.g. terminating subprocesses).
   */
  private final CancellableTask<InterruptedException> task = new CancellableTask<>(this::runBranch);

  /** The results of {@link #runBranch}, which are only accessed by the thread that ran it. */
  @Nullable private ImmutableList<SpawnResult> results;

  @Nullable private ExecException execException;

  /** The {@link Spawn} this branch is running. */
  protected final Spawn spawn;

  /**
   * The {@link SettableFuture} with the results from running the spawn. Must not be null if
   * execution succeeded.
   */
  protected final SettableFuture<ImmutableList<SpawnResult>> future = SettableFuture.create();

  /**
   * The strategy (local or remote) that cancelled the other one. Null until one has been cancelled.
   * This object is shared between the local and remote branch of an action.
   */
  protected final AtomicReference<DynamicMode> strategyThatCancelled;

  protected final DynamicExecutionOptions options;
  protected final ActionExecutionContext context;

  protected Branch otherBranch;

  /**
   * Creates a new branch of dynamic execution.
   *
   * @param context the action execution context given to the dynamic strategy, used to obtain the
   *     final location of the stdout/stderr
   */
  Branch(
      ActionExecutionContext context,
      Spawn spawn,
      AtomicReference<DynamicMode> strategyThatCancelled,
      DynamicExecutionOptions options) {
    this.context = context;
    this.spawn = spawn;
    this.strategyThatCancelled = strategyThatCancelled;
    this.options = options;
  }

  boolean isDone() {
    return future.isDone();
  }

  /** Returns whether this branch has already been cancelled. */
  boolean isCancelled() {
    return future.isCancelled();
  }

  /**
   * Cancels this branch, preventing it from starting or interrupting it if it is already running.
   *
   * <p>Does not wait for a running branch to stop executing, use {@link #awaitStopped} for that.
   *
   * @return whether this call cancelled the branch's future
   */
  boolean cancel() {
    // Cancel the future first so that when the interrupt arrives in the branch, its cancellation
    // is already observable through isCancelled() (e.g. by the StopConcurrentSpawns callbacks).
    boolean cancelled = future.cancel(false);
    task.cancel();
    return cancelled;
  }

  /**
   * Waits until this branch no longer executes, cancelling it if it has not started.
   *
   * <p>Once this method returns, the branch has finished its own cleanup (e.g. terminating
   * subprocesses) and no longer accesses the spawn's outputs.
   */
  void awaitStopped() throws InterruptedException {
    task.cancelAndAwait();
  }

  /**
   * Marks this branch as cancelled without interrupting it.
   *
   * <p>Unlike {@link #cancel}, this may be called by the branch itself, which cannot cancel its
   * own task.
   */
  protected void cancelSelf() {
    var unused = future.cancel(false);
  }

  /** Gets the results from this branch, when available. Behaves like {@link Future#get()} */
  ImmutableList<SpawnResult> getResults() throws ExecutionException, InterruptedException {
    return future.get();
  }

  public Spawn getSpawn() {
    return spawn;
  }

  public abstract DynamicMode getMode();

  /** Returns a human-readable description of what we can tell about the state of this Future. */
  String branchState() {
    return (isCancelled() ? "cancelled" : "not cancelled")
        + " and "
        + (isDone() ? "done" : "not done");
  }

  /** Executes this branch using the provided executor. */
  public void execute(ListeningExecutorService executor) {
    future.setFuture(executor.submit(this));
  }

  /** Sets up the {@link Future} used in the current branch to know what other branch to cancel. */
  protected void prepareFuture(Branch otherBranch) {
    this.otherBranch = otherBranch;
    future.addListener(
        () -> {
          // If the current branch succeeds, there is no need to keep the other branch running.
          // If the current branch fails, cancel the other branch as well. However, that one may
          // in turn cancel us, thus causing an interruption. Don't consider that a failure as
          // we otherwise risk canceling both branches.
          var state = future.state();
          if (state == Future.State.SUCCESS
              || (state == Future.State.FAILED
                  && !(future.exceptionNow() instanceof InterruptedException))) {
            otherBranch.cancel();
          }
          if (options.getDebugSpawnScheduler()) {
            logger.atInfo().log(
                "In listener callback, the future of the remote branch is %s",
                future.state().name());
            try {
              future.get();
            } catch (InterruptedException | ExecutionException e) {
              logger.atInfo().withCause(e).log(
                  "The future of the remote branch failed with an exception.");
            }
          }
        },
        directExecutor());
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
        FileSystemUtils.moveFile(from.getOutputPath(), to.getOutputPath());
      }
      if (from.getErrorPath().exists()) {
        FileSystemUtils.moveFile(from.getErrorPath(), to.getErrorPath());
      }
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Could not move action logs from execution");
    }
  }

  private static FileOutErr getSuffixedFileOutErr(FileOutErr fileOutErr, String suffix) {
    Path outDir = checkNotNull(fileOutErr.getOutputPath().getParentDirectory());
    String outBaseName = fileOutErr.getOutputPath().getBaseName();
    Path errDir = checkNotNull(fileOutErr.getErrorPath().getParentDirectory());
    String errBaseName = fileOutErr.getErrorPath().getBaseName();
    return new FileOutErr(
        outDir.getChild(outBaseName + suffix), errDir.getChild(errBaseName + suffix));
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
   * Executes the {@link #callImpl} hook and handles stdout/stderr, unless the branch was cancelled
   * before it started running.
   *
   * @return the spawn results if execution was successful
   * @throws InterruptedException if the branch was cancelled or an interrupt was caught
   * @throws ExecException if the spawn execution fails
   */
  @Override
  public final ImmutableList<SpawnResult> call() throws InterruptedException, ExecException {
    if (!task.runIfNotCancelled()) {
      throw new InterruptedException(getMode() + " branch was cancelled before it started");
    }
    if (execException != null) {
      throw execException;
    }
    return checkNotNull(results);
  }

  private void runBranch() throws InterruptedException {
    FileOutErr fileOutErr = getSuffixedFileOutErr(context.getFileOutErr(), "." + getMode().name());

    try {
      results = callImpl(context.withFileOutErr(fileOutErr));
    } catch (ExecException e) {
      execException = e;
    } finally {
      try {
        fileOutErr.close();
      } catch (IOException ignored) {
        // Nothing we can do here.
      }
    }

    moveFileOutErr(fileOutErr, context.getFileOutErr());
  }
}
