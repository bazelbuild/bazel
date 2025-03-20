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

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry.DynamicMode;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Wraps the execution of a function that is supposed to execute a spawn via a strategy and only
 * updates the stdout/stderr files if this spawn succeeds.
 */
abstract class Branch implements Callable<ImmutableList<SpawnResult>> {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * True if this branch is still starting up, i.e. didn't get to the inner part of {@link
   * #callImpl(ActionExecutionContext)} yet.
   */
  protected final AtomicBoolean starting = new AtomicBoolean(true);
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
  /**
   * Semaphore that indicates whether this branch is done, i.e. either completed or cancelled. This
   * is needed to wait for the branch to finish its own cleanup (e.g. terminating subprocesses) once
   * it has been cancelled.
   */
  protected final Semaphore done = new Semaphore(0);

  protected final DynamicExecutionOptions options;
  protected final ActionExecutionContext context;

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

  /** Returns the {@code Semaphore} indicating whether this branch is done. */
  Semaphore getDoneSemaphore() {
    return done;
  }

  /** Returns whether this branch has already been cancelled. */
  boolean isCancelled() {
    return future.isCancelled();
  }

  /** Cancels this branch. Equivalent to {@code Future.cancel(true)}. */
  boolean cancel() {
    return future.cancel(true);
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
   * Executes the {@link #callImpl} hook and handles stdout/stderr.
   *
   * @return the spawn results if execution was successful
   * @throws InterruptedException if the branch was cancelled or an interrupt was caught
   * @throws ExecException if the spawn execution fails
   */
  @Override
  public final ImmutableList<SpawnResult> call() throws InterruptedException, ExecException {
    FileOutErr fileOutErr = getSuffixedFileOutErr(context.getFileOutErr(), "." + getMode().name());

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
