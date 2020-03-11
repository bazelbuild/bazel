// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;

/**
 * A representation of a (potentially) multi-step spawn execution, which can return multiple
 * results. This covers cases like remote/local fallback as well as tree artifact packing/unpacking,
 * which both require multiple attempts at running a spawn, and therefore can have multiple results.
 *
 * <p>This is intentionally similar to {@link ActionContinuationOrResult}, which will often wrap one
 * of these.
 *
 * <p>Any client of this class <b>should</b> first call {@link #isDone} before calling any of the
 * other methods. If {@link #isDone} returns true, then {@link #execute} must return this, and
 * {@link #get} must return the result. Use {@link #immediate} to construct such an instance. {@link
 * #getFuture} should return a completed future (see {@link Futures#immediateFuture}).
 *
 * <p>Otherwise, {@link #getFuture} must return a non-null value, {@link #execute} must not throw
 * {@link IllegalStateException}, and {@link #get} must throw {@link IllegalStateException}. Note
 * that calling {@link #execute} without waiting for the future to complete will likely block.
 *
 * <p>That is, it is always safe to call {@link #isDone}, {@link #getFuture}, and {@link #execute},
 * but it is only safe to call {@link #get} if the continuation is done.
 */
public abstract class SpawnContinuation {
  public static SpawnContinuation immediate(SpawnResult... spawnResults) {
    return new Finished(ImmutableList.copyOf(spawnResults));
  }

  public static SpawnContinuation immediate(ImmutableList<SpawnResult> spawnResults) {
    return new Finished(spawnResults);
  }

  public static SpawnContinuation failedWithExecException(ExecException e) {
    return new FailedWithExecException(e);
  }

  /**
   * Runs the state machine represented by the given continuation to completion, blocking as
   * necessary until all asynchronous computations finish, and the final continuation is done. Then
   * returns the list of spawn results (calling {@link SpawnContinuation#get}).
   *
   * <p>This method provides backwards compatibility for the cases where a method that's defined as
   * blocking obtains a continuation and needs the result before it can return. Over time, this
   * method should become less common as more actions are rewritten to support async execution.
   */
  public static ImmutableList<SpawnResult> completeBlocking(SpawnContinuation continuation)
      throws ExecException, InterruptedException {
    while (!continuation.isDone()) {
      continuation = continuation.execute();
    }
    return continuation.get();
  }

  public boolean isDone() {
    return false;
  }

  public abstract ListenableFuture<?> getFuture();

  public abstract SpawnContinuation execute() throws ExecException, InterruptedException;

  public ImmutableList<SpawnResult> get() {
    throw new IllegalStateException();
  }

  private static final class Finished extends SpawnContinuation {
    private final ImmutableList<SpawnResult> spawnResults;

    Finished(ImmutableList<SpawnResult> spawnResults) {
      this.spawnResults = spawnResults;
    }

    @Override
    public boolean isDone() {
      return true;
    }

    @Override
    public ListenableFuture<?> getFuture() {
      return Futures.immediateFuture(null);
    }

    @Override
    public SpawnContinuation execute() {
      return this;
    }

    @Override
    public ImmutableList<SpawnResult> get() {
      return spawnResults;
    }
  }

  private static final class FailedWithExecException extends SpawnContinuation {
    private final ExecException e;

    FailedWithExecException(ExecException e) {
      this.e = e;
    }

    @Override
    public ListenableFuture<?> getFuture() {
      // This call does not allocate memory because immediateFuture returns a singleton for null.
      return Futures.immediateFuture(null);
    }

    @Override
    public SpawnContinuation execute() throws ExecException {
      throw e;
    }
  }
}
