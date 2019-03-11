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
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;

/**
 * A representation of a (potentially) multi-step spawn execution, which can return multiple
 * results. This covers cases like remote/local fallback as well as tree artifact packing/unpacking,
 * which both require multiple attempts at running a spawn, and therefore can have multiple results.
 *
 * <p>This is intentionally similar to {@link ActionContinuationOrResult}, which will often wrap one
 * of these.
 *
 * <p>Any client of this class <b>must</b> first call {@link #isDone} before calling any of the
 * other methods. If {@link #isDone} returns true, then {@link #getFuture} and {@link #execute} must
 * throw {@link IllegalStateException}, but {@link #get} must return a valid value. Use {@link
 * #immediate} to construct such an instance.
 *
 * <p>Otherwise, {@link #getFuture} must return a non-null value, and {@link #execute} must not
 * throw {@link IllegalStateException}, whereas {@link #get} must throw {@link
 * IllegalStateException}.
 */
public abstract class SpawnContinuation {
  public static SpawnContinuation immediate(SpawnResult... spawnResults) {
    return new Finished(ImmutableList.copyOf(spawnResults));
  }

  public static SpawnContinuation immediate(List<SpawnResult> spawnResults) {
    return new Finished(ImmutableList.copyOf(spawnResults));
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
  public static List<SpawnResult> completeBlocking(SpawnContinuation continuation)
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

  public List<SpawnResult> get() {
    throw new IllegalStateException();
  }

  private static final class Finished extends SpawnContinuation {
    private final List<SpawnResult> spawnResults;

    Finished(List<SpawnResult> spawnResults) {
      this.spawnResults = spawnResults;
    }

    public boolean isDone() {
      return true;
    }

    @Override
    public ListenableFuture<?> getFuture() {
      throw new IllegalStateException();
    }

    @Override
    public SpawnContinuation execute() {
      throw new IllegalStateException();
    }

    public List<SpawnResult> get() {
      return spawnResults;
    }
  }
}
