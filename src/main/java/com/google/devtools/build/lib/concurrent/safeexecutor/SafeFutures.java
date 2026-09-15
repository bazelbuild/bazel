// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent.safeexecutor;

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;

import com.google.common.base.Function;
import com.google.common.util.concurrent.AsyncCallable;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.concurrent.Callable;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

/**
 * Static utility providing rejection-safe wrappers around Guava {@link Futures} delegation methods.
 *
 * <p>{@link SafeExecutor} exists to hide underlying {@link Executor} references from clients that
 * could otherwise naively perform rejection-unsafe operations. These wrappers preserve this
 * encapsulation.
 *
 * <p>With the exception of {@link #submit} and {@link #submitAsync}, Guava's implementations are
 * already rejection-safe and these are transparent wrappers. The two exceptions interact
 * synchronously with the underlying executor, so while not transparent, the implementations remain
 * trivial by immediately catching and forwarding {@link RejectedExecutionException}s.
 */
public final class SafeFutures {

  private SafeFutures() {}

  /**
   * Submits a Callable for execution using {@code safeExecutor}, returning a ListenableFuture.
   *
   * <p>Delegates to {@link Futures#submit(Callable, Executor)}. If submission fails (e.g. {@link
   * RejectedExecutionException}), returns an immediately failed ListenableFuture containing the
   * exception instead of throwing synchronously to the submitter.
   */
  public static <T> ListenableFuture<T> submit(Callable<T> callable, SafeExecutor safeExecutor) {
    try {
      return Futures.submit(callable, safeExecutor.getInternalUnsafeExecutor());
    } catch (RejectedExecutionException e) {
      return immediateFailedFuture(e);
    }
  }

  /**
   * Submits an AsyncCallable for execution using {@code safeExecutor}, returning a
   * ListenableFuture.
   *
   * <p>Delegates to {@link Futures#submitAsync(AsyncCallable, Executor)}. If submission fails (e.g.
   * {@link RejectedExecutionException}), returns an immediately failed ListenableFuture containing
   * the exception instead of throwing synchronously to the submitter.
   */
  public static <T> ListenableFuture<T> submitAsync(
      AsyncCallable<T> callable, SafeExecutor safeExecutor) {
    try {
      return Futures.submitAsync(callable, safeExecutor.getInternalUnsafeExecutor());
    } catch (RejectedExecutionException e) {
      return immediateFailedFuture(e);
    }
  }

  /** Wraps {@link Futures#transform(ListenableFuture, Function, Executor)}. */
  public static <I, O> ListenableFuture<O> transform(
      ListenableFuture<I> input,
      Function<? super I, ? extends O> function,
      SafeExecutor safeExecutor) {
    return Futures.transform(input, function, safeExecutor.getInternalUnsafeExecutor());
  }

  /** Wraps {@link Futures#transformAsync(ListenableFuture, AsyncFunction, Executor)}. */
  public static <I, O> ListenableFuture<O> transformAsync(
      ListenableFuture<I> input,
      AsyncFunction<? super I, ? extends O> function,
      SafeExecutor safeExecutor) {
    return Futures.transformAsync(input, function, safeExecutor.getInternalUnsafeExecutor());
  }

  /** Wraps {@link Futures.FutureCombiner#call(Callable, Executor)}. */
  public static <C> ListenableFuture<C> call(
      Futures.FutureCombiner<?> combiner, Callable<C> callable, SafeExecutor safeExecutor) {
    return combiner.call(callable, safeExecutor.getInternalUnsafeExecutor());
  }

  /** Wraps {@link Futures.FutureCombiner#callAsync(AsyncCallable, Executor)}. */
  public static <C> ListenableFuture<C> callAsync(
      Futures.FutureCombiner<?> combiner, AsyncCallable<C> callable, SafeExecutor safeExecutor) {
    return combiner.callAsync(callable, safeExecutor.getInternalUnsafeExecutor());
  }
}
