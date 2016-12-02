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
package com.google.devtools.build.lib.query2.engine;


import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.CompactHashSet;
import java.util.Collections;
import java.util.Set;
import javax.annotation.concurrent.ThreadSafe;

/** Several query utilities to make easier to work with query callbacks and uniquifiers. */
public final class QueryUtil {

  private QueryUtil() { }

  /** A {@link Callback} that can aggregate all the partial results into one set. */
  public interface AggregateAllCallback<T> extends Callback<T> {
    Set<T> getResult();
  }

  /** A {@link AggregateAllCallback} that is a {@link ThreadSafeCallback}. */
  public interface ThreadSafeAggregateAllCallback<T>
      extends AggregateAllCallback<T>, ThreadSafeCallback<T> {
  }

  /** A {@link OutputFormatterCallback} that can aggregate all the partial results into one set. */
  public abstract static class AggregateAllOutputFormatterCallback<T>
      extends OutputFormatterCallback<T> implements AggregateAllCallback<T>  {
  }

  private static class AggregateAllOutputFormatterCallbackImpl<T>
      extends AggregateAllOutputFormatterCallback<T> {
    // We only add to the CompactHashSet, so iteration order is the same as insertion order.
    private final Set<T> result = CompactHashSet.create();

    @Override
    public final void processOutput(Iterable<T> partialResult) {
      Iterables.addAll(result, partialResult);
    }

    @Override
    public Set<T> getResult() {
      return result;
    }
  }

  private static class ThreadSafeAggregateAllOutputFormatterCallbackImpl<T>
      implements ThreadSafeAggregateAllCallback<T> {
    private final Set<T> result = Sets.newConcurrentHashSet();

    @Override
    public final void process(Iterable<T> partialResult) {
      Iterables.addAll(result, partialResult);
    }

    @Override
    public Set<T> getResult() {
      return result;
    }
  }

  /**
   * Returns a fresh {@link AggregateAllOutputFormatterCallback} that can aggregate all the partial
   * results into one set.
   *
   * <p>Intended to be used by top-level evaluation of {@link QueryExpression}s; contrast with
   * {@link #newThreadSafeAggregateAllCallback}.
   */
  public static <T> AggregateAllOutputFormatterCallback<T>
      newAggregateAllOutputFormatterCallback() {
    return new AggregateAllOutputFormatterCallbackImpl<>();
  }

  /**
   * Returns a fresh {@link AggregateAllCallback}.
   *
   * <p>Intended to be used by {@link QueryEnvironment#eval} implementations that make use of
   * neither of streaming evaluation nor parallel evaluation.
   */
  public static <T> AggregateAllCallback<T> newOrderedAggregateAllCallback() {
    return new AggregateAllOutputFormatterCallbackImpl<>();
  }

  /**
   * Returns a fresh {@link ThreadSafeAggregateAllCallback}.
   *
   * <p>Intended to be used by {@link QueryExpression} implementations; contrast with
   * {@link #newAggregateAllOutputFormatterCallback}.
   */
  public static <T> AggregateAllCallback<T> newThreadSafeAggregateAllCallback() {
    return new ThreadSafeAggregateAllOutputFormatterCallbackImpl<>();
  }

  /**
   * Fully evaluate a {@code QueryExpression} and return a set with all the results.
   *
   * <p>Should only be used by QueryExpressions when it is the only way of achieving correctness.
   */
  public static <T> Set<T> evalAll(
      QueryEnvironment<T> env, VariableContext<T> context, QueryExpression expr)
          throws QueryException, InterruptedException {
    AggregateAllCallback<T> callback = newThreadSafeAggregateAllCallback();
    env.eval(expr, context, callback);
    return callback.getResult();
  }

  /** A trivial {@link Uniquifier} base class. */
  public abstract static class AbstractUniquifier<T, K>
      extends AbstractUniquifierBase<T, K> {
    private final CompactHashSet<K> alreadySeen = CompactHashSet.create();

    @Override
    public final boolean unique(T element) {
      return alreadySeen.add(extractKey(element));
    }

    /**
     * Extracts an unique key that can be used to dedupe the given {@code element}.
     *
     * <p>Depending on the choice of {@code K}, this enables potential memory optimizations.
     */
    protected abstract K extractKey(T element);
  }

  /** A trivial {@link ThreadSafeUniquifier} base class. */
  public abstract static class AbstractThreadSafeUniquifier<T, K>
      extends AbstractUniquifierBase<T, K> implements ThreadSafeUniquifier<T> {
    private final Set<K> alreadySeen;

    protected AbstractThreadSafeUniquifier(int concurrencyLevel) {
      this.alreadySeen = Collections.newSetFromMap(
          new MapMaker().concurrencyLevel(concurrencyLevel).<K, Boolean>makeMap());
    }

    @Override
    public final boolean unique(T element) {
      return alreadySeen.add(extractKey(element));
    }

    /**
     * Extracts an unique key that can be used to dedupe the given {@code element}.
     *
     * <p>Depending on the choice of {@code K}, this enables potential memory optimizations.
     */
    protected abstract K extractKey(T element);
  }

  private abstract static class AbstractUniquifierBase<T, K> implements Uniquifier<T> {
    @Override
    public final ImmutableList<T> unique(Iterable<T> newElements) {
      ImmutableList.Builder<T> result = ImmutableList.builder();
      for (T element : newElements) {
        if (unique(element)) {
          result.add(element);
        }
      }
      return result.build();
    }
  }

  /**
   * An abstraction for sending results to a {@link Callback} based on partial query results.
   *
   * <p>Intended as a convenience for {@link QueryFunction#eval}/{@link QueryFunction#parEval}
   * implementations.
   */
  @ThreadSafe
  interface Processor<T> {
    /**
     * Processes the given {@code partialResult} and then invokes the given {@code callback} with
     * a further result.
     */
    void process(Iterable<T> partialResult, Callback<T> callback)
        throws QueryException, InterruptedException;
  }

  /**
   * Similar to {@link Processor}, except the {@link #process} function also gets access to a
   * {@link Uniquifier}.
   */
  @ThreadSafe
  interface ProcessorWithUniquifier<T> {
    /**
     * Processes the given {@code partialResult} using the given {@code uniquifier} and then invokes
     * the given {@code callback} with a further result.
     */
    void process(Iterable<T> partialResult, Uniquifier<T> uniquifier, Callback<T> callback)
        throws QueryException, InterruptedException;
  }

  /**
   * Returns a new {@link Callback} that composes {@code callback} with {@code processor}. That is,
   * this method returns a {@code c} such that {@code c.process(r)} calls
   * {@code callback.process(r, callback)}.
   *
   * <p>The returned {@link Callback} will be a {@link ThreadSafeCallback} if {@code callback}
   * is as well.
   */
  public static <T> Callback<T> compose(
      final Processor<T> processor,
      final Callback<T> callback) {
    return callback instanceof ThreadSafeCallback
        ? new ThreadSafeCallback<T>() {
            @Override
            public void process(Iterable<T> partialResult)
                throws QueryException, InterruptedException {
              processor.process(partialResult, callback);
            }
          }
        : new Callback<T>() {
            @Override
            public void process(Iterable<T> partialResult)
                throws QueryException, InterruptedException {
              processor.process(partialResult, callback);
            }
          };
  }

  /**
   * Returns a new {@link Callback} that composes {@code callback} with {@code processor} (applied
   * to {@code uniquifier}). That is, this method returns a {@code c} such that {@code c.process(r)}
   * calls {@code callback.process(r, uniquifier, c)}.
   *
   * <p>The returned {@link Callback} will be a {@link ThreadSafeCallback} if {@code callback} is a
   * {@link ThreadSafeCallback} and {@code uniquifier} is a {@link ThreadSafeUniquifier}.
   */
  public static <T> Callback<T> compose(
      final ProcessorWithUniquifier<T> processor,
      final Uniquifier<T> uniquifier,
      final Callback<T> callback) {
    return (callback instanceof ThreadSafeCallback) && (uniquifier instanceof ThreadSafeUniquifier)
        ? new ThreadSafeCallback<T>() {
            @Override
            public void process(Iterable<T> partialResult)
                throws QueryException, InterruptedException {
              processor.process(partialResult, uniquifier, callback);
            }
          }
        : new Callback<T>() {
            @Override
            public void process(Iterable<T> partialResult)
                throws QueryException, InterruptedException {
              processor.process(partialResult, uniquifier, callback);
            }
          };
  }
}
