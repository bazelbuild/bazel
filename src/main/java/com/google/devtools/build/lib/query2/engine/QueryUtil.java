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
import com.google.devtools.build.lib.collect.CompactHashSet;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

/** Several query utilities to make easier to work with query callbacks and uniquifiers. */
public final class QueryUtil {

  private QueryUtil() { }

  /** A {@link Callback} that can aggregate all the partial results into one set. */
  public interface AggregateAllCallback<T> extends Callback<T> {
    Set<T> getResult();
  }

  /** A {@link OutputFormatterCallback} that can aggregate all the partial results into one set. */
  public abstract static class AggregateAllOutputFormatterCallback<T>
      extends OutputFormatterCallback<T> implements AggregateAllCallback<T>  {
  }

  private static class AggregateAllOutputFormatterCallbackImpl<T>
      extends AggregateAllOutputFormatterCallback<T> {
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

  /**
   * Returns a fresh {@link AggregateAllOutputFormatterCallback} that can aggregate all the partial
   * results into one set.
   *
   * <p>Intended to be used by top-level evaluation of {@link QueryExpression}s; contrast with
   * {@link #newAggregateAllCallback}.
   */
  public static <T> AggregateAllOutputFormatterCallback<T>
      newAggregateAllOutputFormatterCallback() {
    return new AggregateAllOutputFormatterCallbackImpl<>();
  }

  /**
   * Returns a fresh {@link AggregateAllCallback}.
   *
   * <p>Intended to be used by {@link QueryExpression} implementations; contrast with
   * {@link #newAggregateAllOutputFormatterCallback}.
   */
  public static <T> AggregateAllCallback<T> newAggregateAllCallback() {
    return new AggregateAllOutputFormatterCallbackImpl<>();
  }

  /**
   * Fully evaluate a {@code QueryExpression} and return a set with all the results.
   *
   * <p>Should only be used by QueryExpressions when it is the only way of achieving correctness.
   */
  public static <T> Set<T> evalAll(
      QueryEnvironment<T> env, VariableContext<T> context, QueryExpression expr)
          throws QueryException, InterruptedException {
    AggregateAllCallback<T> callback = newAggregateAllCallback();
    env.eval(expr, context, callback);
    return callback.getResult();
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

  /** A trivial {@link Uniquifier} implementation. */
  public static class UniquifierImpl<T, K> extends AbstractUniquifierBase<T, K> {
    private final KeyExtractor<T, K> extractor;
    private final CompactHashSet<K> alreadySeen = CompactHashSet.create();

    public UniquifierImpl(KeyExtractor<T, K> extractor) {
      this.extractor = extractor;
    }

    @Override
    public final boolean unique(T element) {
      return alreadySeen.add(extractor.extractKey(element));
    }
  }

  /** A trvial {@link ThreadSafeUniquifier} implementation. */
  public static class ThreadSafeUniquifierImpl<T, K>
      extends AbstractUniquifierBase<T, K> implements ThreadSafeUniquifier<T> {
    private final KeyExtractor<T, K> extractor;
    private final Set<K> alreadySeen;

    public ThreadSafeUniquifierImpl(KeyExtractor<T, K> extractor, int concurrencyLevel) {
      this.extractor = extractor;
      this.alreadySeen = Collections.newSetFromMap(
          new MapMaker().concurrencyLevel(concurrencyLevel).<K, Boolean>makeMap());
    }

    @Override
    public final boolean unique(T element) {
      return alreadySeen.add(extractor.extractKey(element));
    }
  }

  /** A trivial {@link ThreadSafeMinDepthUniquifier} implementation. */
  public static class ThreadSafeMinDepthUniquifierImpl<T, K>
      implements ThreadSafeMinDepthUniquifier<T> {
    private final KeyExtractor<T, K> extractor;
    private final ConcurrentMap<K, AtomicInteger> alreadySeenAtDepth;

    public ThreadSafeMinDepthUniquifierImpl(
        KeyExtractor<T, K> extractor, int concurrencyLevel) {
      this.extractor = extractor;
      this.alreadySeenAtDepth = new MapMaker().concurrencyLevel(concurrencyLevel).makeMap();
    }

    @Override
    public final ImmutableList<T> uniqueAtDepthLessThanOrEqualTo(
        Iterable<T> newElements, int depth) {
      ImmutableList.Builder<T> result = ImmutableList.builder();
      for (T element : newElements) {
        AtomicInteger newDepth = new AtomicInteger(depth);
        AtomicInteger previousDepth =
            alreadySeenAtDepth.putIfAbsent(extractor.extractKey(element), newDepth);
        if (previousDepth != null) {
          if (depth < previousDepth.get()) {
            synchronized (previousDepth) {
              if (depth < previousDepth.get()) {
                // We've seen the element before, but never at a depth this shallow.
                previousDepth.set(depth);
                result.add(element);
              }
            }
          }
        } else {
          // We've never seen the element before.
          result.add(element);
        }
      }
      return result.build();
    }
  }
}
