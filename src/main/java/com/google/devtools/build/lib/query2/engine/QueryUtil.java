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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskCallable;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

/** Several query utilities to make easier to work with query callbacks and uniquifiers. */
public final class QueryUtil {

  private QueryUtil() { }

  /** A {@link Callback} that can aggregate all the partial results into one set. */
  public interface AggregateAllCallback<T> extends Callback<T> {
    /** Returns a (mutable) set of all the results. */
    Set<T> getResult();
  }

  /** A {@link OutputFormatterCallback} that is also a {@link AggregateAllCallback}. */
  public abstract static class AggregateAllOutputFormatterCallback<T>
      extends ThreadSafeOutputFormatterCallback<T> implements AggregateAllCallback<T>  {
  }

  private static class AggregateAllOutputFormatterCallbackImpl<T>
      extends AggregateAllOutputFormatterCallback<T> {
    private final Set<T> result = Sets.newConcurrentHashSet();

    @Override
    public final void processOutput(Iterable<T> partialResult) {
      Iterables.addAll(result, partialResult);
    }

    @Override
    public Set<T> getResult() {
      return result;
    }
  }

  private static class OrderedAggregateAllOutputFormatterCallbackImpl<T>
      extends AggregateAllOutputFormatterCallback<T> {
    private final Set<T> result = CompactHashSet.create();

    @Override
    public final synchronized void processOutput(Iterable<T> partialResult) {
      Iterables.addAll(result, partialResult);
    }

    @Override
    public synchronized Set<T> getResult() {
      return result;
    }
  }

  /**
   * Returns a fresh {@link AggregateAllOutputFormatterCallback} instance whose
   * {@link AggregateAllCallback#getResult} returns all the elements of the result in the order they
   * were processed.
   */
  public static <T> AggregateAllOutputFormatterCallback<T>
      newOrderedAggregateAllOutputFormatterCallback() {
    return new OrderedAggregateAllOutputFormatterCallbackImpl<>();
  }

  /** Returns a fresh {@link AggregateAllCallback} instance. */
  public static <T> AggregateAllCallback<T> newAggregateAllCallback() {
    return new AggregateAllOutputFormatterCallbackImpl<>();
  }

  /**
   * Returns a {@link QueryTaskFuture} representing the evaluation of {@code expr} as a (mutable)
   * {@link Set} comprised of all the results.
   *
   * <p>Should only be used by QueryExpressions when it is the only way of achieving correctness.
   */
  public static <T> QueryTaskFuture<Set<T>> evalAll(
      QueryEnvironment<T> env, VariableContext<T> context, QueryExpression expr) {
    final AggregateAllCallback<T> callback = newAggregateAllCallback();
    return env.whenSucceedsCall(
        env.eval(expr, context, callback),
        new QueryTaskCallable<Set<T>>() {
          @Override
          public Set<T> call() {
            return callback.getResult();
          }
        });
  }

  /** A trivial {@link Uniquifier} implementation. */
  public static class UniquifierImpl<T, K> implements Uniquifier<T> {
    private final KeyExtractor<T, K> extractor;
    private final Set<K> alreadySeen;

    public UniquifierImpl(KeyExtractor<T, K> extractor) {
      this(extractor, /*concurrencyLevel=*/ 1);
    }

    public UniquifierImpl(KeyExtractor<T, K> extractor, int concurrencyLevel) {
      this.extractor = extractor;
      this.alreadySeen = Collections.newSetFromMap(
          new MapMaker().concurrencyLevel(concurrencyLevel).<K, Boolean>makeMap());
    }

    @Override
    public boolean unique(T element) {
      return alreadySeen.add(extractor.extractKey(element));
    }

    @Override
    public ImmutableList<T> unique(Iterable<T> newElements) {
      ImmutableList.Builder<T> result = ImmutableList.builder();
      for (T element : newElements) {
        if (unique(element)) {
          result.add(element);
        }
      }
      return result.build();
    }
  }

  /** A trivial {@link MinDepthUniquifier} implementation. */
  public static class MinDepthUniquifierImpl<T, K> implements MinDepthUniquifier<T> {
    private final KeyExtractor<T, K> extractor;
    private final ConcurrentMap<K, AtomicInteger> alreadySeenAtDepth;

    public MinDepthUniquifierImpl(KeyExtractor<T, K> extractor, int concurrencyLevel) {
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
