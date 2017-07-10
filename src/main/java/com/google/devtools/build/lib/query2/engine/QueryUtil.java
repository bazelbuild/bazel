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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.MutableMap;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskCallable;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/** Several query utilities to make easier to work with query callbacks and uniquifiers. */
public final class QueryUtil {

  private QueryUtil() { }

  /** A {@link Callback} that can aggregate all the partial results into one set. */
  public interface AggregateAllCallback<T, S extends Set<T>> extends Callback<T> {
    /** Returns a {@link Set} of all the results. */
    S getResult();
  }

  /** A {@link OutputFormatterCallback} that is also a {@link AggregateAllCallback}. */
  public abstract static class AggregateAllOutputFormatterCallback<T, S extends Set<T>>
      extends ThreadSafeOutputFormatterCallback<T> implements AggregateAllCallback<T, S>  {
  }

  private static class AggregateAllOutputFormatterCallbackImpl<T>
      extends AggregateAllOutputFormatterCallback<T, ThreadSafeMutableSet<T>> {
    private final ThreadSafeMutableSet<T> result;

    private AggregateAllOutputFormatterCallbackImpl(QueryEnvironment<T> env) {
      this.result = env.createThreadSafeMutableSet();
    }

    @Override
    public final void processOutput(Iterable<T> partialResult) {
      Iterables.addAll(result, partialResult);
    }

    @Override
    public ThreadSafeMutableSet<T> getResult() {
      return result;
    }
  }

  private static class OrderedAggregateAllOutputFormatterCallbackImpl<T>
      extends AggregateAllOutputFormatterCallback<T, Set<T>> {
    private final Set<T> resultSet;
    private final List<T> resultList;

    private OrderedAggregateAllOutputFormatterCallbackImpl(QueryEnvironment<T> env) {
      this.resultSet = env.createThreadSafeMutableSet();
      this.resultList = new ArrayList<>();
    }

    @Override
    public final synchronized void processOutput(Iterable<T> partialResult) {
      for (T element : partialResult) {
        if (resultSet.add(element)) {
          resultList.add(element);
        }
      }
    }

    @Override
    public synchronized Set<T> getResult() {
      // A CompactHashSet's iteration order is the same as its insertion order.
      CompactHashSet<T> result = CompactHashSet.createWithExpectedSize(resultList.size());
      result.addAll(resultList);
      return result;
    }
  }

  /**
   * Returns a fresh {@link AggregateAllOutputFormatterCallback} instance whose
   * {@link AggregateAllCallback#getResult} returns all the elements of the result in the order they
   * were processed.
   */
  public static <T> AggregateAllOutputFormatterCallback<T, Set<T>>
      newOrderedAggregateAllOutputFormatterCallback(QueryEnvironment<T> env) {
    return new OrderedAggregateAllOutputFormatterCallbackImpl<>(env);
  }

  /** Returns a fresh {@link AggregateAllCallback} instance. */
  public static <T> AggregateAllCallback<T, ThreadSafeMutableSet<T>> newAggregateAllCallback(
      QueryEnvironment<T> env) {
    return new AggregateAllOutputFormatterCallbackImpl<>(env);
  }

  /**
   * Returns a {@link QueryTaskFuture} representing the evaluation of {@code expr} as a mutable,
   * thread safe {@link Set} comprised of all the results.
   *
   * <p>Should only be used by QueryExpressions when it is the only way of achieving correctness.
   */
  public static <T> QueryTaskFuture<ThreadSafeMutableSet<T>> evalAll(
      QueryEnvironment<T> env, VariableContext<T> context, QueryExpression expr) {
    final AggregateAllCallback<T, ThreadSafeMutableSet<T>> callback = newAggregateAllCallback(env);
    return env.whenSucceedsCall(
        env.eval(expr, context, callback),
        new QueryTaskCallable<ThreadSafeMutableSet<T>>() {
          @Override
          public ThreadSafeMutableSet<T> call() {
            return callback.getResult();
          }
        });
  }

  /**
   * A mutable thread safe {@link Set} that uses a {@link KeyExtractor} for determining equality of
   * its elements. This is useful e.g. when {@code T} isn't guaranteed to have a useful
   * {@link Object#equals} and {@link Object#hashCode} but {@code K} is.
   */
  public static class ThreadSafeMutableKeyExtractorBackedSetImpl<T, K>
      extends AbstractSet<T> implements ThreadSafeMutableSet<T> {
    private final KeyExtractor<T, K> extractor;
    private final Class<T> elementClass;
    private final ConcurrentMap<K, T> map;

    public ThreadSafeMutableKeyExtractorBackedSetImpl(
        KeyExtractor<T, K> extractor, Class<T> elementClass) {
      this(extractor, elementClass, /*concurrencyLevel=*/ 1);
    }

    public ThreadSafeMutableKeyExtractorBackedSetImpl(
        KeyExtractor<T, K> extractor,
        Class<T> elementClass,
        int concurrencyLevel) {
      this.extractor = extractor;
      this.elementClass = elementClass;
      this.map = new MapMaker().concurrencyLevel(concurrencyLevel).makeMap();
    }

    @Override
    public Iterator<T> iterator() {
      return map.values().iterator();
    }

    @Override
    public int size() {
      return map.size();
    }

    @Override
    public boolean add(T element) {
      return map.putIfAbsent(extractor.extractKey(element), element) == null;
    }

    @Override
    public boolean contains(Object obj) {
      if (!elementClass.isInstance(obj)) {
        return false;
      }
      T element = elementClass.cast(obj);
      return map.containsKey(extractor.extractKey(element));
    }

    @Override
    public boolean remove(Object obj) {
      if (!elementClass.isInstance(obj)) {
        return false;
      }
      T element = elementClass.cast(obj);
      return map.remove(extractor.extractKey(element)) != null;
    }
  }

  /**
   * A {@link MutableMap} implementation that uses a {@link KeyExtractor} for determining equality
   * of its keys.
   */
  public static class MutableKeyExtractorBackedMapImpl<T, K, V> implements MutableMap<T, V> {
    private final KeyExtractor<T, K> extractor;
    private final HashMap<K, V> map;

    public MutableKeyExtractorBackedMapImpl(KeyExtractor<T, K> extractor) {
      this.extractor = extractor;
      this.map = new HashMap<>();
    }

    @Override
    @Nullable
    public V get(T key) {
      return map.get(extractor.extractKey(key));
    }

    @Override
    public V put(T key, V value) {
      return map.put(extractor.extractKey(key), value);
    }
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
