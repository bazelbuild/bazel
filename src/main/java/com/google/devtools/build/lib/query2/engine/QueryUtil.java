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


import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.MapMaker;
import com.google.devtools.build.lib.collect.CompactHashSet;
import java.util.Collections;
import java.util.Set;

/** Several query utilities to make easier to work with query callbacks and uniquifiers. */
public final class QueryUtil {

  private QueryUtil() { }

  /** A callback that can aggregate all the partial results in one set */
  public static class AggregateAllCallback<T> implements Callback<T> {

    private final CompactHashSet<T> result = CompactHashSet.create();

    @Override
    public void process(Iterable<T> partialResult) throws QueryException, InterruptedException {
      Iterables.addAll(result, partialResult);
    }

    public Set<T> getResult() {
      return result;
    }

    @Override
    public String toString() {
      return "Aggregate all: " + result;
    }
  }

  /**
   * Fully evaluate a {@code QueryExpression} and return a set with all the results.
   *
   * <p>Should ony be used by QueryExpressions when it is the only way of achieving correctness.
   */
  public static <T> Set<T> evalAll(
      QueryEnvironment<T> env, VariableContext<T> context, QueryExpression expr)
          throws QueryException, InterruptedException {
    AggregateAllCallback<T> callback = new AggregateAllCallback<>();
    env.eval(expr, context, callback);
    return callback.result;
  }

  /**
   * Notify {@code parentCallback} only about the events that match {@code retainIfTrue} predicate.
   *
   * @param parentCallback The parent callback to notify with the matching elements
   * @param retainIfTrue A predicate that defines what elements to notify to the parent callback.
   */
  public static <T> Callback<T> filteredCallback(final Callback<T> parentCallback,
      final Predicate<T> retainIfTrue) {
    return new Callback<T>() {
      @Override
      public void process(Iterable<T> partialResult) throws QueryException, InterruptedException {
        Iterable<T> filter = Iterables.filter(partialResult, retainIfTrue);
        if (!Iterables.isEmpty(filter)) {
          parentCallback.process(filter);
        }
      }

      @Override
      public String toString() {
        return "filtered parentCallback of : " + retainIfTrue;
      }
    };
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
}
