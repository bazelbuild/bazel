// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.base.Function;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * A partial implementation of {@link QueryEnvironment} that has trivial in-thread implementations
 * of all the {@link QueryTaskFuture}/{@link QueryTaskCallable} helper methods.
 */
public abstract class AbstractQueryEnvironment<T> implements QueryEnvironment<T> {
  /** Concrete implementation of {@link QueryTaskFuture}. */
  protected static final class QueryTaskFutureImpl<T>
      extends QueryTaskFutureImplBase<T> implements ListenableFuture<T> {
    private final ListenableFuture<T> delegate;

    private QueryTaskFutureImpl(ListenableFuture<T> delegate) {
      this.delegate = delegate;
    }

    public static <R> QueryTaskFutureImpl<R> ofDelegate(ListenableFuture<R> delegate) {
      return (delegate instanceof QueryTaskFutureImpl)
          ? (QueryTaskFutureImpl<R>) delegate
          : new QueryTaskFutureImpl<>(delegate);
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      return delegate.cancel(mayInterruptIfRunning);
    }

    @Override
    public boolean isCancelled() {
      return delegate.isCancelled();
    }

    @Override
    public boolean isDone() {
      return delegate.isDone();
    }

    @Override
    public T get() throws InterruptedException, ExecutionException {
      return delegate.get();
    }

    @Override
    public T get(long timeout, TimeUnit unit)
        throws InterruptedException, ExecutionException, TimeoutException {
      return delegate.get(timeout, unit);
    }

    @Override
    public void addListener(Runnable listener, Executor executor) {
      delegate.addListener(listener, executor);
    }

    @Override
    public T getIfSuccessful() {
      try {
        return Futures.getDone(delegate);
      } catch (CancellationException | ExecutionException e) {
        throw new IllegalStateException(e);
      }
    }

    public T getChecked() throws InterruptedException, QueryException {
      try {
        return get();
      } catch (CancellationException e) {
        throw new InterruptedException();
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        Throwables.propagateIfPossible(cause, QueryException.class);
        Throwables.propagateIfPossible(cause, InterruptedException.class);
        throw new IllegalStateException(e.getCause());
      }
    }
  }

  @Override
  public <R> QueryTaskFuture<R> immediateSuccessfulFuture(R value) {
    return new QueryTaskFutureImpl<>(Futures.immediateFuture(value));
  }

  @Override
  public <R> QueryTaskFuture<R> immediateFailedFuture(QueryException e) {
    return new QueryTaskFutureImpl<>(Futures.<R>immediateFailedFuture(e));
  }

  @Override
  public <R> QueryTaskFuture<R> immediateCancelledFuture() {
    return new QueryTaskFutureImpl<>(Futures.<R>immediateCancelledFuture());
  }

  @Override
  public QueryTaskFuture<Void> eval(
      QueryExpression expr, QueryExpressionContext<T> context, final Callback<T> callback) {
    // Not all QueryEnvironment implementations embrace the async+streaming evaluation framework. In
    // particular, the streaming callbacks employed by functions like 'deps' use
    // QueryEnvironment#buildTransitiveClosure. So if the implementation of that method does some
    // heavyweight blocking work, then it's best to do this blocking work in a single batch.
    // Importantly, the callback we pass in needs to maintain order.
    final QueryUtil.AggregateAllCallback<T, ?> aggregateAllCallback =
        QueryUtil.newOrderedAggregateAllOutputFormatterCallback(this);
    QueryTaskFuture<Void> evalAllFuture = expr.eval(this, context, aggregateAllCallback);
    return whenSucceedsCall(
        evalAllFuture,
        new QueryTaskCallable<Void>() {
          @Override
          public Void call() throws QueryException, InterruptedException {
            callback.process(aggregateAllCallback.getResult());
            return null;
          }
        });
  }

  @Override
  public <R> QueryTaskFuture<R> execute(QueryTaskCallable<R> callable) {
    try {
      return immediateSuccessfulFuture(callable.call());
    } catch (QueryException e) {
      return immediateFailedFuture(e);
    } catch (InterruptedException e) {
      return immediateCancelledFuture();
    }
  }

  @Override
  public <R> QueryTaskFuture<R> executeAsync(QueryTaskAsyncCallable<R> callable) {
    return callable.call();
  }

  @Override
  public <R> QueryTaskFuture<R> whenSucceedsCall(
      QueryTaskFuture<?> future, QueryTaskCallable<R> callable) {
    return whenAllSucceedCall(ImmutableList.of(future), callable);
  }

  private static class Dummy implements QueryTaskCallable<Void> {
    public static final Dummy INSTANCE = new Dummy();

    private Dummy() {}

    @Override
    public Void call() {
      return null;
    }
  }

  @Override
  public QueryTaskFuture<Void> whenAllSucceed(Iterable<? extends QueryTaskFuture<?>> futures) {
    return whenAllSucceedCall(futures, Dummy.INSTANCE);
  }

  @Override
  public <R> QueryTaskFuture<R> whenAllSucceedCall(
      Iterable<? extends QueryTaskFuture<?>> futures, QueryTaskCallable<R> callable) {
    return QueryTaskFutureImpl.ofDelegate(
        Futures.whenAllSucceed(cast(futures)).call(callable, directExecutor()));
  }

  @Override
  public <T1, T2> QueryTaskFuture<T2> transformAsync(
      QueryTaskFuture<T1> future, Function<T1, QueryTaskFuture<T2>> function) {
    return QueryTaskFutureImpl.ofDelegate(
        Futures.transformAsync(
            (QueryTaskFutureImpl<T1>) future,
            input -> (QueryTaskFutureImpl<T2>) function.apply(input),
            directExecutor()));
  }

  protected static Iterable<QueryTaskFutureImpl<?>> cast(
      Iterable<? extends QueryTaskFuture<?>> futures) {
    return Iterables.transform(futures, QueryTaskFutureImpl.class::cast);
  }
}
