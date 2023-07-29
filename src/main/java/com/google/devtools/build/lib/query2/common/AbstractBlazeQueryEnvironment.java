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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.TimeUnit.MINUTES;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.BazelModuleContext.LoadGraphVisitor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ErrorSensingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.server.FailureDetails.Query.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that can evaluate queries to produce a result, and implements as much of
 * QueryEnvironment as possible while remaining mostly agnostic as to the objects being stored.
 */
public abstract class AbstractBlazeQueryEnvironment<T>
    implements QueryEnvironment<T>, AutoCloseable {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  protected ErrorSensingEventHandler<DetailedExitCode> eventHandler;
  protected final boolean keepGoing;
  protected final boolean strictScope;

  protected final DependencyFilter dependencyFilter;
  protected final Predicate<Label> labelFilter;

  protected final Set<Setting> settings;
  protected final List<QueryFunction> extraFunctions;

  protected AbstractBlazeQueryEnvironment(
      boolean keepGoing,
      boolean strictScope,
      Predicate<Label> labelFilter,
      ExtendedEventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions) {
    this.eventHandler = new ErrorSensingEventHandler<>(eventHandler, DetailedExitCode.class);
    this.keepGoing = keepGoing;
    this.strictScope = strictScope;
    this.dependencyFilter = constructDependencyFilter(settings);
    this.labelFilter = labelFilter;
    this.settings = Sets.immutableEnumSet(settings);
    this.extraFunctions = ImmutableList.copyOf(extraFunctions);
  }

  @Override
  public abstract void close();

  private static DependencyFilter constructDependencyFilter(Set<Setting> settings) {
    DependencyFilter specifiedFilter =
        settings.contains(Setting.ONLY_TARGET_DEPS)
            ? DependencyFilter.ONLY_TARGET_DEPS
            : DependencyFilter.ALL_DEPS;
    if (settings.contains(Setting.NO_IMPLICIT_DEPS)) {
      specifiedFilter = specifiedFilter.and(DependencyFilter.NO_IMPLICIT_DEPS);
    }
    if (settings.contains(Setting.NO_NODEP_DEPS)) {
      specifiedFilter = specifiedFilter.and(DependencyFilter.NO_NODEP_ATTRIBUTES);
    }
    return specifiedFilter;
  }

  /**
   * Used by {@link #evaluateQuery} to evaluate the given {@code expr}. The caller, ({@link
   * #evaluateQuery}), is responsible for managing {@code callback}.
   */
  protected void evalTopLevelInternal(QueryExpression expr, OutputFormatterCallback<T> callback)
      throws QueryException, InterruptedException {
    ((QueryTaskFutureImpl<Void>) eval(expr, createEmptyContext(), callback)).getChecked();
  }

  protected QueryExpressionContext<T> createEmptyContext() {
    return QueryExpressionContext.empty();
  }

  public abstract QueryEvalResult evaluateQuery(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<T> callback)
      throws QueryException, IOException, InterruptedException;

  /**
   * Evaluate the specified query expression in this environment, streaming results to the given
   * {@code callback}. {@code callback.start()} will be called before query evaluation and {@code
   * callback.close()} will be unconditionally called at the end of query evaluation (i.e.
   * regardless of whether it was successful).
   *
   * @return a {@link QueryEvalResult} object that contains the resulting set of targets and a bit
   *     to indicate whether errors occurred during evaluation; note that the success status can
   *     only be false if {@code --keep_going} was in effect
   * @throws QueryException if the evaluation failed and {@code --nokeep_going} was in effect
   * @throws IOException for output formatter failures from {@code callback}
   */
  protected final QueryEvalResult evaluateQueryInternal(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<T> callback)
      throws QueryException, InterruptedException, IOException {
    EmptinessSensingCallback<T> emptySensingCallback = new EmptinessSensingCallback<>(callback);
    long startTime = System.currentTimeMillis();
    // In the --nokeep_going case, errors are reported in the order in which the patterns are
    // specified; using a linked hash set here makes sure that the left-most error is reported.
    Set<String> targetPatternSet = new LinkedHashSet<>();
    try (SilentCloseable closeable = Profiler.instance().profile("collectTargetPatterns")) {
      expr.collectTargetPatterns(targetPatternSet);
    }
    try (SilentCloseable closeable = Profiler.instance().profile("preloadOrThrow")) {
      preloadOrThrow(expr, targetPatternSet);
    } catch (TargetParsingException e) {
      // Unfortunately, by evaluating the patterns in parallel, we lose some location information.
      throw new QueryException(expr, e.getMessage(), e.getDetailedExitCode().getFailureDetail());
    }
    IOException ioExn = null;
    boolean failFast = true;
    try {
      callback.start();
      evalTopLevelInternal(expr, emptySensingCallback);
      failFast = false;
    } catch (QueryException e) {
      throw new QueryException(e, expr);
    } finally {
      try {
        callback.close(failFast);
      } catch (IOException e) {
        // Only throw this IOException if we weren't about to throw a different exception.
        ioExn = e;
      }
    }
    if (ioExn != null) {
      throw ioExn;
    }
    long elapsedTime = System.currentTimeMillis() - startTime;
    if (elapsedTime > 1) {
      logger.atInfo().log("Spent %d milliseconds evaluating query", elapsedTime);
    }

    if (eventHandler.hasErrors()) {
      DetailedExitCode detailedExitCode = eventHandler.getErrorProperty();
      if (!keepGoing) {
        if (detailedExitCode != null) {
          throw new QueryException(
              "Evaluation of query \"" + expr.toTrunctatedString() + "\" failed",
              detailedExitCode.getFailureDetail());
        }
        throw new QueryException(
            "Evaluation of query \""
                + expr.toTrunctatedString()
                + "\" failed due to BUILD file errors",
            Query.Code.BUILD_FILE_ERROR);
      }
      eventHandler.handle(
          Event.warn("--keep_going specified, ignoring errors. Results may be inaccurate"));
      if (detailedExitCode != null) {
        return QueryEvalResult.failure(emptySensingCallback.isEmpty(), detailedExitCode);
      } else {
        return QueryEvalResult.failure(
            emptySensingCallback.isEmpty(),
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage(
                        "Evaluation of query \""
                            + expr.toTrunctatedString()
                            + "\" failed due to BUILD file errors")
                    .setQuery(Query.newBuilder().setCode(Code.BUILD_FILE_ERROR))
                    .build()));
      }
    }
    return QueryEvalResult.success(emptySensingCallback.isEmpty());
  }

  @Override
  public <R> QueryTaskFuture<R> immediateSuccessfulFuture(R value) {
    return new QueryTaskFutureImpl<>(Futures.immediateFuture(value));
  }

  @Override
  public <R> QueryTaskFuture<R> immediateFailedFuture(QueryException e) {
    return new QueryTaskFutureImpl<>(Futures.immediateFailedFuture(e));
  }

  @Override
  public <R> QueryTaskFuture<R> immediateCancelledFuture() {
    return new QueryTaskFutureImpl<>(Futures.immediateCancelledFuture());
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
        () -> {
          callback.process(aggregateAllCallback.getResult());
          return null;
        });
  }

  /**
   * Wrapper for evaluating query expression in a non-streaming blaze query environment.
   *
   * <p>In {@link AbstractBlazeQueryEvaluateExpressionImpl}, {@code futureTask} is created only
   * after {@link #eval(Callback)} provides the callback implementation. So creating an {@link
   * AbstractBlazeQueryEvaluateExpressionImpl} instance and calling {@link #eval(Callback)} method
   * should have the same behavior as directly calling {@code
   * AbstractBlazeQueryEnvironment#eval(QueryExpression, QueryExpressionContext, Callback)} above.
   */
  protected class AbstractBlazeQueryEvaluateExpressionImpl implements EvaluateExpression<T> {
    private final QueryExpression expression;
    private final QueryExpressionContext<T> context;
    private QueryTaskFutureImpl<Void> queryTaskFuture;

    private AbstractBlazeQueryEvaluateExpressionImpl(
        QueryExpression expr, QueryExpressionContext<T> context) {
      this.expression = expr;
      this.context = context;
    }

    @Override
    public QueryTaskFuture<Void> eval(Callback<T> callback) {
      queryTaskFuture =
          (QueryTaskFutureImpl<Void>)
              AbstractBlazeQueryEnvironment.this.eval(expression, context, callback);
      return queryTaskFuture;
    }

    @Override
    public boolean gracefullyCancel() {
      // For non-SkyQueryEnvironment-descended environments, there is no need to cancel the future
      // task, so this should be a no-op implementation.
      return false;
    }

    @Override
    public boolean isUngracefullyCancelled() {
      if (queryTaskFuture == null) {
        return false;
      }

      // Since `#gracefullyCancel` is a no-op for `AbstractBlazeQueryEvaluateExpressionImpl`
      // instance, any situation causing the `queryTaskFuture` to be cancelled should be regarded as
      // an ungraceful behavior.
      return queryTaskFuture.isCancelled();
    }
  }

  @Override
  public EvaluateExpression<T> createEvaluateExpression(
      QueryExpression expr, QueryExpressionContext<T> context) {
    return new AbstractBlazeQueryEvaluateExpressionImpl(expr, context);
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
  public <R> QueryTaskFuture<R> whenSucceedsOrIsCancelledCall(
      QueryTaskFuture<?> future, QueryTaskCallable<R> callable) {
    return QueryTaskFutureImpl.whenSucceedsOrIsCancelledCall(
        (QueryTaskFutureImpl<?>) future, callable, directExecutor());
  }

  @Override
  public <T1, T2> QueryTaskFuture<T2> transformAsync(
      QueryTaskFuture<T1> future, Function<T1, QueryTaskFuture<T2>> function) {
    QueryTaskFutureImpl<T1> futureImpl = (QueryTaskFutureImpl<T1>) future;
    if (futureImpl.isDone()) {
      // Due to how our subclasses use single-threaded query engines, in practice
      // futureImpl will always already be done. Therefore this is a fast-path to make it harder to
      // stack overflow on deeply nested expressions whose evaluation involves #transformAsync.
      //
      // TODO(b/283225081): Do something more effective and more pervasive.
      T1 t1;
      try {
        t1 = futureImpl.getChecked();
      } catch (QueryException e) {
        return immediateFailedFuture(e);
      } catch (InterruptedException e) {
        return immediateCancelledFuture();
      }
      return function.apply(t1);
    }
    return QueryTaskFutureImpl.ofDelegate(
        Futures.transformAsync(
            futureImpl,
            input -> (QueryTaskFutureImpl<T2>) function.apply(input),
            directExecutor()));
  }

  private static class EmptinessSensingCallback<T> extends OutputFormatterCallback<T> {
    private final OutputFormatterCallback<T> callback;
    private final AtomicBoolean empty = new AtomicBoolean(true);

    private EmptinessSensingCallback(OutputFormatterCallback<T> callback) {
      this.callback = callback;
    }

    @Override
    public void start() throws IOException {
      callback.start();
    }

    @Override
    public void processOutput(Iterable<T> partialResult) throws IOException, InterruptedException {
      empty.compareAndSet(true, Iterables.isEmpty(partialResult));
      callback.processOutput(partialResult);
    }

    @Override
    public void close(boolean failFast) throws InterruptedException, IOException {
      callback.close(failFast);
    }

    boolean isEmpty() {
      return empty.get();
    }
  }

  public QueryExpression transformParsedQuery(QueryExpression queryExpression) {
    return queryExpression;
  }

  @Override
  public final void handleError(
      QueryExpression expression, String message, DetailedExitCode detailedExitCode)
      throws QueryException {
    if (!keepGoing) {
      if (detailedExitCode != null) {
        throw new QueryException(expression, message, detailedExitCode.getFailureDetail());
      } else {
        BugReport.sendBugReport(
            new IllegalStateException("Undetailed failure: " + message + " for " + expression));
        throw new QueryException(expression, message, Code.NON_DETAILED_ERROR);
      }
    }
    eventHandler.handle(createErrorEvent(expression, message, detailedExitCode));
  }

  public abstract Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException;

  /** Batch version of {@link #getTarget(Label)}. Missing targets are absent in the returned map. */
  // TODO(http://b/128626678): Implement and use this in more places.
  public Map<Label, Target> getTargets(Iterable<Label> labels)
      throws InterruptedException, QueryException {
    ImmutableMap.Builder<Label, Target> resultBuilder = ImmutableMap.builder();
    for (Label label : labels) {
      Target target;
      try {
        target = getTarget(label);
      } catch (TargetNotFoundException e) {
        logger.atInfo().withCause(e).atMostEvery(1, SECONDS).log("Failure to load %s", label);
        continue;
      }
      resultBuilder.put(label, target);
    }
    return resultBuilder.buildOrThrow();
  }

  protected void validateScopeOfTargets(Set<Target> targets) throws QueryException {
    // Sets.filter would be more convenient here, but can't deal with exceptions.
    if (labelFilter != Predicates.<Label>alwaysTrue()) {
      // The labelFilter is always true for bazel query; it's only used for genquery rules.
      Iterator<Target> targetIterator = targets.iterator();
      while (targetIterator.hasNext()) {
        Target target = targetIterator.next();
        if (!validateScope(target.getLabel(), strictScope)) {
          targetIterator.remove();
        }
      }
    }
  }

  protected boolean validateScope(Label label, boolean strict) throws QueryException {
    if (!labelFilter.test(label)) {
      String error = String.format("target '%s' is not within the scope of the query", label);
      if (strict) {
        throw new QueryException(error, Query.Code.TARGET_NOT_IN_UNIVERSE_SCOPE);
      } else {
        eventHandler.handle(Event.warn(error + ". Skipping"));
        return false;
      }
    }
    return true;
  }

  /** Abstract base class for {@link TransitiveLoadFilesHelper<Target>}. */
  protected abstract static class TransitiveLoadFilesHelperForTargets
      implements TransitiveLoadFilesHelper<Target> {
    @Override
    public PackageIdentifier getPkgId(Target target) {
      return target.getLabel().getPackageIdentifier();
    }

    @Override
    public Target getBuildFileTarget(Target originalTarget) {
      return originalTarget.getPackage().getBuildFile();
    }

    @Override
    public void visitLoads(
        Target originalTarget, LoadGraphVisitor<QueryException, InterruptedException> visitor)
        throws QueryException, InterruptedException {
      originalTarget.getPackage().visitLoadGraph(visitor);
    }
  }

  @Override
  public final void transitiveLoadFiles(
      Iterable<T> targets,
      boolean alsoAddBuildFiles,
      Set<PackageIdentifier> seenPackages,
      Set<Label> seenBzlLabels,
      Uniquifier<T> uniquifier,
      TransitiveLoadFilesHelper<T> helper,
      Callback<T> callback)
      throws QueryException, InterruptedException {
    ArrayList<T> result = new ArrayList<>();
    for (T target : targets) {
      PackageIdentifier pkgId = helper.getPkgId(target);
      if (!seenPackages.add(pkgId)) {
        continue;
      }

      if (alsoAddBuildFiles) {
        T buildFileTarget = helper.getBuildFileTarget(target);
        if (uniquifier.unique(buildFileTarget)) {
          result.add(buildFileTarget);
        }
      }

      helper.visitLoads(
          target,
          bzlLabel -> {
            if (!seenBzlLabels.add(bzlLabel)) {
              return false;
            }
            T loadFileTarget = helper.getLoadFileTarget(target, bzlLabel);
            if (uniquifier.unique(loadFileTarget)) {
              result.add(loadFileTarget);
            }
            if (alsoAddBuildFiles) {
              T buildFileTargetForLoadFileTarget =
                  helper.maybeGetBuildFileTargetForLoadFileTarget(target, bzlLabel);
              // Can be null in genquery: see http://b/123795023#comment6.
              if (buildFileTargetForLoadFileTarget != null) {
                if (uniquifier.unique(buildFileTargetForLoadFileTarget)) {
                  result.add(buildFileTargetForLoadFileTarget);
                }
              }
            }
            return true;
          });
    }
    callback.process(result);
  }

  /**
   * Perform any work that should be done ahead of time to resolve the target patterns in the query.
   * Implementations may choose to cache the results of resolving the patterns, cache intermediate
   * work, or not cache and resolve patterns on the fly.
   */
  protected abstract void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws QueryException, TargetParsingException, InterruptedException;

  @Override
  public boolean isSettingEnabled(Setting setting) {
    return settings.contains(Preconditions.checkNotNull(setting));
  }

  @Override
  public Iterable<QueryFunction> getFunctions() {
    ImmutableList.Builder<QueryFunction> builder = ImmutableList.builder();
    builder.addAll(DEFAULT_QUERY_FUNCTIONS);
    builder.addAll(extraFunctions);
    return builder.build();
  }

  /** A {@link KeyExtractor} that extracts {@code Label}s out of {@link Target}s. */
  protected static class TargetKeyExtractor implements KeyExtractor<Target, Label> {
    public static final TargetKeyExtractor INSTANCE = new TargetKeyExtractor();

    private TargetKeyExtractor() {}

    @Override
    public Label extractKey(Target element) {
      return element.getLabel();
    }
  }

  private static Event createErrorEvent(
      QueryExpression expr, String message, @Nullable DetailedExitCode detailedExitCode) {
    String eventMessage =
        String.format("Evaluation of query \"%s\" failed: %s", expr.toTrunctatedString(), message);
    Event event = Event.error(eventMessage);
    if (detailedExitCode != null) {
      event =
          event.withProperty(
              DetailedExitCode.class,
              DetailedExitCode.of(
                  detailedExitCode.getExitCode(),
                  detailedExitCode.getFailureDetail().toBuilder()
                      .setMessage(eventMessage)
                      .build()));
    } else {
      logger.atWarning().atMostEvery(1, MINUTES).log(
          "Null detailed exit code for %s %s", message, expr);
    }
    return event;
  }

  /** Concrete implementation of {@link QueryTaskFuture}. */
  @SuppressWarnings("ShouldNotSubclass")
  protected static final class QueryTaskFutureImpl<T> extends QueryTaskFutureImplBase<T>
      implements ListenableFuture<T> {
    private final ListenableFuture<T> delegate;

    private QueryTaskFutureImpl(ListenableFuture<T> delegate) {
      this.delegate = delegate;
    }

    public static <R> QueryTaskFutureImpl<R> ofDelegate(ListenableFuture<R> delegate) {
      return (delegate instanceof QueryTaskFutureImpl)
          ? (QueryTaskFutureImpl<R>) delegate
          : new QueryTaskFutureImpl<>(delegate);
    }

    public static <R> QueryTaskFutureImpl<R> whenSucceedsOrIsCancelledCall(
        QueryTaskFutureImpl<?> future, QueryTaskCallable<R> callable, Executor executor) {
      return QueryTaskFutureImpl.ofDelegate(
          Futures.whenAllComplete(cast(ImmutableList.of(future)))
              .call(
                  () -> {
                    try {
                      var unused = future.get();
                    } catch (CancellationException unused) {
                      // If the input future is cancelled, we are supposed to swallow the
                      // `CancellationException` and proceed normally.
                    }
                    return callable.call();
                  },
                  executor));
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

    private T getChecked() throws InterruptedException, QueryException {
      try {
        return get();
      } catch (CancellationException unused) {
        throw new InterruptedException();
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        Throwables.propagateIfPossible(cause, QueryException.class);
        Throwables.propagateIfPossible(cause, InterruptedException.class);
        throw new IllegalStateException(e);
      }
    }
  }

  private static class Dummy implements QueryTaskCallable<Void> {
    public static final Dummy INSTANCE = new Dummy();

    private Dummy() {}

    @Override
    public Void call() {
      return null;
    }
  }

  protected static Iterable<QueryTaskFutureImpl<?>> cast(
      Iterable<? extends QueryTaskFuture<?>> futures) {
    return Iterables.transform(futures, QueryTaskFutureImpl.class::cast);
  }
}
