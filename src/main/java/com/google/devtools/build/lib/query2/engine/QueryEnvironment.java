// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The environment of a Blaze query. Implementations do not need to be thread-safe. The generic type
 * T represents a node of the graph on which the query runs; as such, there is no restriction on T.
 * However, query assumes a certain graph model, and the {@link TargetAccessor} class is used to
 * access properties of these nodes. Also, the query engine doesn't assume T's {@link
 * Object#hashCode} and {@link Object#equals} are meaningful and instead uses {@link
 * QueryEnvironment#createUniquifier}, {@link QueryEnvironment#createThreadSafeMutableSet()}, and
 * {@link QueryEnvironment#createMutableMap()} when appropriate.
 *
 * @param <T> the node type of the dependency graph
 */
public interface QueryEnvironment<T> {
  /** Type of an argument of a user-defined query function. */
  enum ArgumentType {
    EXPRESSION,
    WORD,
    INTEGER;
  }

  /** Value of an argument of a user-defined query function. */
  class Argument {
    private final ArgumentType type;
    private final QueryExpression expression;
    private final String word;
    private final int integer;

    private Argument(ArgumentType type, QueryExpression expression, String word, int integer) {
      this.type = type;
      this.expression = expression;
      this.word = word;
      this.integer = integer;
    }

    public static Argument of(QueryExpression expression) {
      return new Argument(ArgumentType.EXPRESSION, expression, null, 0);
    }

    public static Argument of(String word) {
      return new Argument(ArgumentType.WORD, null, word, 0);
    }

    public static Argument of(int integer) {
      return new Argument(ArgumentType.INTEGER, null, null, integer);
    }

    public ArgumentType getType() {
      return type;
    }

    public QueryExpression getExpression() {
      return Preconditions.checkNotNull(expression, "Expected expression argument");
    }

    public String getWord() {
      return word;
    }

    public int getInteger() {
      return integer;
    }

    @Override
    public String toString() {
      switch (type) {
        case WORD:
          return "'" + word + "'";
        case EXPRESSION:
          return expression.toString();
        case INTEGER:
          return Integer.toString(integer);
      }
      throw new IllegalStateException();
    }
  }

  /** A user-defined query function. */
  interface QueryFunction {
    /** Name of the function as it appears in the query language. */
    String getName();

    /**
     * The number of arguments that are required. The rest is optional.
     *
     * <p>This should be greater than or equal to zero and at smaller than or equal to the length of
     * the list returned by {@link #getArgumentTypes}.
     */
    int getMandatoryArguments();

    /** The types of the arguments of the function. */
    Iterable<ArgumentType> getArgumentTypes();

    /**
     * Returns a {@link QueryTaskFuture} representing the asynchronous application of this {@link
     * QueryFunction} to the given {@code args}, feeding the results to the given {@code callback}.
     *
     * @param env the query environment this function is evaluated in.
     * @param expression the expression being evaluated.
     * @param context the context relevant to the expression being evaluated. Contains the variable
     *     bindings from {@link LetExpression}s.
     * @param args the input arguments. These are type-checked against the specification returned by
     *     {@link #getArgumentTypes} and {@link #getMandatoryArguments}
     */
    <T> QueryTaskFuture<Void> eval(
        QueryEnvironment<T> env,
        QueryExpressionContext<T> context,
        QueryExpression expression,
        List<Argument> args,
        Callback<T> callback);

    /**
     * A filtering function is one whose outputs are a subset of a single input argument. Returns
     * the function as a filtering function if it is one and {@code null} otherwise.
     */
    @Nullable
    default FilteringQueryFunction asFilteringFunction() {
      return null;
    }
  }

  /** A {@link QueryFunction} whose output is a subset of some input argument expression. */
  abstract class FilteringQueryFunction implements QueryFunction {
    @Override
    public final FilteringQueryFunction asFilteringFunction() {
      return this;
    }

    /** Returns a function representing the filter but inverted. */
    public abstract FilteringQueryFunction invert();

    /** Returns the argument index of the expression that is used as the input to be filtered. */
    public abstract int getExpressionToFilterIndex();
  }

  /**
   * Exception type for the case where a target cannot be found. It's basically a wrapper for
   * whatever exception is internally thrown.
   */
  final class TargetNotFoundException extends Exception {
    private final DetailedExitCode detailedExitCode;

    public TargetNotFoundException(Throwable cause, DetailedExitCode detailedExitCode) {
      super(cause.getMessage(), cause);
      this.detailedExitCode = Preconditions.checkNotNull(detailedExitCode);
    }

    public DetailedExitCode getDetailedExitCode() {
      return detailedExitCode;
    }
  }

  /**
   * QueryEnvironment implementations can optionally also implement this interface to provide custom
   * implementations of various operators.
   */
  interface CustomFunctionQueryEnvironment<T> extends QueryEnvironment<T> {
    /**
     * Computes the transitive closure of dependencies at most maxDepth away from the given targets,
     * and calls the given callback with the results.
     */
    void deps(Iterable<T> from, int maxDepth, QueryExpression caller, Callback<T> callback)
        throws InterruptedException, QueryException;

    /** Computes some path from a node in 'from' to a node in 'to'. */
    void somePath(Iterable<T> from, Iterable<T> to, QueryExpression caller, Callback<T> callback)
        throws InterruptedException, QueryException;

    /** Computes all paths from a node in 'from' to a node in 'to'. */
    void allPaths(Iterable<T> from, Iterable<T> to, QueryExpression caller, Callback<T> callback)
        throws InterruptedException, QueryException;

    /**
     * Computes all reverse dependencies of a node in 'from' with at most distance maxDepth within
     * the transitive closure of 'universe'.
     */
    void rdeps(
        Iterable<T> from,
        Iterable<T> universe,
        int maxDepth,
        QueryExpression caller,
        Callback<T> callback)
        throws InterruptedException, QueryException;

    /** Computes direct reverse deps of all nodes in 'from' within the same package. */
    void samePkgDirectRdeps(Iterable<T> from, QueryExpression caller, Callback<T> callback)
        throws InterruptedException, QueryException;
  }

  /** Returns all of the targets in <code>target</code>'s package, in some stable order. */
  Collection<T> getSiblingTargetsInPackage(T target) throws QueryException;

  /**
   * Invokes {@code callback} with the set of target nodes in the graph for the specified target
   * pattern, in 'blaze build' syntax.
   */
  QueryTaskFuture<Void> getTargetsMatchingPattern(
      QueryExpression owner, String pattern, Callback<T> callback);

  /** Ensures the specified target exists. */
  // NOTE(bazel-team): this method is left here as scaffolding from a previous refactoring. It may
  // be possible to remove it.
  T getOrCreate(T target);

  /** Returns the direct forward dependencies of the specified targets. */
  Iterable<T> getFwdDeps(Iterable<T> targets, QueryExpressionContext<T> context)
      throws InterruptedException;

  /** Returns the direct reverse dependencies of the specified targets. */
  Iterable<T> getReverseDeps(Iterable<T> targets, QueryExpressionContext<T> context)
      throws InterruptedException;

  /**
   * Returns the forward transitive closure of all of the targets in "targets". Callers must ensure
   * that {@link #buildTransitiveClosure} has been called for the relevant subgraph.
   */
  ThreadSafeMutableSet<T> getTransitiveClosure(
      ThreadSafeMutableSet<T> targets, QueryExpressionContext<T> context)
      throws InterruptedException;

  /**
   * Construct the dependency graph for a depth-bounded forward transitive closure of all nodes in
   * "targetNodes". The identity of the calling expression is required to produce error messages.
   *
   * <p>If a larger transitive closure was already built, returns it to improve incrementality,
   * since all depth-constrained methods filter it after it is built anyway.
   */
  void buildTransitiveClosure(
      QueryExpression caller, ThreadSafeMutableSet<T> targetNodes, int maxDepth)
      throws QueryException, InterruptedException;

  /** Returns the ordered sequence of nodes on some path from "from" to "to". */
  Iterable<T> getNodesOnPath(T from, T to, QueryExpressionContext<T> context)
      throws InterruptedException;

  /**
   * Returns a {@link QueryTaskFuture} representing the asynchronous evaluation of the given {@code
   * expr} and passing of the results to the given {@code callback}.
   *
   * <p>Note that this method should guarantee that the callback does not see repeated elements.
   *
   * @param expr The expression to evaluate
   * @param callback The caller callback to notify when results are available
   */
  QueryTaskFuture<Void> eval(
      QueryExpression expr, QueryExpressionContext<T> context, Callback<T> callback);

  /**
   * An asynchronous computation of part of a query evaluation.
   *
   * <p>A {@link QueryTaskFuture} can only be produced from scratch via {@link #eval}, {@link
   * #execute}, {@link #immediateSuccessfulFuture}, {@link #immediateFailedFuture}, and {@link
   * #immediateCancelledFuture}.
   *
   * <p>Combined with the helper methods like {@link #whenSucceedsCall} below, this is very similar
   * to Guava's {@link ListenableFuture}.
   *
   * <p>This class is deliberately opaque; the only ways to compose/use {@link #QueryTaskFuture}
   * instances are the helper methods like {@link #whenSucceedsCall} below. A crucial consequence of
   * this is there is no way for a {@link QueryExpression} or {@link QueryFunction} implementation
   * to block on the result of a {@link #QueryTaskFuture}. This eliminates a large class of
   * deadlocks by design!
   */
  @ThreadSafe
  public abstract class QueryTaskFuture<T> {
    // We use a public abstract class with a private constructor so that this type is visible to all
    // the query codebase, but yet the only possible implementation is under our control in this
    // file.
    private QueryTaskFuture() {}

    /**
     * If this {@link QueryTasksFuture}'s encapsulated computation is currently complete and
     * successful, returns the result. This method is intended to be used in combination with {@link
     * #whenSucceedsCall}.
     *
     * <p>See the javadoc for the various helper methods that produce {@link QueryTasksFuture} for
     * the precise definition of "successful".
     */
    public abstract T getIfSuccessful();
  }

  /**
   * Returns a {@link QueryTaskFuture} representing the successful computation of {@code value}.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "successful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful}.
   */
  abstract <R> QueryTaskFuture<R> immediateSuccessfulFuture(R value);

  /**
   * Returns a {@link QueryTaskFuture} representing a computation that was unsuccessful because of
   * {@code e}.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "unsuccessful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful}.
   */
  abstract <R> QueryTaskFuture<R> immediateFailedFuture(QueryException e);

  /**
   * Returns a {@link QueryTaskFuture} representing a cancelled computation.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "unsuccessful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful}.
   */
  abstract <R> QueryTaskFuture<R> immediateCancelledFuture();

  /** A {@link ThreadSafe} {@link Callable} for computations during query evaluation. */
  @ThreadSafe
  public interface QueryTaskCallable<T> extends Callable<T> {
    /**
     * Returns the computed value or throws a {@link QueryException} on failure or a {@link
     * InterruptedException} on interruption.
     */
    @Override
    T call() throws QueryException, InterruptedException;
  }

  /** Like Guava's AsyncCallable, but for {@link QueryTaskFuture}. */
  @ThreadSafe
  public interface QueryTaskAsyncCallable<T> {
    /**
     * Returns a {@link QueryTaskFuture} whose completion encapsulates the result of the
     * computation.
     */
    QueryTaskFuture<T> call();
  }

  /**
   * Returns a {@link QueryTaskFuture} representing the given computation {@code callable} being
   * performed asynchronously.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "successful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful} iff
   * {@code callable#call} does not throw an exception.
   */
  <R> QueryTaskFuture<R> execute(QueryTaskCallable<R> callable);

  /**
   * Returns a {@link QueryTaskFuture} representing both the given {@code callable} being performed
   * asynchronously and also the returned {@link QueryTaskFuture} returned therein being completed.
   */
  <R> QueryTaskFuture<R> executeAsync(QueryTaskAsyncCallable<R> callable);

  /**
   * Returns a {@link QueryTaskFuture} representing the given computation {@code callable} being
   * performed after the successful completion of the computation encapsulated by the given {@code
   * future} has completed successfully.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "successful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful} iff
   * {@code future} is successful and {@code callable#call} does not throw an exception.
   */
  <R> QueryTaskFuture<R> whenSucceedsCall(QueryTaskFuture<?> future, QueryTaskCallable<R> callable);

  /**
   * Returns a {@link QueryTaskFuture} representing the successful completion of all the
   * computations encapsulated by the given {@code futures}.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "successful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful} iff
   * all of the given computations are "successful".
   */
  QueryTaskFuture<Void> whenAllSucceed(Iterable<? extends QueryTaskFuture<?>> futures);

  /**
   * Returns a {@link QueryTaskFuture} representing the given computation {@code callable} being
   * performed after the successful completion of all the computations encapsulated by the given
   * {@code futures}.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "successful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful} iff
   * all of the given computations are "successful" and {@code callable#call} does not throw an
   * exception.
   */
  <R> QueryTaskFuture<R> whenAllSucceedCall(
      Iterable<? extends QueryTaskFuture<?>> futures, QueryTaskCallable<R> callable);

  /**
   * Returns a {@link QueryTaskFuture} representing the asynchronous application of the given {@code
   * function} to the value produced by the computation encapsulated by the given {@code future}.
   *
   * <p>The returned {@link QueryTaskFuture} is considered "successful" for purposes of {@link
   * #whenSucceedsCall}, {@link #whenAllSucceed}, and {@link QueryTaskFuture#getIfSuccessful} iff
   * {@code} future is "successful".
   */
  <T1, T2> QueryTaskFuture<T2> transformAsync(
      QueryTaskFuture<T1> future, Function<T1, QueryTaskFuture<T2>> function);

  /**
   * The sole package-protected subclass of {@link QueryTaskFuture}.
   *
   * <p>Do not subclass this class; it's an implementation detail. {@link QueryExpression} and
   * {@link QueryFunction} implementations should use {@link #eval} and {@link #execute} to get
   * access to {@link QueryTaskFuture} instances and the then use the helper methods like {@link
   * #whenSucceedsCall} to transform them.
   */
  abstract class QueryTaskFutureImplBase<T> extends QueryTaskFuture<T> {
    protected QueryTaskFutureImplBase() {}
  }

  /**
   * A mutable {@link ThreadSafe} {@link Set} that uses proper equality semantics for {@code T}.
   * {@link QueryExpression}/{@link QueryFunction} implementations should use {@code
   * ThreadSafeMutableSet<T>} they need a set-like data structure for {@code T}.
   */
  @ThreadSafe
  interface ThreadSafeMutableSet<T> extends Set<T> {}

  /** Returns a fresh {@link ThreadSafeMutableSet} instance for the type {@code T}. */
  ThreadSafeMutableSet<T> createThreadSafeMutableSet();

  /**
   * A simple map-like interface that uses proper equality semantics for the key type. {@link
   * QueryExpression}/{@link QueryFunction} implementations should use {@code
   * ThreadSafeMutableSet<T, V>} they need a map-like data structure for {@code T}.
   */
  interface MutableMap<K, V> {
    /**
     * Returns the value {@code value} associated with the given key by the most recent call to
     * {@code put(key, value)}, or {@code null} if there was no such call.
     */
    @Nullable
    V get(K key);

    /**
     * Associates the given key with the given value and returns the previous value associated with
     * the key, or {@code null} if there wasn't one.
     */
    V put(K key, V value);
  }

  /** Returns a fresh {@link MutableMap} instance with key type {@code T}. */
  <V> MutableMap<T, V> createMutableMap();

  /**
   * Creates a Uniquifier for use in a {@code QueryExpression}. Note that the usage of this
   * uniquifier should not be used for returning unique results to the parent callback. It should
   * only be used to avoid processing the same elements multiple times within this QueryExpression.
   */
  Uniquifier<T> createUniquifier();

  /**
   * Creates a {@link MinDepthUniquifier} for use in a {@code QueryExpression}. Note that the usage
   * of this uniquifier should not be used for returning unique results to the parent callback. It
   * should only be used to try to avoid processing the same elements multiple times at the same
   * depth bound within this QueryExpression.
   */
  MinDepthUniquifier<T> createMinDepthUniquifier();

  void reportBuildFileError(QueryExpression expression, String msg) throws QueryException;

  /**
   * Returns the set of BUILD, and optionally Starlark files that define the given set of targets.
   * Each such file is itself represented as a target in the result.
   */
  ThreadSafeMutableSet<T> getBuildFiles(
      QueryExpression caller,
      ThreadSafeMutableSet<T> nodes,
      boolean buildFiles,
      boolean loads,
      QueryExpressionContext<T> context)
      throws QueryException, InterruptedException;

  /**
   * Returns an object that can be used to query information about targets. Implementations should
   * create a single instance and return that for all calls. A class can implement both {@code
   * QueryEnvironment} and {@code TargetAccessor} at the same time, in which case this method simply
   * returns {@code this}.
   */
  TargetAccessor<T> getAccessor();

  /**
   * Whether the given setting is enabled. The code should default to return {@code false} for all
   * unknown settings. The enum is used rather than a method for each setting so that adding more
   * settings is backwards-compatible.
   *
   * @throws NullPointerException if setting is null
   */
  boolean isSettingEnabled(@Nonnull Setting setting);

  /** Returns the set of query functions implemented by this query environment. */
  Iterable<QueryFunction> getFunctions();

  /** Settings for the query engine. See {@link QueryEnvironment#isSettingEnabled}. */
  enum Setting {

    /**
     * Whether to evaluate tests() expressions in strict mode. If {@link #isSettingEnabled} returns
     * true for this setting, then the tests() expression will give an error when expanding tests
     * suites, if the test suite contains any non-test targets.
     */
    TESTS_EXPRESSION_STRICT,

    /**
     * Do not consider implicit deps (any label that was not explicitly specified in the BUILD file)
     * when traversing dependency edges.
     */
    NO_IMPLICIT_DEPS,

    /** Do not consider non-target dependencies when traversing dependency edges. */
    ONLY_TARGET_DEPS,

    /** Do not consider nodep attributes when traversing dependency edges. */
    NO_NODEP_DEPS,

    /** Include aspect-generated output. No-op for query, which always follows aspects. */
    INCLUDE_ASPECTS;
  }

  /**
   * An adapter interface giving access to properties of T. There are four types of targets: rules,
   * package groups, source files, and generated files. Of these, only rules can have attributes.
   */
  interface TargetAccessor<T> {
    /**
     * Returns the target type represented as a string of the form {@code &lt;type&gt; rule} or
     * {@code package group} or {@code source file} or {@code generated file}. This is widely used
     * for target filtering, so implementations must use the Blaze rule class naming scheme.
     */
    String getTargetKind(T target);

    /** Returns the full label of the target as a string, e.g. {@code //some:target}. */
    String getLabel(T target);

    /** Returns the label of the target's package as a string, e.g. {@code //some/package} */
    String getPackage(T target);

    /** Returns whether the given target is a rule. */
    boolean isRule(T target);

    /**
     * Returns whether the given target is a test target. If this returns true, then {@link #isRule}
     * must also return true for the target.
     */
    boolean isTestRule(T target);

    /**
     * Returns whether the given target is a test suite target. If this returns true, then {@link
     * #isRule} must also return true for the target, but {@link #isTestRule} must return false;
     * test suites are not test rules, and vice versa.
     */
    boolean isTestSuite(T target);

    /**
     * If the attribute of the given name on the given target is a label or label list, then this
     * method returns the list of corresponding target instances. Otherwise returns an empty list.
     * If an error occurs during resolution, it throws a {@link QueryException} using the caller and
     * error message prefix.
     *
     * @throws IllegalArgumentException if target is not a rule (according to {@link #isRule})
     */
    Iterable<T> getPrerequisites(
        QueryExpression caller, T target, String attrName, String errorMsgPrefix)
        throws QueryException, InterruptedException;

    /**
     * If the attribute of the given name on the given target is a string list, then this method
     * returns it.
     *
     * @throws IllegalArgumentException if target is not a rule (according to {@link #isRule}), or
     *     if the target does not have an attribute of type string list with the given name
     */
    List<String> getStringListAttr(T target, String attrName);

    /**
     * If the attribute of the given name on the given target is a string, then this method returns
     * it.
     *
     * @throws IllegalArgumentException if target is not a rule (according to {@link #isRule}), or
     *     if the target does not have an attribute of type string with the given name
     */
    String getStringAttr(T target, String attrName);

    /**
     * Returns the given attribute represented as a list of strings. For "normal" attributes, this
     * should just be a list of size one containing the attribute's value. For configurable
     * attributes, there should be one entry for each possible value the attribute may take.
     *
     * <p>Note that for backwards compatibility, tristate and boolean attributes are returned as int
     * using the values {@code 0, 1} and {@code -1}. If there is no such attribute, this method
     * returns an empty list.
     *
     * @throws IllegalArgumentException if target is not a rule (according to {@link #isRule})
     */
    Iterable<String> getAttrAsString(T target, String attrName);

    /**
     * Returns the set of package specifications the given target is visible from, represented as
     * {@link QueryVisibility}s.
     */
    Set<QueryVisibility<T>> getVisibility(T from) throws QueryException, InterruptedException;
  }

  /** List of the default query functions. */
  ImmutableList<QueryFunction> DEFAULT_QUERY_FUNCTIONS =
      ImmutableList.of(
          new AllPathsFunction(),
          new AttrFunction(),
          new BuildFilesFunction(),
          new DepsFunction(),
          new FilterFunction(),
          new KindFunction(),
          new LabelsFunction(),
          new LoadFilesFunction(),
          new RdepsFunction(),
          new SamePkgDirectRdepsFunction(),
          new SiblingsFunction(),
          new SomeFunction(),
          new SomePathFunction(),
          new TestsFunction(),
          new VisibleFunction());
}
