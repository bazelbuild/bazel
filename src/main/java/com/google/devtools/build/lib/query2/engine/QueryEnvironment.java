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

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningExecutorService;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import javax.annotation.Nonnull;

/**
 * The environment of a Blaze query. Implementations do not need to be thread-safe. The generic type
 * T represents a node of the graph on which the query runs; as such, there is no restriction on T.
 * However, query assumes a certain graph model, and the {@link TargetAccessor} class is used to
 * access properties of these nodes.
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

    static Argument of(QueryExpression expression) {
      return new Argument(ArgumentType.EXPRESSION, expression, null, 0);
    }

    static Argument of(String word) {
      return new Argument(ArgumentType.WORD, null, word, 0);
    }

    static Argument of(int integer) {
      return new Argument(ArgumentType.INTEGER, null, null, integer);
    }

    public ArgumentType getType() {
      return type;
    }

    public QueryExpression getExpression() {
      return expression;
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
        case WORD: return "'" + word + "'";
        case EXPRESSION: return expression.toString();
        case INTEGER: return Integer.toString(integer);
        default: throw new IllegalStateException();
      }
    }
  }

  /** A user-defined query function. */
  interface QueryFunction {
    /**
     * Name of the function as it appears in the query language.
     */
    String getName();

    /**
     * The number of arguments that are required. The rest is optional.
     *
     * <p>This should be greater than or equal to zero and at smaller than or equal to the length
     * of the list returned by {@link #getArgumentTypes}.
     */
    int getMandatoryArguments();

    /** The types of the arguments of the function. */
    Iterable<ArgumentType> getArgumentTypes();

    /**
     * Called when a user-defined function is to be evaluated.
     *
     * @param env the query environment this function is evaluated in.
     * @param expression the expression being evaluated.
     * @param args the input arguments. These are type-checked against the specification returned
     *     by {@link #getArgumentTypes} and {@link #getMandatoryArguments}
     */
    <T> void eval(
        QueryEnvironment<T> env,
        VariableContext<T> context,
        QueryExpression expression,
        List<Argument> args,
        Callback<T> callback) throws QueryException, InterruptedException;

    /**
     * Same as {@link #eval(QueryEnvironment, VariableContext, QueryExpression, List, Callback)},
     * except that this {@link QueryFunction} may use {@code executorService} to achieve
     * parallelism.
     *
     * <p>The caller must ensure that {@code env} is thread safe.
     */
    <T> void parEval(
        QueryEnvironment<T> env,
        VariableContext<T> context,
        QueryExpression expression,
        List<Argument> args,
        ThreadSafeCallback<T> callback,
        ListeningExecutorService executorService) throws QueryException, InterruptedException;
  }

  /**
   * Exception type for the case where a target cannot be found. It's basically a wrapper for
   * whatever exception is internally thrown.
   */
  final class TargetNotFoundException extends Exception {
    public TargetNotFoundException(String msg) {
      super(msg);
    }

    public TargetNotFoundException(Throwable cause) {
      super(cause.getMessage(), cause);
    }
  }

  /**
   * Invokes {@code callback} with the set of target nodes in the graph for the specified target
   * pattern, in 'blaze build' syntax.
   */
  void getTargetsMatchingPattern(QueryExpression owner, String pattern, Callback<T> callback)
      throws QueryException, InterruptedException;

  /** Ensures the specified target exists. */
  // NOTE(bazel-team): this method is left here as scaffolding from a previous refactoring. It may
  // be possible to remove it.
  T getOrCreate(T target);

  /** Returns the direct forward dependencies of the specified targets. */
  Collection<T> getFwdDeps(Iterable<T> targets) throws InterruptedException;

  /** Returns the direct reverse dependencies of the specified targets. */
  Collection<T> getReverseDeps(Iterable<T> targets) throws InterruptedException;

  /**
   * Returns the forward transitive closure of all of the targets in "targets". Callers must ensure
   * that {@link #buildTransitiveClosure} has been called for the relevant subgraph.
   */
  Set<T> getTransitiveClosure(Set<T> targets) throws InterruptedException;

  /**
   * Construct the dependency graph for a depth-bounded forward transitive closure
   * of all nodes in "targetNodes".  The identity of the calling expression is
   * required to produce error messages.
   *
   * <p>If a larger transitive closure was already built, returns it to
   * improve incrementality, since all depth-constrained methods filter it
   * after it is built anyway.
   */
  void buildTransitiveClosure(QueryExpression caller,
                              Set<T> targetNodes,
                              int maxDepth) throws QueryException, InterruptedException;

  /** Returns the set of nodes on some path from "from" to "to". */
  Set<T> getNodesOnPath(T from, T to) throws InterruptedException;

  /**
   * Eval an expression {@code expr} and pass the results to the {@code callback}.
   *
   * <p>Note that this method should guarantee that the callback does not see repeated elements.
   * @param expr The expression to evaluate
   * @param callback The caller callback to notify when results are available
   */
  void eval(QueryExpression expr, VariableContext<T> context, Callback<T> callback)
      throws QueryException, InterruptedException;

  /**
   * Creates a Uniquifier for use in a {@code QueryExpression}. Note that the usage of this an
   * uniquifier should not be used for returning unique results to the parent callback. It should
   * only be used to avoid processing the same elements multiple times within this QueryExpression.
   */
  Uniquifier<T> createUniquifier();

  void reportBuildFileError(QueryExpression expression, String msg) throws QueryException;

  /**
   * Returns the set of BUILD, and optionally sub-included and Skylark files that define the given
   * set of targets. Each such file is itself represented as a target in the result.
   */
  Set<T> getBuildFiles(
      QueryExpression caller, Set<T> nodes, boolean buildFiles, boolean subincludes, boolean loads)
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

  /**
   * Returns the set of query functions implemented by this query environment.
   */
  Iterable<QueryFunction> getFunctions();

  /**
   * Settings for the query engine. See {@link QueryEnvironment#isSettingEnabled}.
   */
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

    /**
     * Do not consider host dependencies when traversing dependency edges.
     */
    NO_HOST_DEPS,

    /**
     * Do not consider nodep attributes when traversing dependency edges.
     */
    NO_NODEP_DEPS;
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

    /**
     * Returns the full label of the target as a string, e.g. {@code //some:target}.
     */
    String getLabel(T target);

    /**
     * Returns the label of the target's package as a string, e.g. {@code //some/package}
     */
    String getPackage(T target);

    /**
     * Returns whether the given target is a rule.
     */
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
    List<T> getLabelListAttr(
        QueryExpression caller, T target, String attrName, String errorMsgPrefix)
        throws QueryException, InterruptedException;

    /**
     * If the attribute of the given name on the given target is a string list, then this method
     * returns it.
     *
     * @throws IllegalArgumentException if target is not a rule (according to {@link #isRule}), or
     *                                  if the target does not have an attribute of type string list
     *                                  with the given name
     */
    List<String> getStringListAttr(T target, String attrName);

    /**
     * If the attribute of the given name on the given target is a string, then this method returns
     * it.
     *
     * @throws IllegalArgumentException if target is not a rule (according to {@link #isRule}), or
     *                                  if the target does not have an attribute of type string with
     *                                  the given name
     */
    String getStringAttr(T target, String attrName);

    /**
     * Returns the given attribute represented as a list of strings. For "normal" attributes,
     * this should just be a list of size one containing the attribute's value. For configurable
     * attributes, there should be one entry for each possible value the attribute may take.
     *
     *<p>Note that for backwards compatibility, tristate and boolean attributes are returned as
     * int using the values {@code 0, 1} and {@code -1}. If there is no such attribute, this
     * method returns an empty list.
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

  /** Returns the {@link QueryExpressionEvalListener} that this {@link QueryEnvironment} uses. */
  QueryExpressionEvalListener<T> getEvalListener();

  /** List of the default query functions. */
  List<QueryFunction> DEFAULT_QUERY_FUNCTIONS =
      ImmutableList.of(
          new AllPathsFunction(),
          new BuildFilesFunction(),
          new LoadFilesFunction(),
          new AttrFunction(),
          new FilterFunction(),
          new LabelsFunction(),
          new KindFunction(),
          new SomeFunction(),
          new SomePathFunction(),
          new TestsFunction(),
          new DepsFunction(),
          new RdepsFunction(),
          new VisibleFunction());
}
