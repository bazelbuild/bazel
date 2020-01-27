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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A tests(x) filter expression, which returns all the tests in set x, expanding test_suite rules
 * into their constituents.
 *
 * <p>Unfortunately this class reproduces a substantial amount of logic from {@code
 * TestSuiteConfiguredTarget}, albeit in a somewhat simplified form. This is basically inevitable
 * since the expansion of test_suites cannot be done during the loading phase, because it involves
 * inter-package references. We make no attempt to validate the input, or report errors or warnings
 * other than missing target.
 *
 * <pre>expr ::= TESTS '(' expr ')'</pre>
 */
public class TestsFunction implements QueryFunction {
  @VisibleForTesting
  public TestsFunction() {}

  @Override
  public String getName() {
    return "tests";
  }

  @Override
  public int getMandatoryArguments() {
    return 1;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION);
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) {
    Closure<T> closure = new Closure<>(expression, callback, env);

    // A callback that appropriately feeds top-level test and test_suite targets to 'closure'.
    Callback<T> visitAllTestSuitesCallback =
        partialResult -> {
          PartitionResult<T> partitionResult = closure.partition(partialResult);
          closure
              .getUniqueTestSuites(partitionResult.testSuiteTargets)
              .forEach(closure::visitUniqueTestsInUniqueSuite);
          callback.process(closure.getUniqueTests(partitionResult.testTargets));
        };

    // Get a future that represents full evaluation of the argument expression.
    QueryTaskFuture<Void> testSuiteVisitationStartedFuture =
        env.eval(args.get(0).getExpression(), context, visitAllTestSuitesCallback);

    return env.transformAsync(
        // When this future is done, all top-level test_suite targets have already been fed to the
        // 'closure', meaning that ...
        testSuiteVisitationStartedFuture,
        // ... 'closure.getTopLevelRecursiveVisitationFutures()' represents the full visitation of
        // all these test_suite targets.
        dummyVal -> env.whenAllSucceed(closure.getTopLevelRecursiveVisitationFutures()));
  }

  private static class PartitionResult<T> {
    final ImmutableList<T> testTargets;
    final ImmutableList<T> testSuiteTargets;
    final ImmutableList<T> otherTargets;

    private PartitionResult(
        ImmutableList<T> testTargets,
        ImmutableList<T> testSuiteTargets,
        ImmutableList<T> otherTargets) {
      this.testTargets = testTargets;
      this.testSuiteTargets = testSuiteTargets;
      this.otherTargets = otherTargets;
    }
  }

  /** A closure over the state needed to do asynchronous test_suite visitation and expansion. */
  @ThreadSafe
  private static final class Closure<T> {
    private final QueryExpression expression;
    private final Callback<T> callback;
    /** The environment in which this query is being evaluated. */
    private final QueryEnvironment<T> env;

    private final TargetAccessor<T> accessor;
    private final boolean strict;
    private final Uniquifier<T> testUniquifier;
    private final Uniquifier<T> testSuiteUniquifier;
    private final List<QueryTaskFuture<Void>> topLevelRecursiveVisitationFutures =
        Collections.synchronizedList(new ArrayList<>());

    private Closure(QueryExpression expression, Callback<T> callback, QueryEnvironment<T> env) {
      this.expression = expression;
      this.callback = callback;
      this.env = env;
      this.accessor = env.getAccessor();
      this.strict = env.isSettingEnabled(Setting.TESTS_EXPRESSION_STRICT);
      this.testUniquifier = env.createUniquifier();
      this.testSuiteUniquifier = env.createUniquifier();
    }

    private Iterable<T> getUniqueTests(Iterable<T> tests) throws QueryException {
      return testUniquifier.unique(tests);
    }

    private Iterable<T> getUniqueTestSuites(Iterable<T> testSuites) throws QueryException {
      return testSuiteUniquifier.unique(testSuites);
    }

    private void visitUniqueTestsInUniqueSuite(T testSuite) {
      topLevelRecursiveVisitationFutures.add(
          env.executeAsync(() -> recursivelyVisitUniqueTestsInUniqueSuite(testSuite)));
    }

    /**
     * Returns all the futures representing the work items entailed by all the previous calls to
     * {@link #visitUniqueTestsInUniqueSuite}.
     */
    private ImmutableList<QueryTaskFuture<Void>> getTopLevelRecursiveVisitationFutures() {
      return ImmutableList.copyOf(topLevelRecursiveVisitationFutures);
    }

    private QueryTaskFuture<Void> recursivelyVisitUniqueTestsInUniqueSuite(T testSuite) {
      List<String> tagsAttribute = accessor.getStringListAttr(testSuite, "tags");
      // Split the tags list into positive and negative tags
      Set<String> requiredTags = new HashSet<>();
      Set<String> excludedTags = new HashSet<>();
      sortTagsBySense(tagsAttribute, requiredTags, excludedTags);

      List<T> testsToProcess = new ArrayList<>();
      List<T> testSuites;

      try {
        PartitionResult<T> partitionResult = partition(getPrerequisites(testSuite, "tests"));

        for (T testTarget : partitionResult.testTargets) {
          if (includeTest(requiredTags, excludedTags, testTarget)
              && testUniquifier.unique(testTarget)) {
            testsToProcess.add(testTarget);
          }
        }

        testSuites = testSuiteUniquifier.unique(partitionResult.testSuiteTargets);

        // If strict mode is enabled, then give an error for any non-test, non-test-suite target.
        if (strict) {
          for (T otherTarget : partitionResult.otherTargets) {
            env.reportBuildFileError(
                expression,
                "The label '"
                    + accessor.getLabel(otherTarget)
                    + "' in the test_suite '"
                    + accessor.getLabel(testSuite)
                    + "' does not refer to a test or test_suite "
                    + "rule!");
          }
        }

        // Add implicit dependencies on tests in same package, if any.
        for (T target : getPrerequisites(testSuite, "$implicit_tests")) {
          // The Package construction of $implicit_tests ensures that this check never fails, but we
          // add it here anyway for compatibility with future code.
          if (accessor.isTestRule(target)
              && includeTest(requiredTags, excludedTags, target)
              && testUniquifier.unique(target)) {
            testsToProcess.add(target);
          }
        }
      } catch (InterruptedException e) {
        return env.immediateCancelledFuture();
      } catch (QueryException e) {
        return env.immediateFailedFuture(e);
      }

      // Process all tests, asynchronously.
      QueryTaskFuture<Void> allTestsProcessedFuture =
          env.execute(
              () -> {
                callback.process(testsToProcess);
                return null;
              });

      // Visit all suites recursively, asynchronously.
      QueryTaskFuture<Void> allTestSuitsVisitedFuture =
          env.whenAllSucceed(
              Iterables.transform(testSuites, this::recursivelyVisitUniqueTestsInUniqueSuite));

      return env.whenAllSucceed(
          ImmutableList.of(allTestsProcessedFuture, allTestSuitsVisitedFuture));
    }

    private PartitionResult<T> partition(Iterable<T> targets) {
      ImmutableList.Builder<T> testTargetsBuilder = ImmutableList.builder();
      ImmutableList.Builder<T> testSuiteTargetsBuilder = ImmutableList.builder();
      ImmutableList.Builder<T> otherTargetsBuilder = ImmutableList.builder();

      for (T target : targets) {
        if (accessor.isTestRule(target)) {
          testTargetsBuilder.add(target);
        } else if (accessor.isTestSuite(target)) {
          testSuiteTargetsBuilder.add(target);
        } else {
          otherTargetsBuilder.add(target);
        }
      }

      return new PartitionResult<>(
          testTargetsBuilder.build(), testSuiteTargetsBuilder.build(), otherTargetsBuilder.build());
    }

    /**
     * Returns the set of rules named by the attribute 'attrName' of test_suite rule 'testSuite'.
     * The attribute must be a list of labels. If a target cannot be resolved, then an error is
     * reported to the environment (which may throw an exception if {@code keep_going} is disabled).
     *
     * @precondition env.getAccessor().isTestSuite(testSuite)
     */
    private Iterable<T> getPrerequisites(T testSuite, String attrName)
        throws QueryException, InterruptedException {
      return accessor.getPrerequisites(
          expression,
          testSuite,
          attrName,
          "couldn't expand '"
              + attrName
              + "' attribute of test_suite "
              + accessor.getLabel(testSuite)
              + ": ");
    }

    /**
     * Filters 'tests' (by mutation) according to the 'tags' attribute, specifically those that
     * match ALL of the tags in tagsAttribute.
     *
     * @precondition {@code env.getAccessor().isTestSuite(testSuite)}
     * @precondition {@code env.getAccessor().isTestRule(test)}
     */
    private boolean includeTest(Set<String> requiredTags, Set<String> excludedTags, T test) {
      List<String> testTags = new ArrayList<>(accessor.getStringListAttr(test, "tags"));
      testTags.add(accessor.getStringAttr(test, "size"));
      return TestsFunction.includeTest(testTags, requiredTags, excludedTags);
    }
  }

  // TODO(ulfjack): This must match the code in TestTargetUtils. However, we don't currently want
  // to depend on the packages library. Extract to a neutral place?
  /**
   * Decides whether to include a test in a test_suite or not.
   *
   * @param testTags Collection of all tags exhibited by a given test.
   * @param positiveTags Tags declared by the suite. A test must match ALL of these.
   * @param negativeTags Tags declared by the suite. A test must match NONE of these.
   * @return false is the test is to be removed.
   */
  private static boolean includeTest(
      Collection<String> testTags,
      Collection<String> positiveTags,
      Collection<String> negativeTags) {
    // Add this test if it matches ALL of the positive tags and NONE of the
    // negative tags in the tags attribute.
    for (String tag : negativeTags) {
      if (testTags.contains(tag)) {
        return false;
      }
    }
    for (String tag : positiveTags) {
      if (!testTags.contains(tag)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Separates a list of text "tags" into a Pair of Collections, where
   * the first element are the required or positive tags and the second element
   * are the excluded or negative tags.
   * This should work on tag list provided from the command line
   * --test_tags_filters flag or on tag filters explicitly declared in the
   * suite.
   *
   * Keep this function in sync with the version in
   *  java.com.google.devtools.build.lib.view.packages.TestTargetUtils.sortTagsBySense
   *
   * @param tagList A collection of text tags to separate.
   */
  private static void sortTagsBySense(
      Collection<String> tagList, Set<String> requiredTags, Set<String> excludedTags) {
    for (String tag : tagList) {
      if (tag.startsWith("-")) {
        excludedTags.add(tag.substring(1));
      } else if (tag.startsWith("+")) {
        requiredTags.add(tag.substring(1));
      } else if (tag.equals("manual")) {
        // Ignore manual attribute because it is an exception: it is not a filter
        // but a property of test_suite
        continue;
      } else {
        requiredTags.add(tag);
      }
    }
  }
}
