// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A tests(x) filter expression, which returns all the tests in set x,
 * expanding test_suite rules into their constituents.
 *
 * <p>Unfortunately this class reproduces a substantial amount of logic from
 * {@code TestSuiteConfiguredTarget}, albeit in a somewhat simplified form.
 * This is basically inevitable since the expansion of test_suites cannot be
 * done during the loading phase, because it involves inter-package references.
 * We make no attempt to validate the input, or report errors or warnings other
 * than missing target.
 *
 * <pre>expr ::= TESTS '(' expr ')'</pre>
 */
class TestsFunction implements QueryFunction {
  TestsFunction() {
  }

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
  public <T> Set<T> eval(QueryEnvironment<T> env, QueryExpression expression, List<Argument> args)
      throws QueryException, InterruptedException {
    Closure<T> closure = new Closure<>(expression, env);
    Set<T> result = new HashSet<>();
    for (T target : args.get(0).getExpression().eval(env)) {
      if (env.getAccessor().isTestRule(target)) {
        result.add(target);
      } else if (env.getAccessor().isTestSuite(target)) {
        for (T test : closure.getTestsInSuite(target)) {
          result.add(env.getOrCreate(test));
        }
      }
    }
    return result;
  }

  /**
   * Decides whether to include a test in a test_suite or not.
   * @param testTags Collection of all tags exhibited by a given test.
   * @param positiveTags Tags declared by the suite. A test must match ALL of these.
   * @param negativeTags Tags declared by the suite. A test must match NONE of these.
   * @return false is the test is to be removed.
   */
  private static boolean includeTest(Collection<String> testTags,
      Collection<String> positiveTags, Collection<String> negativeTags) {
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

  /**
   * A closure over the temporary state needed to compute the expression. This makes the evaluation
   * thread-safe, as long as instances of this class are used only within a single thread.
   */
  private final class Closure<T> {
    private final QueryExpression expression;
    /** A dynamically-populated mapping from test_suite rules to their tests. */
    private final Map<T, Set<T>> testsInSuite = new HashMap<>();

    /** The environment in which this query is being evaluated. */
    private final QueryEnvironment<T> env;

    private final boolean strict;

    private Closure(QueryExpression expression, QueryEnvironment<T> env) {
      this.expression = expression;
      this.env = env;
      this.strict = env.isSettingEnabled(Setting.TESTS_EXPRESSION_STRICT);
    }

    /**
     * Computes and returns the set of test rules in a particular suite.  Uses
     * dynamic programming---a memoized version of {@link #computeTestsInSuite}.
     *
     * @precondition env.getAccessor().isTestSuite(testSuite)
     */
    private Set<T> getTestsInSuite(T testSuite) throws QueryException {
      Set<T> tests = testsInSuite.get(testSuite);
      if (tests == null) {
        tests = Sets.newHashSet();
        testsInSuite.put(testSuite, tests); // break cycles by inserting empty set early.
        computeTestsInSuite(testSuite, tests);
      }
      return tests;
    }

    /**
     * Populates 'result' with all the tests associated with the specified
     * 'testSuite'.  Throws an exception if any target is missing.
     *
     * <p>CAUTION!  Keep this logic consistent with {@code TestsSuiteConfiguredTarget}!
     *
     * @precondition env.getAccessor().isTestSuite(testSuite)
     */
    private void computeTestsInSuite(T testSuite, Set<T> result) throws QueryException {
      List<T> testsAndSuites = new ArrayList<>();
      // Note that testsAndSuites can contain input file targets; the test_suite rule does not
      // restrict the set of targets that can appear in tests or suites.
      testsAndSuites.addAll(getPrerequisites(testSuite, "tests"));
      testsAndSuites.addAll(getPrerequisites(testSuite, "suites"));

      // 1. Add all tests
      for (T test : testsAndSuites) {
        if (env.getAccessor().isTestRule(test)) {
          result.add(test);
        } else if (strict && !env.getAccessor().isTestSuite(test)) {
          // If strict mode is enabled, then give an error for any non-test, non-test-suite targets.
          env.reportBuildFileError(expression, "The label '"
              + env.getAccessor().getLabel(test) + "' in the test_suite '"
              + env.getAccessor().getLabel(testSuite) + "' does not refer to a test or test_suite "
              + "rule!");
        }
      }

      // 2. Add implicit dependencies on tests in same package, if any.
      for (T target : getPrerequisites(testSuite, "$implicit_tests")) {
        // The Package construction of $implicit_tests ensures that this check never fails, but we
        // add it here anyway for compatibility with future code.
        if (env.getAccessor().isTestRule(target)) {
          result.add(target);
        }
      }

      // 3. Filter based on tags, size, env.
      filterTests(testSuite, result);

      // 4. Expand all suites recursively.
      for (T suite : testsAndSuites) {
        if (env.getAccessor().isTestSuite(suite)) {
          result.addAll(getTestsInSuite(suite));
        }
      }
    }

    /**
     * Returns the set of rules named by the attribute 'attrName' of test_suite rule 'testSuite'.
     * The attribute must be a list of labels. If a target cannot be resolved, then an error is
     * reported to the environment (which may throw an exception if {@code keep_going} is disabled).
     *
     * @precondition env.getAccessor().isTestSuite(testSuite)
     */
    private List<T> getPrerequisites(T testSuite, String attrName) throws QueryException {
      return env.getAccessor().getLabelListAttr(expression, testSuite, attrName,
          "couldn't expand '" + attrName
          + "' attribute of test_suite " + env.getAccessor().getLabel(testSuite) + ": ");
    }

    /**
     * Filters 'tests' (by mutation) according to the 'tags' attribute, specifically those that
     * match ALL of the tags in tagsAttribute.
     *
     * @precondition {@code env.getAccessor().isTestSuite(testSuite)}
     * @precondition {@code env.getAccessor().isTestRule(test)} for all test in tests
     */
    private void filterTests(T testSuite, Set<T> tests) {
      List<String> tagsAttribute = env.getAccessor().getStringListAttr(testSuite, "tags");
      // Split the tags list into positive and negative tags
      Set<String> requiredTags = new HashSet<>();
      Set<String> excludedTags = new HashSet<>();
      sortTagsBySense(tagsAttribute, requiredTags, excludedTags);

      Iterator<T> it = tests.iterator();
      while (it.hasNext()) {
        T test = it.next();
        List<String> testTags = new ArrayList<>(env.getAccessor().getStringListAttr(test, "tags"));
        testTags.add(env.getAccessor().getStringAttr(test, "size"));
        if (!includeTest(testTags, requiredTags, excludedTags)) {
          it.remove();
        }
      }
    }
  }
}
