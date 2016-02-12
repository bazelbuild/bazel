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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.skyframe.TestsInSuiteValue.TestsInSuite;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * TestsInSuiteFunction takes a single test_suite target and expands all of the tests it contains,
 * possibly recursively.
 */
// TODO(ulfjack): What about test_suite rules that include each other.
final class TestsInSuiteFunction implements SkyFunction {
  @Override
  public SkyValue compute(SkyKey key, Environment env) {
    TestsInSuite expansion = (TestsInSuite) key.argument();
    ResolvedTargets<Target> result =
        computeTestsInSuite(env, expansion.getTestSuite(), expansion.isStrict());
    if (env.valuesMissing()) {
      return null;
    }
    return new TestsInSuiteValue(result);
  }

  /**
   * Populates 'result' with all the tests associated with the specified
   * 'testSuite'.  Throws an exception if any target is missing.
   *
   * <p>CAUTION!  Keep this logic consistent with {@code TestSuite}!
   */
  private ResolvedTargets<Target> computeTestsInSuite(
      Environment env, Rule testSuite, boolean strict) {
    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    List<Target> testsAndSuites = new ArrayList<>();
    // Note that testsAndSuites can contain input file targets; the test_suite rule does not
    // restrict the set of targets that can appear in tests or suites.
    builder.mergeError(getPrerequisites(env, testSuite, "tests", testsAndSuites));
    if (testSuite.getRuleClassObject().hasAttr("suites", BuildType.LABEL_LIST)) {
      builder.mergeError(getPrerequisites(env, testSuite, "suites", testsAndSuites));
    }

    // 1. Add all tests
    for (Target test : testsAndSuites) {
      if (TargetUtils.isTestRule(test)) {
        builder.add(test);
      } else if (strict && !TargetUtils.isTestSuiteRule(test)) {
        // If strict mode is enabled, then give an error for any non-test, non-test-suite targets.
        // TODO(ulfjack): We need to throw to end the process if we happen to be in --nokeep_going,
        // but we can't know whether or not we are at this point.
        env.getListener().handle(Event.error(testSuite.getLocation(),
            "in test_suite rule '" + testSuite.getLabel()
            + "': expecting a test or a test_suite rule but '" + test.getLabel()
            + "' is not one."));
        builder.setError();
      }
    }

    // 2. Add implicit dependencies on tests in same package, if any.
    List<Target> implicitTests = new ArrayList<>();
    builder.mergeError(getPrerequisites(env, testSuite, "$implicit_tests", implicitTests));
    for (Target target : implicitTests) {
      // The Package construction of $implicit_tests ensures that this check never fails, but we
      // add it here anyway for compatibility with future code.
      if (TargetUtils.isTestRule(target)) {
        builder.add(target);
      }
    }

    // 3. Filter based on tags, size, env.
    filterTests(testSuite, builder);

    // 4. Expand all suites recursively.
    for (Target suite : testsAndSuites) {
      if (TargetUtils.isTestSuiteRule(suite)) {
        TestsInSuiteValue value =
            (TestsInSuiteValue) env.getValue(TestsInSuiteValue.key(suite, strict));
        if (value == null) {
          continue;
        }
        builder.merge(value.getTargets());
      }
    }

    return builder.build();
  }

  /**
   * Adds the set of targets found in the attribute named {@code attrName}, which must be of label
   * list type, of the {@code test_suite} rule named {@code testSuite}. Returns true if the method
   * found a problem during the lookup process; the actual error message is reported to the
   * environment.
   */
  private boolean getPrerequisites(Environment env, Rule testSuite, String attrName,
      List<Target> targets) {
    List<Label> labels =
        NonconfigurableAttributeMapper.of(testSuite).get(attrName, BuildType.LABEL_LIST);
    Set<PackageIdentifier> pkgIdentifiers = new LinkedHashSet<>();
    for (Label label : labels) {
      pkgIdentifiers.add(label.getPackageIdentifier());
    }
    Map<SkyKey, ValueOrException<BuildFileNotFoundException>> packages = env.getValuesOrThrow(
        PackageValue.keys(pkgIdentifiers), BuildFileNotFoundException.class);
    if (env.valuesMissing()) {
      return false;
    }
    boolean hasError = false;
    Map<PackageIdentifier, Package> packageMap = new HashMap<>();
    for (Entry<SkyKey, ValueOrException<BuildFileNotFoundException>> entry : packages.entrySet()) {
      try {
        packageMap.put(
            (PackageIdentifier) entry.getKey().argument(),
            ((PackageValue) entry.getValue().get()).getPackage());
      } catch (BuildFileNotFoundException e) {
        env.getListener().handle(Event.error(e.getMessage()));
        hasError = true;
      }
    }

    for (Label label : labels) {
      Package pkg = packageMap.get(label.getPackageIdentifier());
      if (pkg == null) {
        continue;
      }
      try {
        targets.add(pkg.getTarget(label.getName()));
      } catch (NoSuchTargetException e) {
        env.getListener().handle(Event.error(e.getMessage()));
        hasError = true;
      }
    }
    return hasError;
  }

  /**
   * Filters 'tests' (by mutation) according to the 'tags' attribute, specifically those that
   * match ALL of the tags in tagsAttribute.
   */
  private static void filterTests(Rule testSuite, ResolvedTargets.Builder<Target> tests) {
    List<String> tagsAttribute =
        NonconfigurableAttributeMapper.of(testSuite).get("tags", Type.STRING_LIST);
    // Split the tags list into positive and negative tags.
    Pair<Collection<String>, Collection<String>> tagLists =
        TestTargetUtils.sortTagsBySense(tagsAttribute);
    final Collection<String> positiveTags = tagLists.first;
    final Collection<String> negativeTags = tagLists.second;

    tests.filter(new Predicate<Target>() {
      @Override
      public boolean apply(@Nullable Target input) {
        Rule test = (Rule) input;
        AttributeMap nonConfigurableAttributes = NonconfigurableAttributeMapper.of(test);
        List<String> testTags =
            new ArrayList<>(nonConfigurableAttributes.get("tags", Type.STRING_LIST));
        testTags.add(nonConfigurableAttributes.get("size", Type.STRING));
        return includeTest(testTags, positiveTags, negativeTags);
      }
    });
  }

  /**
   * Decides whether to include a test in a test_suite or not.
   *
   * @param testTags Collection of all tags exhibited by a given test.
   * @param positiveTags Tags declared by the suite. A Test must match ALL of these.
   * @param negativeTags Tags declared by the suite. A Test must match NONE of these.
   * @return false if the test is to be removed.
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

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
