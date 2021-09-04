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
package com.google.devtools.build.lib.rules.test;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule object implementing "test_suite". */
public final class TestSuiteRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        // Technically, test_suite does not use TestConfiguration. But the tests it depends on
        // will always depend on TestConfiguration, so requiring it here simply acknowledges that
        // and prevents pruning by --trim_test_configuration.
        .requiresConfigurationFragments(TestConfiguration.class)
        .setMissingFragmentPolicy(TestConfiguration.class, MissingFragmentPolicy.IGNORE)
        .override(
            attr("testonly", BOOLEAN)
                .value(true)
                .nonconfigurable("policy decision: should be consistent across configurations"))
        /* <!-- #BLAZE_RULE(test_suite).ATTRIBUTE(tags) -->
        List of text tags such as "small" or "database" or "-flaky". Tags may be any valid string.
        <p>
          Tags which begin with a "-" character are considered negative tags. The
          preceding "-" character is not considered part of the tag, so a suite tag
          of "-small" matches a test's "small" size. All other tags are considered
          positive tags.
        </p>
        <p>
          Optionally, to make positive tags more explicit, tags may also begin with the
          "+" character, which will not be evaluated as part of the text of the tag. It
          merely makes the positive and negative distinction easier to read.
        </p>
        <p>
          Only test rules that match <b>all</b> of the positive tags and <b>none</b> of the negative
          tags will be included in the test suite. Note that this does not mean that error checking
          for dependencies on tests that are filtered out is skipped; the dependencies on skipped
          tests still need to be legal (e.g. not blocked by visibility constraints).
        </p>
        <p>
          The <code>manual</code> tag keyword is treated differently than the above by the
          "test_suite expansion" performed by the <code>blaze test</code> command on invocations
          involving wildcard
          <a href="https://docs.bazel.build/versions/main/guide.html#specifying-targets-to-build">target patterns</a>.
          There, <code>test_suite</code> targets tagged "manual" are filtered out (and thus not
          expanded). This behavior is consistent with how <code>blaze build</code> and
          <code>blaze test</code> handle wildcard target patterns in general.
        </p>
        <p>
          Note that a test's <code>size</code> is considered a tag for the purpose of filtering.
        </p>
        <p>
          If you need a <code>test_suite</code> that contains tests with mutually exclusive tags
          (e.g. all small and medium tests), you'll have to create three <code>test_suite</code>
          rules: one for all small tests, one for all medium tests, and one that includes the
          previous two.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */

        /* <!-- #BLAZE_RULE(test_suite).ATTRIBUTE(tests) -->
        A list of test suites and test targets of any language.
        <p>
          Any <code>*_test</code> is accepted here, independent of the language. No
          <code>*_binary</code> targets are accepted however, even if they happen to run a test.
          Filtering by the specified <code>tags</code> is only done for tests listed directly in
          this attribute. If this attribute contains <code>test_suite</code>s, the tests inside
          those will not be filtered by this <code>test_suite</code> (they are considered to be
          filtered already).
        </p>
        <p>
          If the <code>tests</code> attribute is unspecified or empty, the rule will default to
          including all test rules in the current BUILD file that are not tagged as
          <code>manual</code>. These rules are still subject to <code>tag</code> filtering.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("tests", LABEL_LIST)
                .orderIndependent()
                .allowedFileTypes()
                .nonconfigurable("policy decision: should be consistent across configurations"))
        // This magic attribute contains all *test rules in the package, iff
        // tests=[].
        .add(
            attr("$implicit_tests", LABEL_LIST)
                .orderIndependent()
                .nonconfigurable("Accessed in TestTargetUtils without config context"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("test_suite")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(TestSuite.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = test_suite, FAMILY = General)[GENERIC_RULE] -->

<p>
A <code>test_suite</code> defines a set of tests that are considered "useful" to humans. This
allows projects to define sets of tests, such as "tests you must run before checkin", "our
project's stress tests" or "all small tests." The <code>blaze test</code> command respects this sort
of organization: For an invocation like <code>blaze test //some/test:suite</code>, Blaze first
enumerates all test targets transitively included by the <code>//some/test:suite</code> target (we
call this "test_suite expansion"), then Blaze builds and tests those targets.
</p>

<h4 id="test_suite_examples">Examples</h4>

<p>A test suite to run all of the small tests in the current package.</p>
<pre class="code">
test_suite(
    name = "small_tests",
    tags = ["small"],
)
</pre>

<p>A test suite that runs a specified set of tests:</p>

<pre class="code">
test_suite(
    name = "smoke_tests",
    tests = [
        "system_unittest",
        "public_api_unittest",
    ],
)
</pre>

<p>A test suite to run all tests in the current package which are not flaky.</p>
<pre class="code">
test_suite(
    name = "non_flaky_test",
    tags = ["-flaky"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
