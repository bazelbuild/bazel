// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for rule ConfiguredTargets.
 */
@RunWith(JUnit4.class)
public final class RuleConfiguredTargetTest extends BuildViewTestCase {

  private ConfiguredTarget configure(String ruleLabel) throws Exception {
    return getConfiguredTarget(ruleLabel);
  }

  @Test
  public void smokeNonexistentFailure() throws Exception {
    scratch.file("a/BUILD", "");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("target 'a' not declared in package 'a'");
  }

  @Test
  public void testFeatureEnabledOnCommandLine() throws Exception {
    useConfiguration("--features=feature");
    scratch.file("a/BUILD",
        "cc_library(name = 'a')");
    ImmutableSet<String> features = getRuleContext(configure("//a")).getFeatures();
    assertThat(features).contains("feature");
    assertThat(features).doesNotContain("other");
  }

  @Test
  public void testTargetIgnoresHostFeatures() throws Exception {
    useConfiguration("--features=feature", "--host_features=host_feature");
    scratch.file("a/BUILD", "cc_library(name = 'a')");
    ImmutableSet<String> features = getRuleContext(configure("//a")).getFeatures();
    assertThat(features).contains("feature");
    assertThat(features).doesNotContain("host_feature");
  }

  @Test
  public void testHostFeatures() throws Exception {
    useConfiguration("--features=feature", "--host_features=host_feature");
    scratch.file("a/BUILD", "cc_library(name = 'a')");
    ImmutableSet<String> features =
        getRuleContext(getConfiguredTarget("//a", getExecConfiguration())).getFeatures();
    assertThat(features).contains("host_feature");
    assertThat(features).doesNotContain("feature");
  }

  @Test
  public void testHostFeaturesIncompatibleDisabled() throws Exception {
    useConfiguration(
        "--features=feature",
        "--host_features=host_feature",
        "--incompatible_use_host_features=false");
    scratch.file("a/BUILD", "cc_library(name = 'a')");
    ImmutableSet<String> features =
        getRuleContext(getConfiguredTarget("//a", getExecConfiguration())).getFeatures();
    assertThat(features).contains("feature");
    assertThat(features).doesNotContain("host_feature");
  }

  @Test
  public void testFeatureDisabledOnCommandLine() throws Exception {
    useConfiguration("--features=-feature");
    scratch.file("a/BUILD", "cc_library(name = 'a')");
    ImmutableSet<String> disabledFeatures = getRuleContext(configure("//a")).getDisabledFeatures();
    assertThat(disabledFeatures).contains("feature");
    assertThat(disabledFeatures).doesNotContain("other");
  }

  @Test
  public void testFeatureEnabledInPackage() throws Exception {
    scratch.file("a/BUILD", "package(features = ['feature'])", "cc_library(name = 'a')");
    ImmutableSet<String> features = getRuleContext(configure("//a")).getFeatures();
    assertThat(features).contains("feature");
    assertThat(features).doesNotContain("other");
  }

  @Test
  public void testFeatureDisableddInPackage() throws Exception {
    scratch.file("a/BUILD", "package(features = ['-feature'])", "cc_library(name = 'a')");
    ImmutableSet<String> disabledFeatures = getRuleContext(configure("//a")).getDisabledFeatures();
    assertThat(disabledFeatures).contains("feature");
    assertThat(disabledFeatures).doesNotContain("other");
  }

  @Test
  public void testFeatureEnabledInRule() throws Exception {
    scratch.file("a/BUILD",
        "cc_library(name = 'a', features = ['feature'])");
    ImmutableSet<String> features = getRuleContext(configure("//a")).getFeatures();
    assertThat(features).contains("feature");
    assertThat(features).doesNotContain("other");
  }

  @Test
  public void testFeatureDisabledInRule() throws Exception {
    scratch.file("a/BUILD", "cc_library(name = 'a', features = ['-feature'])");
    ImmutableSet<String> disabledFeatures = getRuleContext(configure("//a")).getDisabledFeatures();
    assertThat(disabledFeatures).contains("feature");
    assertThat(disabledFeatures).doesNotContain("other");
  }

  @Test
  public void testFeaturesInPackageOverrideFeaturesFromCommandLine() throws Exception {
    useConfiguration("--features=feature");
    scratch.file("a/BUILD", "package(features = ['-feature'])", "cc_library(name = 'a')");
    RuleContext ruleContext = getRuleContext(configure("//a"));
    ImmutableSet<String> features = ruleContext.getFeatures();
    ImmutableSet<String> disabledFeatures = ruleContext.getDisabledFeatures();
    assertThat(features).doesNotContain("feature");
    assertThat(disabledFeatures).contains("feature");
  }

  @Test
  public void testFeaturesInRuleOverrideFeaturesFromCommandLine() throws Exception {
    useConfiguration("--features=feature");
    scratch.file("a/BUILD", "cc_library(name = 'a', features = ['-feature'])");
    RuleContext ruleContext = getRuleContext(configure("//a"));
    ImmutableSet<String> features = ruleContext.getFeatures();
    ImmutableSet<String> disabledFeatures = ruleContext.getDisabledFeatures();
    assertThat(features).doesNotContain("feature");
    assertThat(disabledFeatures).contains("feature");
  }

  @Test
  public void testFeaturesInRuleOverrideFeaturesFromPackage() throws Exception {
    scratch.file("a/BUILD",
        "package(features = ['a', '-b', 'c'])",
        "cc_library(name = 'a', features = ['b', '-c', 'd'])");
    RuleContext ruleContext = getRuleContext(configure("//a"));
    ImmutableSet<String> features = ruleContext.getFeatures();
    ImmutableSet<String> disabledFeatures = ruleContext.getDisabledFeatures();
    assertThat(features).containsAtLeast("a", "b", "d");
    assertThat(disabledFeatures).contains("c");
  }

  @Test
  public void testFeaturesDisabledFromCommandLineOverrideAll() throws Exception {
    useConfiguration("--features=-package_feature", "--features=-rule_feature");
    scratch.file(
        "a/BUILD",
        "package(features = ['package_feature'])",
        "cc_library(name = 'a', features = ['rule_feature'])");
    RuleContext ruleContext = getRuleContext(configure("//a"));
    ImmutableSet<String> features = ruleContext.getFeatures();
    ImmutableSet<String> disabledFeatures = ruleContext.getDisabledFeatures();
    assertThat(features).doesNotContain("package_feature");
    assertThat(features).doesNotContain("rule_feature");
    assertThat(disabledFeatures).contains("package_feature");
    assertThat(disabledFeatures).contains("rule_feature");
  }

  @Test
  public void testExperimentalDependenciesOnThirdPartyExperimentalAllowed() throws Exception {
    scratch.file(
        "third_party/experimental/p1/BUILD",
        "licenses(['unencumbered'])",
        "exports_files(['p1.cc'])",
        "cc_library(name = 'p1')");
    scratch.file(
        "experimental/p2/BUILD",
        "exports_files(['p2.cc'])",
        "cc_library(name = 'p2', deps=['//third_party/experimental/p1:p1'])");

    getConfiguredTarget("//experimental/p2:p2"); // No errors.
  }

  @Test
  public void testThirdPartyExperimentalDependenciesOnExperimentalAllowed() throws Exception {
    scratch.file("experimental/p1/BUILD", "exports_files(['p1.cc'])", "cc_library(name = 'p1')");
    scratch.file(
        "third_party/experimental/p2/BUILD",
        "licenses(['unencumbered'])",
        "exports_files(['p2.cc'])",
        "cc_library(name = 'p2', deps=['//experimental/p1:p1'])");

    getConfiguredTarget("//third_party/experimental/p2:p2"); // No errors.
  }

  @Test
  public void testDependencyOnTestOnlyAllowed() throws Exception {
    scratch.file("testonly/BUILD",
        "cc_library(name = 'testutil',",
        "           srcs = ['testutil.cc'],",
        "           testonly = 1)");

    scratch.file("util/BUILD",
        "cc_library(name = 'util',",
        "           srcs = ['util.cc'])");

    scratch.file("cc/common/BUILD",
        // testonly=1 -> testonly=1
        "cc_library(name = 'lib1',",
        "           srcs = ['foo1.cc'],",
        "           deps = ['//testonly:testutil'],",
        "           testonly = 1)",
        // testonly=0 -> testonly=0
        "cc_library(name = 'lib2',",
        "           srcs = ['foo2.cc'],",
        "           deps = ['//util'],",
        "           testonly = 0)",
        // testonly=1 -> testonly=0
        "cc_library(name = 'lib3',",
        "           srcs = ['foo3.cc'],",
        "           deps = [':lib2'],",
        "           testonly = 1)");
    getConfiguredTarget("//cc/common:lib1"); // No errors.
    getConfiguredTarget("//cc/common:lib2"); // No errors.
    getConfiguredTarget("//cc/common:lib3"); // No errors.
  }

  @Test
  public void testDependsOnTestOnlyDisallowed() throws Exception {
    scratch.file("testonly/BUILD",
        "cc_library(name = 'testutil',",
        "           srcs = ['testutil.cc'],",
        "           testonly = 1)");
    checkError("cc/error", "cclib",
        // error:
        "non-test target '//cc/error:cclib' depends on testonly target '//testonly:testutil' and "
        + "doesn't have testonly attribute set",
        // build file: testonly=0 -> testonly=1
        "cc_library(name = 'cclib',",
        "           srcs  = ['foo.cc'],",
        "           deps = ['//testonly:testutil'],",
        "           testonly = 0)");
  }

  @Test
  public void testDependsOnTestOnlyOutputFileDisallowed() throws Exception {
    useConfiguration("--incompatible_check_testonly_for_output_files");
    scratch.file(
        "testonly/BUILD",
        "genrule(name = 'testutil',",
        "        outs = ['testutil.cc'],",
        "        cmd = 'touch testutil.cc',",
        "        srcs = [],",
        "        testonly = 1)");
    checkError(
        "cc/error",
        "cclib",
        // error:
        "non-test target '//cc/error:cclib' depends on the output file target"
            + " '//testonly:testutil.cc' of a testonly rule //testonly:testutil and doesn't have"
            + " testonly attribute set",
        // build file: testonly=0 -> testonly=1
        "cc_library(name = 'cclib',",
        "           srcs  = ['//testonly:testutil.cc'],",
        "           testonly = 0)");
  }

  @Test
  public void testDependenceOnDeprecatedRule() throws Exception {
    scratch.file("p/BUILD",
                "cc_library(name='p', deps=['//q'])");
    scratch.file("q/BUILD",
                "cc_library(name='q', deprecation='Obsolete!')");

    reporter.removeHandler(failFastHandler); // expect errors
    ConfiguredTarget p = getConfiguredTarget("//p");
    assertThat(view.hasErrors(p)).isFalse();
    assertContainsEvent("target '//p:p' depends on deprecated target '//q:q':"
                        + " Obsolete!");
    assertThat(eventCollector.count()).isEqualTo(1);
  }

  @Test
  public void testDependenceOnDeprecatedRuleEmptyExplanation() throws Exception {
    scratch.file("p/BUILD",
                "cc_library(name='p', deps=['//q'])");
    scratch.file("q/BUILD",
                "cc_library(name='q', deprecation='')");  // explicitly specified; still counts!

    reporter.removeHandler(failFastHandler); // expect errors
    ConfiguredTarget p = getConfiguredTarget("//p");
    assertThat(view.hasErrors(p)).isFalse();
    assertContainsEvent("target '//p:p' depends on deprecated target '//q:q'");
    assertThat(eventCollector.count()).isEqualTo(1);
  }

  @Test
  public void testNoWarningWhenDeprecatedDependsOnDeprecatedRule() throws Exception {
    scratch.file("foo/BUILD",
        "java_library(name='foo', srcs=['foo.java'], deps=['//bar:bar'])");
    scratch.file("bar/BUILD",
        "java_library(name='bar', srcs=['bar.java'], deps=['//baz:baz'], deprecation='BAR')");
    scratch.file("baz/BUILD",
        "java_library(name='baz', srcs=['baz.java'], deprecation='BAZ')");

    reporter.removeHandler(failFastHandler); // expect errors
    getConfiguredTarget("//foo");
    assertContainsEvent("target '//foo:foo' depends on deprecated "
        + "target '//bar:bar': BAR");
    assertDoesNotContainEvent("target '//bar:bar' depends on deprecated "
        + "target '//baz:baz': BAZ");
    assertThat(eventCollector.count()).isEqualTo(1);
  }

  @Test
  public void testAttributeErrorContainsLocationOfRule() throws Exception {
    Event e =
        checkError(
            "x",
            "x",
            // error:
            getErrorNonExistingTarget("srcs", "java_library", "//x:x", "//x:a.cc"),
            // build file:
            "# blank line",
            "java_library(name = 'x',",
            "           srcs = ['a.cc'])");
    assertThat(e.getLocation().toString()).isEqualTo("/workspace/x/BUILD:2:13");
  }

  @Test
  public void testJavatestsIsTestonly() throws Exception {
    scratch.file("java/x/BUILD",
                "java_library(name='x', exports=['//javatests/y'])");
    scratch.file("javatests/y/BUILD",
                "java_library(name='y')");
    reporter.removeHandler(failFastHandler); // expect warning
    ConfiguredTarget target = getConfiguredTarget("//java/x");
    assertContainsEvent("non-test target '//java/x:x' depends on testonly target"
        + " '//javatests/y:y' and doesn't have testonly attribute set");
    assertThat(view.hasErrors(target)).isTrue();
  }

  @Test
  public void testDependenceOfJavaProductionCodeOnTestPackageGroups() throws Exception {
    scratch.file("java/banana/BUILD",
        "java_library(name='banana',",
        "             visibility=['//javatests/plantain:chips'])");
    scratch.file("javatests/plantain/BUILD",
        "package_group(name='chips',",
        "              packages=['//javatests/plantain'])");

    getConfiguredTarget("//java/banana");
    assertNoEvents();
  }

  @Test
  public void testUnexpectedSourceFileInDeps() throws Exception {
    scratch.file("x/y.java", "foo");
    checkError("x", "x", getErrorMsgMisplacedFiles(
        "deps", "java_library", "//x:x", "//x:y.java"),
        "java_library(name='x', srcs=['x.java'], deps=['y.java'])");
  }

  @Test
  public void testUnexpectedButExistingSourceFileDependency() throws Exception {
    scratch.file("x/y.java");
    checkError("x", "x", getErrorMsgMisplacedFiles(
        "deps", "java_library", "//x:x", "//x:y.java"),
        "java_library(name='x', srcs=['x.java'], deps=['y.java'])");
  }

  @Test
  public void testGetArtifactForImplicitOutput() throws Exception {
    scratch.file("java/x/BUILD",
                "java_binary(name='x', srcs=['x.java'])");

    ConfiguredTarget javaBinary = getConfiguredTarget("//java/x:x");
    Artifact classJarArtifact = getFileConfiguredTarget("//java/x:x.jar").getArtifact();
    // Checks if the deploy jar is generated
    getFileConfiguredTarget("//java/x:x_deploy.jar").getArtifact();

    assertThat(getOutputGroup(javaBinary, OutputGroupInfo.FILES_TO_COMPILE).toList())
        .containsExactly(classJarArtifact);
  }

  @Test
  public void testSelfEdgeInRule() throws Exception {
    scratch.file("x/BUILD",

        "genrule(name='x', srcs=['x'], outs=['out'], cmd=':')");
    reporter.removeHandler(failFastHandler); // expect errors
    getConfiguredTarget("//x");
    assertContainsSelfEdgeEvent("//x:x");
  }

  @Test
  public void testNegativeShardCount() throws Exception {
    checkError("foo", "bar", "Must not be negative.",
        "sh_test(name='bar', srcs=['mockingbird.sh'], shard_count=-1)");
  }

  @Test
  public void testExcessiveShardCount() throws Exception {
    checkError("foo", "bar", "indicative of poor test organization",
        "sh_test(name='bar', srcs=['mockingbird.sh'], shard_count=51)");
  }

  @Test
  public void testNonexistingTargetErrorMsg() throws Exception {
    checkError("foo", "foo", getErrorNonExistingTarget(
        "deps", "cc_binary", "//foo:foo", "//foo:nonesuch"),
        "cc_binary(name = 'foo',",
        "srcs = ['foo.cc'],",
        "deps = [':nonesuch'])");
  }

  @Test
  public void testRulesDontProvideRequiredFragmentsByDefault() throws Exception {
    scratch.file(
        "a/BUILD",
        "config_setting(name = 'config', values = {'start_end_lib': '1'})",
        "py_library(name = 'pylib', srcs = ['pylib.py'])",
        "cc_library(name = 'a', srcs = ['A.cc'], data = [':pylib'])");
    assertThat(getConfiguredTarget("//a:a").getProvider(RequiredConfigFragmentsProvider.class))
        .isNull();
    assertThat(getConfiguredTarget("//a:config").getProvider(RequiredConfigFragmentsProvider.class))
        .isNull();
  }

  @Test
  public void findArtifactByOutputLabel_twoOutputsWithSameBasename() throws Exception {
    scratch.file(
        "foo/BUILD", "genrule(name = 'gen', outs = ['sub/out', 'out'], cmd = 'touch $(OUTS)')");
    RuleConfiguredTarget foo = (RuleConfiguredTarget) getConfiguredTarget("//foo:gen");
    assertThat(
            foo.findArtifactByOutputLabel(Label.parseCanonical("//foo:sub/out"))
                .getRepositoryRelativePath()
                .getPathString())
        .isEqualTo("foo/sub/out");
    assertThat(
            foo.findArtifactByOutputLabel(Label.parseCanonical("//foo:out"))
                .getRepositoryRelativePath()
                .getPathString())
        .isEqualTo("foo/out");
  }
}
