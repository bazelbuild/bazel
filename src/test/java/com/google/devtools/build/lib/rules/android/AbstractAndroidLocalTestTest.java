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
// limitations under the License.package com.google.devtools.build.lib.rules.android;
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.util.FileType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Abstract base class for tests of {@link BazelAndroidLocalTest}. These tests are for the
 * java related portions of the rule. Be sure to use writeFile() and overwriteFile() (instead of
 * scratch.writeFile() and scratch.overwriteFile()).
 */
@RunWith(JUnit4.class)
public abstract class AbstractAndroidLocalTestTest extends BuildViewTestCase {

  @Test
  public void testSimpleAndroidRobolectricConfiguredTarget() throws Exception {
    writeFile("java/test/BUILD",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'])");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    assertThat(target).isNotNull();
    checkMainClass(target, "com.google.testing.junit.runner.GoogleTestRunner");
  }

  @Test
  public void testOneVersionEnforcement() throws Exception {
    useConfiguration("--experimental_one_version_enforcement=error");
    writeFile("java/test/resource/BUILD",
        "android_local_test(name = 'dummyTest',",
        "                         srcs = ['test.java'],",
        "                         deps = [':dummyLibraryOne', ':dummyLibraryTwo'])",
        "",
        "android_library(name = 'dummyLibraryOne',",
        "                srcs = ['libraryOne.java'])",
        "",
        "android_library(name = 'dummyLibraryTwo',",
        "                srcs = ['libraryTwo.java'],",
        "                deps = [':dummyLibraryThree'])",
        "",
        "android_library(name = 'dummyLibraryThree',",
        "                srcs = ['libraryThree.java'])",
        "");

    ConfiguredTarget thingToTest = getConfiguredTarget("//java/test/resource:dummyTest");
    Action oneVersionAction =
        getGeneratingActionInOutputGroup(
            thingToTest,
            "java/test/resource/dummyTest-one-version.txt",
            OutputGroupProvider.HIDDEN_TOP_LEVEL);

    Iterable<Artifact> jartifacts =
        ImmutableList.copyOf(FileType.filter(oneVersionAction.getInputs(), JavaSemantics.JAR));
    assertThat(prettyArtifactNames(jartifacts))
        .containsExactly(
            "java/test/resource/dummyTest.jar",
            "java/test/resource/dummyTest_resources.jar",
            "third_party/java/junit/junit.jar",
            "third_party/java/android/android_sdk_linux/platforms/stable/android_blaze.jar",
            "third_party/java/robolectric/robolectric.jar",
            "java/com/google/thirdparty/robolectric/robolectric.jar",
            "java/test/resource/libdummyLibraryOne.jar",
            "java/test/resource/libdummyLibraryTwo.jar",
            "java/test/resource/libdummyLibraryThree.jar",
            "java/com/google/testing/junit/runner/librunner.jar")
        .inOrder();
  }

  @Test
  public void testCollectCodeCoverageWorks() throws Exception {
    writeFile("java/test/BUILD",
        "android_local_test(name = 'dummyTest',",
        "    srcs = [ 'test.java'])");

    useConfiguration("--collect_code_coverage");
    checkMainClass(getConfiguredTarget("//java/test:dummyTest"),
        "com.google.testing.coverage.JacocoCoverageRunner");
  }

  @Test
  public void testDataDependency() throws Exception {
    writeFile("java/test/BUILD",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    data = ['data.dat'])");

    writeFile("java/test/data.dat",
        "this is a dummy data file");

    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    Artifact data = getFileConfiguredTarget("//java/test:data.dat")
        .getArtifact();
    RunfilesProvider runfiles = target.getProvider(RunfilesProvider.class);

    assertThat(runfiles.getDataRunfiles().getAllArtifacts().toSet()).contains(data);
    assertThat(runfiles.getDefaultRunfiles().getAllArtifacts().toSet()).contains(data);
    assertThat(target).isNotNull();
  }

  @Test
  public void testNeverlinkRuntimeDepsExclusionReportsError() throws Exception {
    useConfiguration("--noexperimental_allow_runtime_deps_on_neverlink");
    checkError("java/test", "test",
        "neverlink dep //java/test:neverlink_lib not allowed in runtime deps",
        String.format("%s(name = 'test',", getRuleName()),
        "    srcs = ['test.java'],",
        "    runtime_deps = [':neverlink_lib'])",
        "android_library(name = 'neverlink_lib',",
        "                srcs = ['dummyNeverlink.java'],",
        "                neverlink = 1,)");
  }

  @Test
  public void testDisallowPrecompiledJars() throws Exception {
    checkError(
        "java/test",
        "dummyTest",
        // messages:
        "expected .java, .srcjar, .properties or .xmb",
        // build file:
        String.format("%s(name = 'dummyTest',", getRuleName()),
        "    srcs = ['test.java', ':jar'])",
        "filegroup(name = 'jar',",
        "    srcs = ['lib.jar'])");
  }

  private void checkMainClass(ConfiguredTarget target, String mainClass) throws Exception {
    TemplateExpansionAction action = (TemplateExpansionAction) getGeneratingAction(
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "dummyTest"));
    assertThat(action.getFileContents()).contains(mainClass);
  }

  @Test
  public void testResourcesFromRuntimeDepsAreIncluded() throws Exception {
    writeFile(
        "java/android/BUILD",
        "android_library(name = 'dummyLibraryOne',",
        "                exports_manifest = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/drawable/dummyResource1.png'],",
        "                srcs = ['libraryOne.java'])",
        "",
        "android_library(name = 'dummyLibraryTwo',",
        "                exports_manifest = 1,",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/drawable/dummyResource2.png'],",
        "                srcs = ['libraryTwo.java'])");
    final String libraryOne = "dummyLibraryOne";
    final String libraryTwo = "dummyLibraryTwo";

    useConfiguration("--noexperimental_android_include_library_resource_jars");

    checkForCorrectLibraries(
        "no-runtime", Arrays.asList(libraryOne), Collections.<String>emptyList());
    checkForCorrectLibraries(
        "no-runtime-2", Arrays.asList(libraryOne, libraryTwo), Collections.<String>emptyList());
    checkForCorrectLibraries(
        "only-runtime", Collections.<String>emptyList(), Arrays.asList(libraryOne));
    checkForCorrectLibraries(
        "only-runtime-2", Collections.<String>emptyList(), Arrays.asList(libraryOne, libraryTwo));
    checkForCorrectLibraries(
        "runtime-and-dep", Arrays.asList(libraryOne), Arrays.asList(libraryTwo));
    checkForCorrectLibraries(
        "runtime-and-dep-2", Arrays.asList(libraryTwo), Arrays.asList(libraryOne));
  }

  private String createDepArrayString(Collection<String> deps) {
    if (deps.isEmpty()) {
      return "";
    }
    ArrayList<String> list = new ArrayList<>();
    for (String dep : deps) {
      list.add(String.format("//java/android:%s", dep));
    }
    return "'" + Joiner.on("', '").join(list) + "'";
  }

  private void checkForCorrectLibraries(
      String name, Collection<String> deps, Collection<String> runtimeDeps) throws Exception {
    final String libraryFormat =
        "java/android/%s_processed_manifest/AndroidManifest.xml:" + "java/android/%s.aar";
    writeFile(
        String.format("javatests/android/%s/BUILD", name),
        "android_local_test(name = 'dummyTest',",
        "                         srcs = ['test.java'],",
        "                         runtime_deps = [" + createDepArrayString(runtimeDeps) + "],",
        "                         deps = [" + createDepArrayString(deps) + "])");
    ConfiguredTarget target =
        getConfiguredTarget(String.format("//javatests/android/%s:dummyTest", name));
    assertThat(target).isNotNull();
    RunfilesSupport support = target.getProvider(FilesToRunProvider.class).getRunfilesSupport();
    assertThat(support).isNotNull();
    Artifact deployJar =
        getFileConfiguredTarget(String.format("//javatests/android/%s:dummyTest_deploy.jar", name))
            .getArtifact();
    List<String> deployJarInputs =
        ActionsTestUtil.prettyArtifactNames(getGeneratingAction(deployJar).getInputs());

    LinkedHashSet<String> uniqueDeps = new LinkedHashSet<>();
    for (String dep : Iterables.concat(runtimeDeps, deps)) {
      uniqueDeps.add(String.format(libraryFormat, dep, dep));
      assertThat(deployJarInputs).contains("java/android/" + dep + "_resources.jar");
    }
    checkRuntimeSupportInputs(uniqueDeps, support);
  }

  protected abstract void checkRuntimeSupportInputs(
      LinkedHashSet<String> uniqueDeps, RunfilesSupport support) throws Exception;

  @Test
  public void testFeatureFlagsAttributeSetsSelectInDependency() throws Exception {
    useConfiguration("--experimental_dynamic_configs=on");
    writeFile(
        "java/com/foo/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        ")",
        "config_feature_flag(",
        "  name = 'flag2',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag2@on',",
        "  flag_values = {':flag2': 'on'},",
        ")",
        "android_library(",
        "  name = 'lib',",
        "  srcs = select({",
        "    ':flag1@on': ['Flag1On.java'],",
        "    '//conditions:default': ['Flag1Off.java'],",
        "  }) + select({",
        "    ':flag2@on': ['Flag2On.java'],",
        "    '//conditions:default': ['Flag2Off.java'],",
        "  }),",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  deps = [':lib'],",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  }",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/foo");
    List<String> inputs =
        actionsTestUtil()
            .prettyArtifactNames(actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)));

    assertThat(inputs).containsAllOf("java/com/foo/Flag1On.java", "java/com/foo/Flag2Off.java");
    assertThat(inputs).containsNoneOf("java/com/foo/Flag1Off.java", "java/com/foo/Flag2On.java");
  }

  @Test
  public void testFeatureFlagsAttributeSetsSelectInTest() throws Exception {
    useConfiguration("--experimental_dynamic_configs=on");
    writeFile(
        "java/com/foo/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        ")",
        "config_feature_flag(",
        "  name = 'flag2',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag2@on',",
        "  flag_values = {':flag2': 'on'},",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'] + select({",
        "    ':flag1@on': ['Flag1On.java'],",
        "    '//conditions:default': ['Flag1Off.java'],",
        "  }) + select({",
        "    ':flag2@on': ['Flag2On.java'],",
        "    '//conditions:default': ['Flag2Off.java'],",
        "  }),",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  }",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/foo");
    List<String> inputs =
        actionsTestUtil()
            .prettyArtifactNames(actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)));

    assertThat(inputs).containsAllOf("java/com/foo/Flag1On.java", "java/com/foo/Flag2Off.java");
    assertThat(inputs).containsNoneOf("java/com/foo/Flag1Off.java", "java/com/foo/Flag2On.java");
  }

  @Test
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagValueIsInvalid() throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration("--experimental_dynamic_configs=on");
    writeFile(
        "java/com/foo/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        ")",
        "android_library(",
        "  name = 'lib',",
        "  srcs = select({",
        "    ':flag1@on': ['Flag1On.java'],",
        "    '//conditions:default': ['Flag1Off.java'],",
        "  })",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  deps = [':lib'],",
        "  feature_flags = {",
        "    'flag1': 'invalid',",
        "  }",
        ")");
    assertThat(getConfiguredTarget("//java/com/foo")).isNull();
    assertContainsEvent(
        "in config_feature_flag rule //java/com/foo:flag1: "
            + "value must be one of [\"off\", \"on\"], but was \"invalid\"");
  }

  @Test
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagValueIsInvalidEvenIfNotUsed()
      throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration("--experimental_dynamic_configs=on");
    writeFile(
        "java/com/foo/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  feature_flags = {",
        "    'flag1': 'invalid',",
        "  }",
        ")");
    assertThat(getConfiguredTarget("//java/com/foo")).isNull();
    assertContainsEvent(
        "in config_feature_flag rule //java/com/foo:flag1: "
            + "value must be one of [\"off\", \"on\"], but was \"invalid\"");
  }

  @Test
  public void testFeatureFlagsAttributeSetsFeatureFlagProviderValues() throws Exception {
    useConfiguration("--experimental_dynamic_configs=on");
    writeFile(
        "java/com/foo/reader.bzl",
        "def _impl(ctx):",
        "  ctx.actions.write(",
        "      ctx.outputs.java,",
        "      '\\n'.join([",
        "          str(target.label) + ': ' + target[config_common.FeatureFlagInfo].value",
        "          for target in ctx.attr.flags]))",
        "  return struct(files=depset([ctx.outputs.java]))",
        "flag_reader = rule(",
        "  implementation=_impl,",
        "  attrs={'flags': attr.label_list(providers=[config_common.FeatureFlagInfo])},",
        "  outputs={'java': '%{name}.java'},",
        ")");
    writeFile(
        "java/com/foo/BUILD",
        "load('//java/com/foo:reader.bzl', 'flag_reader')",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_feature_flag(",
        "  name = 'flag2',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "flag_reader(",
        "  name = 'FooFlags',",
        "  flags = [':flag1', ':flag2'],",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  }",
        ")");
    Artifact flagList =
        actionsTestUtil().getFirstArtifactEndingWith(
            actionsTestUtil()
                .artifactClosureOf(getFilesToBuild(getConfiguredTarget("//java/com/foo"))),
            "/FooFlags.java");
    FileWriteAction action = (FileWriteAction) getGeneratingAction(flagList);
    assertThat(action.getFileContents())
        .isEqualTo("//java/com/foo:flag1: on\n//java/com/foo:flag2: off");
  }

  @Test
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagIsAliased()
      throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration("--experimental_dynamic_configs=on");
    writeFile(
        "java/com/foo/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "alias(",
        "  name = 'alias',",
        "  actual = 'flag1',",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  feature_flags = {",
        "    'alias': 'on',",
        "  }",
        ")");
    assertThat(getConfiguredTarget("//java/com/foo")).isNull();
    assertContainsEvent(String.format(
        "in feature_flags attribute of %s rule //java/com/foo:foo: "
            + "Feature flags must be named directly, not through aliases; "
            + "use '//java/com/foo:flag1', not '//java/com/foo:alias'", getRuleName()));
  }

  @Test
  public void testFeatureFlagPolicyMustBeVisibleToRuleToUseFeatureFlags() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    overwriteFile(
        "tools/whitelists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['//flag'])");
    writeFile(
        "flag/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['right', 'wrong'],",
        "    default_value = 'right',",
        "    visibility = ['//java/com/google/android/foo:__pkg__'],",
        ")");
    writeFile(
        "java/com/google/android/foo/BUILD",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        "  feature_flags = {",
        "    '//flag:flag': 'right',",
        "  }",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNull();
    assertContainsEvent(
        String.format("in feature_flags attribute of %s rule "
            + "//java/com/google/android/foo:foo: the feature_flags attribute is not available in "
            + "package 'java/com/google/android/foo'", getRuleName()));
  }

  @Test
  public void testFeatureFlagPolicyDoesNotBlockRuleIfInPolicy() throws Exception {
    overwriteFile(
        "tools/whitelists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['//flag', '//java/com/google/android/foo'])");
    writeFile(
        "flag/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['right', 'wrong'],",
        "    default_value = 'right',",
        "    visibility = ['//java/com/google/android/foo:__pkg__'],",
        ")");
    writeFile(
        "java/com/google/android/foo/BUILD",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        "  feature_flags = {",
        "    '//flag:flag': 'right',",
        "  }",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testFeatureFlagPolicyIsNotUsedIfFlagValuesNotUsed() throws Exception {
    overwriteFile(
        "tools/whitelists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['*super* busted package group'])");
    writeFile(
        "java/com/google/android/foo/BUILD",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNotNull();
    // the package_group is busted, so we would have failed to get this far if we depended on it
    assertNoEvents();
    // sanity check time: does this test actually test what we're testing for?
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//tools/whitelists/config_feature_flag:config_feature_flag"))
        .isNull();
    assertContainsEvent("*super* busted package group");
  }

  protected abstract String getRuleName();

  protected abstract void writeFile(String path, String... lines) throws Exception;

  protected abstract void overwriteFile(String path, String... lines) throws Exception;


}
