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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Abstract base class for tests of {@link BazelAndroidLocalTest}. These tests are for the java
 * related portions of the rule. Be sure to use writeFile() and overwriteFile() (instead of
 * scratch.writeFile() and scratch.overwriteFile()).
 */
@RunWith(JUnit4.class)
public abstract class AbstractAndroidLocalTestTestBase extends AndroidBuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    writeFile("java/bar/BUILD", "");
    writeFile("java/bar/foo.bzl", "extra_deps = []");
  }

  @Test
  public void testSimpleAndroidRobolectricConfiguredTarget() throws Exception {
    writeFile(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    assertThat(target).isNotNull();
    checkMainClass(target, "dummyTest", false);
  }

  @Test
  public void testDataDependency() throws Exception {
    writeFile(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps,",
        "    data = ['data.dat'])");

    writeFile("java/test/data.dat", "this is a dummy data file");

    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    Artifact data = getFileConfiguredTarget("//java/test:data.dat").getArtifact();
    RunfilesProvider runfiles = target.getProvider(RunfilesProvider.class);

    assertThat(runfiles.getDataRunfiles().getAllArtifacts().toSet()).contains(data);
    assertThat(runfiles.getDefaultRunfiles().getAllArtifacts().toSet()).contains(data);
    assertThat(target).isNotNull();
  }

  @Test
  public void testNeverlinkRuntimeDepsExclusionReportsError() throws Exception {
    useConfiguration("--noexperimental_allow_runtime_deps_on_neverlink");
    checkError(
        "java/test",
        "test",
        "neverlink dep //java/test:neverlink_lib not allowed in runtime deps",
        String.format("%s(name = 'test',", getRuleName()),
        "    srcs = ['test.java'],",
        "    runtime_deps = [':neverlink_lib'])",
        "android_library(name = 'neverlink_lib',",
        "                srcs = ['dummyNeverlink.java'],",
        "                neverlink = 1,)");
  }

  @Test
  public void testFeatureFlagsAttributeSetsSelectInDependency() throws Exception {
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
    writeFile(
        "java/com/foo/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        "  transitive_configs = [':flag1'],",
        ")",
        "config_feature_flag(",
        "  name = 'flag2',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag2@on',",
        "  flag_values = {':flag2': 'on'},",
        "  transitive_configs = [':flag2'],",
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
        "  transitive_configs = [':flag1', ':flag2'],",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  deps = [':lib'] + extra_deps,",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        "  transitive_configs = [':flag1', ':flag2'],",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/foo");
    List<String> inputs =
        ActionsTestUtil.prettyArtifactNames(
            actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)));

    assertThat(inputs).containsAtLeast("java/com/foo/Flag1On.java", "java/com/foo/Flag2Off.java");
    assertThat(inputs).containsNoneOf("java/com/foo/Flag1Off.java", "java/com/foo/Flag2On.java");
  }

  @Test
  public void testFeatureFlagsAttributeSetsSelectInTest() throws Exception {
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
    writeFile(
        "java/com/foo/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        "  transitive_configs = [':flag1'],",
        ")",
        "config_feature_flag(",
        "  name = 'flag2',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag2@on',",
        "  flag_values = {':flag2': 'on'},",
        "  transitive_configs = [':flag2'],",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  deps = extra_deps,",
        "  srcs = ['Test.java'] + select({",
        "    ':flag1@on': ['Flag1On.java'],",
        "    '//conditions:default': ['Flag1Off.java'],",
        "  }) + select({",
        "    ':flag2@on': ['Flag2On.java'],",
        "    '//conditions:default': ['Flag2Off.java'],",
        "  }),",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        "  transitive_configs = [':flag1', ':flag2'],",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/foo");
    List<String> inputs =
        ActionsTestUtil.prettyArtifactNames(
            actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)));

    assertThat(inputs).containsAtLeast("java/com/foo/Flag1On.java", "java/com/foo/Flag2Off.java");
    assertThat(inputs).containsNoneOf("java/com/foo/Flag1Off.java", "java/com/foo/Flag2On.java");
  }

  @Test
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagValueIsInvalid() throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
    writeFile(
        "java/com/foo/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        "  transitive_configs = [':flag1'],",
        ")",
        "android_library(",
        "  name = 'lib',",
        "  srcs = select({",
        "    ':flag1@on': ['Flag1On.java'],",
        "    '//conditions:default': ['Flag1Off.java'],",
        "  }),",
        "  transitive_configs = [':flag1'],",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  deps = [':lib',] + extra_deps,",
        "  feature_flags = {",
        "    'flag1': 'invalid',",
        "  },",
        "  transitive_configs = [':flag1'],",
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
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
    writeFile(
        "java/com/foo/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
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
        "  deps = extra_deps,",
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
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
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
        "load('//java/bar:foo.bzl', 'extra_deps')",
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
        "  transitive_configs = [':flag1', ':flag2'],",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        "  deps = extra_deps,",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        "  transitive_configs = [':flag1', ':flag2'],",
        ")");
    Artifact flagList =
        ActionsTestUtil.getFirstArtifactEndingWith(
            actionsTestUtil()
                .artifactClosureOf(getFilesToBuild(getConfiguredTarget("//java/com/foo"))),
            "/FooFlags.java");
    FileWriteAction action = (FileWriteAction) getGeneratingAction(flagList);
    assertThat(action.getFileContents())
        .isEqualTo("//java/com/foo:flag1: on\n//java/com/foo:flag2: off");
  }

  @Test
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagIsAliased() throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
    writeFile(
        "java/com/foo/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "alias(",
        "  name = 'alias',",
        "  actual = 'flag1',",
        "  transitive_configs = [':flag1'],",
        ")",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java'],",
        "  deps = extra_deps,",
        "  feature_flags = {",
        "    'alias': 'on',",
        "  },",
        "  transitive_configs = [':flag1'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/foo")).isNull();
    assertContainsEvent(
        String.format(
            "in feature_flags attribute of %s rule //java/com/foo:foo: "
                + "Feature flags must be named directly, not through aliases; "
                + "use '//java/com/foo:flag1', not '//java/com/foo:alias'",
            getRuleName()));
  }

  @Test
  public void testFeatureFlagPolicyMustBeVisibleToRuleToUseFeatureFlags() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    overwriteFile(
        "tools/allowlists/config_feature_flag/BUILD",
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
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        "  deps = extra_deps,",
        "  feature_flags = {",
        "    '//flag:flag': 'right',",
        "  },",
        "  transitive_configs = ['//flag:flag'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNull();
    assertContainsEvent(
        String.format(
            "in feature_flags attribute of %s rule //java/com/google/android/foo:foo: the"
                + " feature_flags attribute is not available in package"
                + " 'java/com/google/android/foo'",
            getRuleName()));
  }

  @Test
  public void testFeatureFlagPolicyDoesNotBlockRuleIfInPolicy() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    overwriteFile(
        "tools/allowlists/config_feature_flag/BUILD",
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
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(",
        "  name = 'foo',",
        "  srcs = ['Test.java', ':FooFlags.java'],",
        "  deps = extra_deps,",
        "  feature_flags = {",
        "    '//flag:flag': 'right',",
        "  },",
        "  transitive_configs = ['//flag:flag'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testFeatureFlagPolicyIsNotUsedIfFlagValuesNotUsed() throws Exception {
    overwriteFile(
        "tools/allowlists/config_feature_flag/BUILD",
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
    // NOTE: someone should verify that this test actually does what we think it does.
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//tools/allowlists/config_feature_flag:config_feature_flag"))
        .isNull();
    assertContainsEvent("*super* busted package group");
  }

  @Test
  public void androidManifestMergerOrderAlphabeticalByConfiguration_MergeesSortedByPathInBinOrGen()
      throws Exception {
    useConfiguration(
        "--fat_apk_cpu=k8", "--android_manifest_merger_order=alphabetical_by_configuration");
    scratch.overwriteFile(
        "java/android/BUILD",
        "android_library(",
        "    name = 'core',",
        "    manifest = 'core/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['core/res/values/strings.xml'],",
        ")",
        "android_library(",
        "    name = 'utility',",
        "    manifest = 'utility/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['utility/res/values/values.xml'],",
        "    deps = ['//java/common:common'],",
        ")");
    scratch.file(
        "java/binary/BUILD",
        "android_binary(",
        "    name = 'application',",
        "    srcs = ['App.java'],",
        "    manifest = 'app/AndroidManifest.xml',",
        "    deps = [':library'],",
        ")",
        "android_library(",
        "    name = 'library',",
        "    manifest = 'library/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    deps = ['//java/common:theme', '//java/android:utility'],",
        ")");
    scratch.file(
        "java/test/BUILD",
        "android_local_test(",
        "    name = 'test',",
        "    srcs = ['Test.java'],",
        "    manifest = 'testing/AndroidManifest.xml',",
        "    deps = ['//java/binary:application', '//java/binary:library'],",
        ")");
    scratch.file(
        "java/common/BUILD",
        "android_library(",
        "    name = 'common',",
        "    manifest = 'common/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['common/res/values/common.xml'],",
        "    deps = ['//java/android:core'],",
        ")",
        "android_library(",
        "    name = 'theme',",
        "    manifest = 'theme/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['theme/res/values/values.xml'],",
        ")");

    ConfiguredTarget test = getConfiguredTarget("//java/test:test");
    assertNoEvents();
    assertThat(test).isNotNull();

    // each entry appears twice because the application duplicates them all into another config;
    // this is partially testing that doing so does not cause a crash because of matching
    // root-relative paths.
    assertThat(getLocalTestMergeeManifests(test).values())
        .containsExactly(
            "//java/android:core",
            "//java/android:core",
            "//java/android:utility",
            "//java/android:utility",
            "//java/binary:library",
            "//java/binary:library",
            "//java/common:common",
            "//java/common:common",
            "//java/common:theme",
            "//java/common:theme")
        .inOrder();
  }

  @Test
  public void testDeployJar() throws Exception {
    writeFile(
        "java/com/google/android/foo/BUILD",
        "android_local_test(name = 'test',",
        "                srcs =['test.java'],",
        "                )",
        "android_library(name = 'lib',",
        "                data = [':test_deploy.jar'],",
        "                srcs =['lib.java'],",
        "                )");

    Action deployJarAction =
        getGeneratingAction(
            getFileConfiguredTarget("//java/com/google/android/foo:test_deploy.jar").getArtifact());
    List<String> inputs = ActionsTestUtil.baseArtifactNames(deployJarAction.getInputs());

    assertThat(inputs).containsAtLeast("test_resources.jar", "test.jar");
  }

  public abstract void checkMainClass(
      ConfiguredTarget target, String targetName, boolean coverageEnabled) throws Exception;

  protected abstract String getRuleName();

  protected abstract void writeFile(String path, String... lines) throws Exception;

  protected abstract void overwriteFile(String path, String... lines) throws Exception;
}
