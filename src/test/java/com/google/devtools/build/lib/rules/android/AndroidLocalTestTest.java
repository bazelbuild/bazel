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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaPrimaryClassProvider;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for android_local_test. */
@RunWith(JUnit4.class)
public abstract class AndroidLocalTestTest extends AbstractAndroidLocalTestTestBase {

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  @Test
  public void testSimpleTestNotNull() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    assertThat(target).isNotNull();
  }

  @Test
  public void testResourceFilesZipCalledResourceFilesZip() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");

    Artifact resourcesZip =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_ZIP);
    assertThat(resourcesZip.getFilename()).isEqualTo("resource_files.zip");
  }

  @Test
  public void testManifestInRunfiles() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    NestedSet<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact manifest =
        ActionsTestUtil.getFirstArtifactEndingWith(
            runfilesArtifacts, "dummyTest_processed_manifest/AndroidManifest.xml");
    assertThat(manifest).isNotNull();
  }

  @Test
  public void testResourcesClassJarInRunfiles() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    NestedSet<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact resourceClassJar =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);
    assertThat(runfilesArtifacts.toList()).contains(resourceClassJar);
  }

  @Test
  public void testResourcesZipFileInRunfiles() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    NestedSet<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact resourcesZip =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_ZIP);
    assertThat(runfilesArtifacts.toList()).contains(resourcesZip);
  }

  @Test
  public void testCanHaveManifestNotNamedAndroidManifestXml() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps",
        "    manifest = 'NotAndroidManifest.xml')");
    assertNoEvents();
  }

  @Test
  public void testCustomPackage() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    custom_package = 'custom.pkg',",
        "    test_class = 'test',",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//a:dummyTest");
    Artifact resourcesClassJar =
        getImplicitOutputArtifact(target, AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);
    List<String> args = getGeneratingSpawnActionArgs(resourcesClassJar);
    MoreAsserts.assertContainsSublist(args, "--packageForR", "custom.pkg");
  }

  @Test
  public void testCustomPackageMissingAttribute() throws Exception {
    scratch.file(
        "wrong_package_name/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");

    boolean noCrashFlag = false;
    try {
      getConfiguredTarget("//wrong_package_name/test:dummyTest");
      noCrashFlag = true;
    } catch (AssertionError error) {
      assertThat(error).hasMessageThat().contains("'custom_package'");
    }
    assertThat(noCrashFlag).isFalse();
  }

  @Test
  public void testBinaryResources() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    useConfiguration("--experimental_android_local_test_binary_resources");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    NestedSet<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact resourceApk =
        ActionsTestUtil.getFirstArtifactEndingWith(runfilesArtifacts, "dummyTest.ap_");
    assertThat(resourceApk).isNotNull();
  }

  @Test
  public void testNoBinaryResources() throws Exception {
    useConfiguration("--noexperimental_android_local_test_binary_resources");
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps)");
    ConfiguredTarget target = getConfiguredTarget("//java/test:dummyTest");
    NestedSet<Artifact> runfilesArtifacts = collectRunfiles(target);
    Artifact resourceApk =
        ActionsTestUtil.getFirstArtifactEndingWith(runfilesArtifacts, "dummyTest.ap_");
    assertThat(resourceApk).isNull();
  }

  /**
   * Tests that the Java package can be correctly inferred from the path to a target, not just the
   * path to the corresponding BUILD file.
   */
  @Test
  public void testInferredJavaPackageFromPackageName() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "java-src/test",
            "test/java/foo/bar",
            "android_local_test(name ='test/java/foo/bar',",
            "  manifest = 'AndroidManifest.xml')");

    assertThat(target.getProvider(JavaPrimaryClassProvider.class).getPrimaryClass())
        .isEqualTo("foo.bar");
  }

  @Test
  public void testNocompressExtensions() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_binary(",
        "  name = 'r',",
        "  srcs = ['Foo.java'],",
        "  manifest = 'AndroidManifest.xml',",
        "  resource_files = ['res/raw/foo.apk'],",
        "  nocompress_extensions = ['.apk', '.so'],",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/r/android:r");
    ValidatedAndroidResources resource = getValidatedResources(binary);
    List<String> args = getGeneratingSpawnActionArgs(resource.getApk());
    Artifact inputManifest =
        getFirstArtifactEndingWith(
            getGeneratingSpawnAction(resource.getManifest()).getInputs(), "AndroidManifest.xml");
    Artifact finalUnsignedApk =
        getFirstArtifactEndingWith(
            binary.getProvider(FileProvider.class).getFilesToBuild(), "_unsigned.apk");
    Artifact compressedUnsignedApk =
        artifactByPath(
            actionsTestUtil().artifactClosureOf(finalUnsignedApk),
            "_unsigned.apk",
            "_unsigned.apk");
    assertContainsSublist(
        args,
        ImmutableList.of(
            "--primaryData", "java/r/android/res::" + inputManifest.getExecPathString()));
    assertThat(args).contains("--uncompressedExtensions");
    assertThat(args.get(args.indexOf("--uncompressedExtensions") + 1)).isEqualTo(".apk,.so");
    assertThat(getGeneratingSpawnActionArgs(compressedUnsignedApk))
        .containsAtLeast("--nocompress_suffixes", ".apk", ".so")
        .inOrder();
    assertThat(getGeneratingSpawnActionArgs(finalUnsignedApk))
        .containsAtLeast("--nocompress_suffixes", ".apk", ".so")
        .inOrder();
  }

  @Test
  public void testResourceConfigurationFilters() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "android_local_test(name = 'dummyTest',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps,",
        "    resource_configuration_filters = ['ar_XB'])");

    ConfiguredTarget binary = getConfiguredTarget("//java/test:dummyTest");
    final ImmutableList<ActionAnalysisMetadata> actions =
        ((RuleConfiguredTarget) binary).getActions();

    ActionAnalysisMetadata aaptAction = null;
    for (ActionAnalysisMetadata action : actions) {
      if (action.getMnemonic().equals("AndroidAapt2")) {
        aaptAction = action;
      }
    }
    assertThat(aaptAction).isNotNull();
    final List<String> aaptArguments = ((SpawnAction) aaptAction).getArguments();
    assertThat(aaptArguments).contains("--resourceConfigs");
    assertThat(aaptArguments).contains("ar_XB");
  }

  @Test
  public void featureFlagsSetByAndroidLocalTestAreInRequiredFragments() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct");
    scratch.overwriteFile(
        "tools/whitelists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['//java/com/google/android/foo'])");
    scratch.file(
        "java/com/google/android/foo/BUILD",
        "load('//java/bar:foo.bzl', 'extra_deps')",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "android_binary(",
        "  name = 'foo_under_test',",
        "  srcs = ['Test.java'],",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_local_test(",
        "    name = 'local_test',",
        "    srcs = ['test.java'],",
        "    deps = extra_deps,",
        "    feature_flags = {",
        "      'flag1': 'on',",
        "    },",
        "    resource_configuration_filters = ['ar_XB'])");

    ConfiguredTarget ct = getConfiguredTarget("//java/com/google/android/foo:local_test");
    assertThat(ct.getProvider(RequiredConfigFragmentsProvider.class).getRequiredConfigFragments())
        .contains("//java/com/google/android/foo:flag1");
  }

  @Override
  protected String getRuleName() {
    return "android_local_test";
  }

  @Override
  protected void writeFile(String path, String... lines) throws Exception {
    scratch.file(path, lines);
  }

  @Override
  protected void overwriteFile(String path, String... lines) throws Exception {
    scratch.overwriteFile(path, lines);
  }
}
