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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getArtifactsEndingWith;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.hasInput;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getJavacArguments;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getProcessorpath;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Ascii;
import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.FileTarget;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import com.google.devtools.build.lib.rules.android.deployinfo.AndroidDeployInfoOuterClass.AndroidDeployInfo;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link com.google.devtools.build.lib.rules.android.AndroidBinary}. */
@RunWith(JUnit4.class)
public class AndroidBinaryTest extends AndroidBuildViewTestCase {

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  @Before
  public void setup() throws Exception {
    scratch.file(
        "java/android/BUILD",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "              )");
    scratch.file(
        "java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file("java/android/A.java", "package android; public class A {};");
    setBuildLanguageOptions("--experimental_google_legacy_api");
  }

  @Test
  public void testAndroidSplitTransitionWithInvalidCpu() throws Exception {
    scratch.file(
        "test/starlark/my_rule.bzl",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'deps': attr.label_list(cfg = android_common.multi_cpu_configuration),",
        "    'dep':  attr.label(cfg = android_common.multi_cpu_configuration),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', deps = [':main'], dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");
    BazelMockAndroidSupport.setupNdk(mockToolsConfig);

    // --android_cpu with --android_crosstool_top also triggers the split transition.
    useConfiguration(
        "--fat_apk_cpu=doesnotexist", "--android_crosstool_top=//android/crosstool:everything");

    AssertionError noToolchainError =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test/starlark:test"));
    assertThat(noToolchainError)
        .hasMessageThat()
        .contains("does not contain a toolchain for cpu 'doesnotexist'");
  }

  @Test
  public void testAssetsInExternalRepository() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"), "local_repository(name='r', path='/r')");
    scratch.file("/r/WORKSPACE");
    scratch.file("/r/p/BUILD", "filegroup(name='assets', srcs=['a/b'])");
    scratch.file("/r/p/a/b");
    invalidatePackages();
    scratchConfiguredTarget(
        "java/a",
        "a",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    assets = ['@r//p:assets'],",
        "    assets_dir = '')");
  }

  @Test
  public void testMultidexModeAndMainDexProguardSpecs() throws Exception {
    checkError(
        "java/a",
        "a",
        "only allowed if 'multidex' is set to 'legacy'",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    main_dex_proguard_specs = ['foo'])");
  }

  @Test
  public void testAndroidManifestWithCustomName() throws Exception {
    scratchConfiguredTarget(
        "java/a",
        "a",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'SomeOtherAndroidManifest.xml')");
    assertNoEvents();
  }

  @Test
  public void testMainDexProguardSpecs() throws Exception {
    useConfiguration("--noincremental_dexing");
    ConfiguredTarget ct =
        scratchConfiguredTarget(
            "java/a",
            "a",
            "android_binary(",
            "    name = 'a',",
            "    srcs = ['A.java'],",
            "    manifest = 'AndroidManifest.xml',",
            "    multidex = 'legacy',",
            "    main_dex_proguard_specs = ['a.spec'])");

    Artifact intermediateJar =
        artifactByPath(
            ImmutableList.of(getCompressedUnsignedApk(ct)),
            ".apk",
            ".dex.zip",
            ".dex.zip",
            "main_dex_list.txt",
            "_intermediate.jar");
    List<String> args = getGeneratingSpawnActionArgs(intermediateJar);
    MoreAsserts.assertContainsSublist(args, "-include", "java/a/a.spec");
    assertThat(Joiner.on(" ").join(args)).doesNotContain("mainDexClasses.rules");
  }

  @Test
  public void testLegacyMainDexListGenerator() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'legacy')");
    scratch.file(
        "tools/fake/BUILD",
        "cc_binary(",
        "    name = 'generate_main_dex_list',",
        "    srcs = ['main.cc'])");
    useConfiguration("--legacy_main_dex_list_generator=//tools/fake:generate_main_dex_list");

    ConfiguredTarget binary = getConfiguredTarget("//java/a:a");
    Artifact mainDexList =
        ActionsTestUtil.getFirstArtifactEndingWith(
            actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)), "main_dex_list.txt");
    List<String> args = getGeneratingSpawnActionArgs(mainDexList);
    NestedSet<Artifact> mainDexInputs = getGeneratingAction(mainDexList).getInputs();

    MoreAsserts.assertContainsSublist(args, "--lib", getAndroidJarPath());
    MoreAsserts.assertContainsSublist(args, "--main-dex-rules", getMainDexClassesPath());

    assertThat(ActionsTestUtil.baseArtifactNames(mainDexInputs)).contains("generate_main_dex_list");
    assertThat(ActionsTestUtil.baseArtifactNames(mainDexInputs)).contains("a_deploy.jar");
    assertThat(ActionsTestUtil.baseArtifactNames(mainDexInputs)).contains(getAndroidJarFilename());
    assertThat(ActionsTestUtil.baseArtifactNames(mainDexInputs))
        .contains(getMainDexClassesFilename());
    assertThat(ActionsTestUtil.baseArtifactNames(mainDexInputs))
        .contains("main_dex_a_proguard.cfg");
    assertThat(getFirstArtifactEndingWith(mainDexInputs, "main_dex_list_creator")).isNull();
  }

  @Test
  public void testMainDexListObfuscation() throws Exception {
    useConfiguration("--noincremental_dexing");
    scratch.file("/java/a/list.txt");
    ConfiguredTarget ct =
        scratchConfiguredTarget(
            "java/a",
            "a",
            "android_binary(",
            "    name = 'a',",
            "    srcs = ['A.java'],",
            "    manifest = 'AndroidManifest.xml',",
            "    multidex = 'manual_main_dex',",
            "    proguard_generate_mapping = 1,",
            "    main_dex_list = 'list.txt')");

    Artifact obfuscatedDexList =
        artifactByPath(
            ImmutableList.of(getCompressedUnsignedApk(ct)),
            ".apk",
            ".dex.zip",
            ".dex.zip",
            "main_dex_list_obfuscated.txt");
    List<String> args = getGeneratingSpawnActionArgs(obfuscatedDexList);
    assertThat(args.get(0)).contains("dex_list_obfuscator");
    MoreAsserts.assertContainsSublist(args, "--input", "java/a/list.txt");
  }

  @Test
  public void testNonLegacyNativeDepsDoesNotPolluteDexSharding() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(name = 'a',",
        "               manifest = 'AndroidManifest.xml',",
        "               multidex = 'native',",
        "               deps = [':cc'],",
        "               dex_shards = 2)",
        "cc_library(name = 'cc',",
        "           srcs = ['cc.cc'])");

    Artifact jarShard =
        artifactByPath(
            ImmutableList.of(getCompressedUnsignedApk(getConfiguredTarget("//java/a:a"))),
            ".apk",
            "classes.dex.zip",
            "shard1.dex.zip",
            "shard1.jar.dex.zip");
    NestedSet<Artifact> shardInputs = getGeneratingAction(jarShard).getInputs();
    assertThat(getFirstArtifactEndingWith(shardInputs, ".txt")).isNull();
  }

  @Test
  public void testCcInfoDeps() throws Exception {
    scratch.file(
        "java/a/cc_info.bzl",
        "def _impl(ctx):",
        "  cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(",
        "    ctx = ctx,",
        "    cc_toolchain = cc_toolchain,",
        "    requested_features = ctx.features,",
        "    unsupported_features = ctx.disabled_features,",
        "  )",
        "  library_to_link = cc_common.create_library_to_link(",
        "    actions=ctx.actions, feature_configuration=feature_configuration, ",
        "    cc_toolchain = cc_toolchain, ",
        "    static_library=ctx.file.static_library)",
        "  linker_input = cc_common.create_linker_input(",
        "    libraries = depset([library_to_link]),",
        "    user_link_flags=depset(ctx.attr.user_link_flags),",
        "    owner = ctx.label,",
        "  )",
        "  linking_context = cc_common.create_linking_context(",
        "    linker_inputs=depset([linker_input]))",
        "  return [CcInfo(linking_context=linking_context)]",
        "cc_info = rule(",
        "  implementation=_impl,",
        "  fragments = [\"cpp\"],",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True),",
        "    'user_link_flags' : attr.string_list(),",
        "    'static_library': attr.label(allow_single_file=True),",
        "    '_cc_toolchain': attr.label(default=Label('//java/a:alias'))",
        "  },",
        ");");
    scratch.file(
        "java/a/BUILD",
        "load('//java/a:cc_info.bzl', 'cc_info')",
        "cc_toolchain_alias(name='alias')",
        "android_binary(",
        "    name = 'a',",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'native',",
        "    deps = [':cc_info'],",
        ")",
        "cc_info(",
        "    name = 'cc_info',",
        "    user_link_flags = ['-first_flag', '-second_flag'],",
        "    static_library = 'cc_info.a',",
        ")");

    ConfiguredTarget app = getConfiguredTarget("//java/a:a");
    assertNoEvents();

    Artifact copiedLib = getOnlyElement(getNativeLibrariesInApk(app));
    Artifact linkedLib = getGeneratingAction(copiedLib).getInputs().getSingleton();
    CppLinkAction action = (CppLinkAction) getGeneratingAction(linkedLib);

    assertThat(action.getArguments()).containsAtLeast("-first_flag", "-second_flag");

    NestedSet<Artifact> linkInputs = action.getInputs();
    assertThat(ActionsTestUtil.baseArtifactNames(linkInputs)).contains("cc_info.a");
  }

  @Test
  public void testCcInfoDepsViaAndroidLibrary() throws Exception {
    scratch.file(
        "java/a/cc_info.bzl",
        "def _impl(ctx):",
        "  cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(",
        "    ctx = ctx,",
        "    cc_toolchain = cc_toolchain,",
        "    requested_features = ctx.features,",
        "    unsupported_features = ctx.disabled_features,",
        "  )",
        "  library_to_link = cc_common.create_library_to_link(",
        "    actions=ctx.actions, feature_configuration=feature_configuration, ",
        "    cc_toolchain = cc_toolchain, ",
        "    static_library=ctx.file.static_library)",
        "  linker_input = cc_common.create_linker_input(",
        "    libraries = depset([library_to_link]),",
        "    user_link_flags=depset(ctx.attr.user_link_flags),",
        "    owner = ctx.label,",
        "  )",
        "  linking_context = cc_common.create_linking_context(",
        "    linker_inputs=depset([linker_input]))",
        "  return [CcInfo(linking_context=linking_context)]",
        "cc_info = rule(",
        "  implementation=_impl,",
        "  fragments = [\"cpp\"],",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True),",
        "    'user_link_flags' : attr.string_list(),",
        "    'static_library': attr.label(allow_single_file=True),",
        "    '_cc_toolchain': attr.label(default=Label('//java/a:alias'))",
        "  },",
        ");");
    scratch.file(
        "java/a/BUILD",
        "load('//java/a:cc_info.bzl', 'cc_info')",
        "cc_toolchain_alias(name='alias')",
        "android_binary(",
        "    name = 'a',",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'native',",
        "    deps = [':liba'],",
        ")",
        "android_library(",
        "    name = 'liba',",
        "    srcs = ['a.java'],",
        "    deps = [':cc_info'],",
        ")",
        "cc_info(",
        "    name = 'cc_info',",
        "    user_link_flags = ['-first_flag', '-second_flag'],",
        "    static_library = 'cc_info.a',",
        ")");

    ConfiguredTarget app = getConfiguredTarget("//java/a:a");

    Artifact copiedLib = getOnlyElement(getNativeLibrariesInApk(app));
    Artifact linkedLib = getGeneratingAction(copiedLib).getInputs().getSingleton();
    CppLinkAction action = (CppLinkAction) getGeneratingAction(linkedLib);

    assertThat(action.getArguments()).containsAtLeast("-first_flag", "-second_flag");

    NestedSet<Artifact> linkInputs = action.getInputs();
    assertThat(ActionsTestUtil.baseArtifactNames(linkInputs)).contains("cc_info.a");
  }

  @Test
  public void testJavaPluginProcessorPath() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "java_library(name = 'plugin_dep',",
        "    srcs = [ 'ProcessorDep.java'])",
        "java_plugin(name = 'plugin',",
        "    srcs = ['AnnotationProcessor.java'],",
        "    processor_class = 'com.google.process.stuff',",
        "    deps = [ ':plugin_dep' ])",
        "android_binary(name = 'to_be_processed',",
        "    manifest = 'AndroidManifest.xml',",
        "    plugins = [':plugin'],",
        "    srcs = ['ToBeProcessed.java'])");
    ConfiguredTarget target = getConfiguredTarget("//java/test:to_be_processed");
    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingAction(getBinArtifact("libto_be_processed.jar", target));

    assertThat(getProcessorNames(javacAction)).contains("com.google.process.stuff");
    assertThat(getProcessorNames(javacAction)).hasSize(1);

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                getInputs(javacAction, getProcessorpath(javacAction))))
        .containsExactly("libplugin.jar", "libplugin_dep.jar");
    assertThat(
            actionsTestUtil()
                .predecessorClosureOf(getFilesToBuild(target), JavaSemantics.JAVA_SOURCE))
        .isEqualTo("ToBeProcessed.java AnnotationProcessor.java ProcessorDep.java");
  }

  // Same test as above, enabling the plugin through the command line.
  @Test
  public void testPluginCommandLine() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "java_library(name = 'plugin_dep',",
        "    srcs = [ 'ProcessorDep.java'])",
        "java_plugin(name = 'plugin',",
        "    srcs = ['AnnotationProcessor.java'],",
        "    processor_class = 'com.google.process.stuff',",
        "    deps = [ ':plugin_dep' ])",
        "android_binary(name = 'to_be_processed',",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['ToBeProcessed.java'])");

    useConfiguration("--plugin=//java/test:plugin");
    ConfiguredTarget target = getConfiguredTarget("//java/test:to_be_processed");
    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingAction(getBinArtifact("libto_be_processed.jar", target));

    assertThat(getProcessorNames(javacAction)).contains("com.google.process.stuff");
    assertThat(getProcessorNames(javacAction)).hasSize(1);
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                getInputs(javacAction, getProcessorpath(javacAction))))
        .containsExactly("libplugin.jar", "libplugin_dep.jar");
    assertThat(
            actionsTestUtil()
                .predecessorClosureOf(getFilesToBuild(target), JavaSemantics.JAVA_SOURCE))
        .isEqualTo("ToBeProcessed.java AnnotationProcessor.java ProcessorDep.java");
  }

  @Test
  public void testInvalidPlugin() throws Exception {
    checkError(
        "java/test",
        "lib",
        // error:
        getErrorMsgMisplacedRules(
            "plugins",
            "android_binary",
            "//java/test:lib",
            "java_library",
            "//java/test:not_a_plugin"),
        // BUILD file:
        "java_library(name = 'not_a_plugin',",
        "    srcs = [ 'NotAPlugin.java'])",
        "android_binary(name = 'lib',",
        "    plugins = [':not_a_plugin'],",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['Lib.java'])");
  }

  @Test
  public void testBaselineCoverageArtifacts() throws Exception {
    useConfiguration("--collect_code_coverage");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "java/com/google/a",
            "bin",
            "android_binary(name='bin', srcs=['Main.java'], manifest='AndroidManifest.xml')");

    assertThat(baselineCoverageArtifactBasenames(target)).containsExactly("Main.java");
  }

  @Test
  public void testSameSoFromMultipleDeps() throws Exception {
    scratch.file(
        "java/d/BUILD",
        "genrule(name='genrule', srcs=[], outs=['genrule.so'], cmd='')",
        "cc_library(name='cc1', srcs=[':genrule.so'])",
        "cc_library(name='cc2', srcs=[':genrule.so'])",
        "android_binary(name='ab', deps=[':cc1', ':cc2'], manifest='AndroidManifest.xml')");
    getConfiguredTarget("//java/d:ab");
  }

  @Test
  public void testSimpleBinary_desugarJava8() throws Exception {
    useConfiguration("--experimental_desugar_for_android");
    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)), "_deploy.jar");
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .contains("libapp.jar_desugared.jar");
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs())).doesNotContain("libapp.jar");
  }

  /**
   * Tests that --experimental_check_desugar_deps causes the relevant flags to be set on desugaring
   * and singlejar actions, and makes sure the deploy jar is built even when just building an APK.
   */
  @Test
  public void testSimpleBinary_checkDesugarDepsAlwaysHappens() throws Exception {
    useConfiguration("--experimental_check_desugar_deps");
    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");
    assertNoEvents();

    // 1. Find app's deploy jar and make sure checking flags are set for it and its inputs
    SpawnAction singlejar =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "/app_deploy.jar");
    assertThat(getGeneratingSpawnActionArgs(singlejar.getPrimaryOutput()))
        .contains("--check_desugar_deps");

    SpawnAction desugar =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(singlejar.getInputs(), "/libapp.jar_desugared.jar");
    assertThat(desugar).isNotNull();
    assertThat(getGeneratingSpawnActionArgs(desugar.getPrimaryOutput()))
        .contains("--emit_dependency_metadata_as_needed");

    // 2. Make sure all APK outputs depend on the deploy Jar.
    int found = 0;
    for (Artifact built : getFilesToBuild(binary).toList()) {
      if (built.getExtension().equals("apk")) {
        // If this assertion breaks then APK artifacts have stopped depending on deploy jars.
        // If that's desired then we'll need to make sure dependency checking is done in another
        // action that APK artifacts depend on, in addition to the check that happens when building
        // deploy.jars, which we assert above.
        assertWithMessage("%s dependency on deploy.jar", built.getFilename())
            .that(actionsTestUtil().artifactClosureOf(built))
            .contains(singlejar.getPrimaryOutput());
        ++found;
      }
    }
    assertThat(found).isEqualTo(2 /* signed and unsigned apks */);
  }

  // regression test for #3169099
  @Test
  public void testBinarySrcs() throws Exception {
    scratch.file("java/srcs/a.foo", "foo");
    scratch.file(
        "java/srcs/BUILD",
        "android_binary(name = 'valid', manifest = 'AndroidManifest.xml', "
            + "srcs = ['a.java', 'b.srcjar', ':gvalid', ':gmix'])",
        "android_binary(name = 'invalid', manifest = 'AndroidManifest.xml', "
            + "srcs = ['a.foo', ':ginvalid'])",
        "android_binary(name = 'mix', manifest = 'AndroidManifest.xml', "
            + "srcs = ['a.java', 'a.foo'])",
        "genrule(name = 'gvalid', srcs = ['a.java'], outs = ['b.java'], cmd = '')",
        "genrule(name = 'ginvalid', srcs = ['a.java'], outs = ['b.foo'], cmd = '')",
        "genrule(name = 'gmix', srcs = ['a.java'], outs = ['c.java', 'c.foo'], cmd = '')");
    assertSrcsValidityForRuleType("//java/srcs", "android_binary", ".java or .srcjar");
  }

  // regression test for #3169095
  @Test
  public void testXmbInSrcs_notPermittedButDoesNotThrow() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratchConfiguredTarget(
        "java/xmb",
        "a",
        "android_binary(name = 'a', manifest = 'AndroidManifest.xml', srcs = ['a.xmb'])");
    // We expect there to be an error here because a.xmb is not a valid src,
    // and more importantly, no exception to have been thrown.
    assertContainsEvent(
        "in srcs attribute of android_binary rule //java/xmb:a: "
            + "target '//java/xmb:a.xmb' does not exist");
  }

  @Test
  public void testNativeLibraryBasenameCollision() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "java/android/common/BUILD",
        "cc_library(name = 'libcommon_armeabi',",
        "           srcs = ['armeabi/native.so'],)");
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'libnative',",
        "           srcs = ['native.so'],)",
        "android_binary(name = 'b',",
        "               srcs = ['A.java'],",
        "               deps = [':libnative', '//java/android/common:libcommon_armeabi'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");
    getConfiguredTarget("//java/android/app:b");
    assertContainsEvent(
        "Each library in the transitive closure must have a unique basename to avoid name"
            + " collisions when packaged into an apk, but two libraries have the basename"
            + " 'native.so': java/android/common/armeabi/native.so and"
            + " java/android/app/native.so");
  }

  private void setupNativeLibrariesForLinking() throws Exception {
    scratch.file(
        "java/android/common/BUILD",
        "cc_library(name = 'common_native',",
        "           srcs = ['common.cc'],)",
        "android_library(name = 'common',",
        "                exports = [':common_native'],)");
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'native',",
        "           srcs = ['native.cc'],)",
        "android_binary(name = 'auto',",
        "               srcs = ['A.java'],",
        "               deps = [':native', '//java/android/common:common'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )",
        "android_binary(name = 'off',",
        "               srcs = ['A.java'],",
        "               deps = [':native', '//java/android/common:common'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");
  }

  private void assertNativeLibraryLinked(ConfiguredTarget target, String... srcNames) {
    Artifact linkedLib = getOnlyElement(getNativeLibrariesInApk(target));
    assertThat(linkedLib.getFilename())
        .isEqualTo("lib" + target.getLabel().toPathFragment().getBaseName() + ".so");
    assertThat(linkedLib.isSourceArtifact()).isFalse();
    assertWithMessage("Native libraries were not linked to produce " + linkedLib)
        .that(getGeneratingLabelForArtifact(linkedLib))
        .isEqualTo(target.getLabel());
    assertThat(artifactsToStrings(actionsTestUtil().artifactClosureOf(linkedLib)))
        .containsAtLeastElementsIn(ImmutableSet.copyOf(Arrays.asList(srcNames)));
  }

  @Test
  public void testNativeLibrary_linksLibrariesWhenCodeIsPresent() throws Exception {
    setupNativeLibrariesForLinking();
    assertNativeLibraryLinked(
        getConfiguredTarget("//java/android/app:auto"),
        "src java/android/common/common.cc",
        "src java/android/app/native.cc");
    assertNativeLibraryLinked(
        getConfiguredTarget("//java/android/app:off"),
        "src java/android/common/common.cc",
        "src java/android/app/native.cc");
  }

  @Test
  public void testNativeLibrary_copiesLibrariesDespiteExtraLayersOfIndirection() throws Exception {
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'native_dep',",
        "           srcs = ['dep.so'])",
        "cc_library(name = 'native',",
        "           srcs = ['native_prebuilt.so'],",
        "           deps = [':native_dep'])",
        "cc_library(name = 'native_wrapper',",
        "           deps = [':native'])",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               deps = [':native_wrapper'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");
    assertNativeLibrariesCopiedNotLinked(
        getConfiguredTarget("//java/android/app:app"),
        "src java/android/app/dep.so",
        "src java/android/app/native_prebuilt.so");
  }

  @Test
  public void testNativeLibrary_copiesLibrariesWrappedInCcLibraryWithSameName() throws Exception {
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'native',",
        "           srcs = ['libnative.so'])",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               deps = [':native'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");
    assertNativeLibrariesCopiedNotLinked(
        getConfiguredTarget("//java/android/app:app"), "src java/android/app/libnative.so");
  }

  @Test
  public void testNativeLibrary_linksWhenPrebuiltArchiveIsSupplied() throws Exception {
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'native_dep',",
        "           srcs = ['dep.lo'])",
        "cc_library(name = 'native',",
        "           srcs = ['native_prebuilt.a'],",
        "           deps = [':native_dep'])",
        "cc_library(name = 'native_wrapper',",
        "           deps = [':native'])",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               deps = [':native_wrapper'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");
    assertNativeLibraryLinked(
        getConfiguredTarget("//java/android/app:app"), "src java/android/app/native_prebuilt.a");
  }

  @Test
  public void testNativeLibrary_copiesFullLibrariesInIfsoMode() throws Exception {
    useConfiguration("--interface_shared_objects");
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'native_dep',",
        "           srcs = ['dep.so'])",
        "cc_library(name = 'native',",
        "           srcs = ['native.cc', 'native_prebuilt.so'],",
        "           deps = [':native_dep'])",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               deps = [':native'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");
    ConfiguredTarget app = getConfiguredTarget("//java/android/app:app");
    Iterable<Artifact> nativeLibraries = getNativeLibrariesInApk(app);
    assertThat(artifactsToStrings(nativeLibraries))
        .containsAtLeast("src java/android/app/native_prebuilt.so", "src java/android/app/dep.so");
    assertThat(FileType.filter(nativeLibraries, CppFileTypes.INTERFACE_SHARED_LIBRARY)).isEmpty();
  }

  @Test
  public void testNativeLibrary_providesLinkerScriptToLinkAction() throws Exception {
    scratch.file(
        "java/android/app/BUILD",
        "cc_library(name = 'native',",
        "           srcs = ['native.cc'],",
        "           linkopts = ['-Wl,-version-script', '$(location jni.lds)'],",
        "           deps = ['jni.lds'],)",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               deps = [':native'],",
        "               manifest = 'AndroidManifest.xml',",
        "              )");

    ConfiguredTarget app = getConfiguredTarget("//java/android/app:app");
    Artifact copiedLib = getOnlyElement(getNativeLibrariesInApk(app));
    Artifact linkedLib = getGeneratingAction(copiedLib).getInputs().getSingleton();
    NestedSet<Artifact> linkInputs = getGeneratingAction(linkedLib).getInputs();
    assertThat(ActionsTestUtil.baseArtifactNames(linkInputs)).contains("jni.lds");
  }

  /** Regression test for http://b/33173461. */
  @Test
  public void testIncrementalDexingUsesDexArchives_binaryDependingOnAliasTarget() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_library(",
        "  name = 'dep',",
        "  srcs = ['dep.java'],",
        "  resource_files = glob(['res/**']),",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "alias(",
        "  name = 'alt',",
        "  actual = ':dep',",
        ")",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  multidex = 'native',",
        "  manifest = 'AndroidManifest.xml',",
        "  deps = [':alt',],",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();

    Action shardAction = getGeneratingAction(getBinArtifact("_dx/top/classes.jar", topTarget));
    for (Artifact input : getNonToolInputs(shardAction)) {
      String basename = input.getFilename();
      // all jars are converted to dex archives
      assertWithMessage(basename)
          .that(!basename.contains(".jar") || basename.endsWith(".jar.dex.zip"))
          .isTrue();
      // all jars are desugared before being converted
      if (basename.endsWith(".jar.dex.zip")) {
        assertThat(getGeneratingAction(input).getPrimaryInput().getFilename())
            .isEqualTo(
                basename.substring(0, basename.length() - ".jar.dex.zip".length())
                    + ".jar_desugared.jar");
      }
    }
    // Make sure exactly the dex archives generated for top and dependents appear.  We also *don't*
    // want neverlink and unused_dep to appear, and to be safe we do so by explicitly enumerating
    // *all* expected input dex archives.
    assertThat(
            Iterables.filter(
                ActionsTestUtil.baseArtifactNames(getNonToolInputs(shardAction)),
                Predicates.containsPattern("\\.jar")))
        .containsExactly(
            // top's dex archives
            "libtop.jar.dex.zip",
            "top_resources.jar.dex.zip",
            // dep's dex archives
            "libdep.jar.dex.zip");
  }

  @Test
  public void testIncrementalDexingDisabledWithBlacklistedDexopts() throws Exception {
    // Even if we mark a dx flag as supported, incremental dexing isn't used with disallowlisted
    // dexopts (unless incremental_dexing attribute is set, which a different test covers)
    useConfiguration(
        "--incremental_dexing",
        "--non_incremental_per_target_dexopts=--no-locals",
        "--dexopts_supported_in_incremental_dexing=--no-locals");
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  dexopts = ['--no-locals'],",
        "  dex_shards = 2,",
        "  multidex = 'native',",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();
    Action shardAction = getGeneratingAction(getBinArtifact("_dx/top/shard1.jar", topTarget));
    assertThat(
            Iterables.filter(
                ActionsTestUtil.baseArtifactNames(getNonToolInputs(shardAction)),
                Predicates.containsPattern("\\.jar\\.dex\\.zip")))
        .isEmpty(); // no dex archives are used
  }

  @Test
  public void testIncrementalDexingDisabledWithProguard() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  proguard_specs = ['proguard.cfg'],",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();
    Action dexAction = getGeneratingAction(getBinArtifact("_dx/top/classes.dex", topTarget));
    assertThat(
            Iterables.filter(
                ActionsTestUtil.baseArtifactNames(dexAction.getInputs()),
                Predicates.containsPattern("\\.jar")))
        .containsExactly("top_proguard.jar", "dx_binary.jar"); // proguard output is used directly
  }

  @Test
  public void testIncrementalDexing_incompatibleWithProguardWhenDisabled() throws Exception {
    useConfiguration("--experimental_incremental_dexing_after_proguard=0"); // disable with Proguard
    checkError(
        "java/com/google/android",
        "top",
        "target cannot be incrementally dexed",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  proguard_specs = ['proguard.cfg'],",
        "  incremental_dexing = 1,",
        ")");
  }

  @Test
  public void testIncrementalDexingAfterProguard_unsharded() throws Exception {
    useConfiguration("--experimental_incremental_dexing_after_proguard=1");
    // Use "legacy" multidex mode so we get a main dex list file and can test that it's passed to
    // the splitter action (similar to _withDexShards below), unlike without the dex splitter where
    // the main dex list goes to the merging action.
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  incremental_dexing = 1,",
        "  multidex = 'legacy',",
        "  dexopts = ['--minimal-main-dex', '--positions=none'],",
        "  proguard_specs = ['b.pro'],",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();

    SpawnAction shardAction =
        getGeneratingSpawnAction(getBinArtifact("_dx/top/classes.dex.zip", topTarget));
    assertThat(shardAction.getArguments()).contains("--main-dex-list");
    assertThat(shardAction.getArguments()).contains("--minimal-main-dex");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(shardAction)))
        .containsExactly("classes.jar", "main_dex_list.txt");

    // --positions dexopt is supported after Proguard, even though not normally otherwise
    assertThat(
            paramFileArgsForAction(
                getGeneratingSpawnAction(getBinArtifact("_dx/top/classes.jar", topTarget))))
        .contains("--positions=none");
  }

  @Test
  public void testIncrementalDexingAfterProguard_autoShardedMultidexAutoOptIn() throws Exception {
    useConfiguration(
        "--experimental_incremental_dexing_after_proguard=3",
        "--experimental_incremental_dexing_after_proguard_by_default");
    // Use "legacy" multidex mode so we get a main dex list file and can test that it's passed to
    // the splitter action (similar to _withDexShards below), unlike without the dex splitter where
    // the main dex list goes to the merging action.
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  multidex = 'legacy',",
        "  dexopts = ['--minimal-main-dex', '--positions=none'],",
        "  proguard_specs = ['b.pro'],",
        ")"); // incremental_dexing = 1 attribute not needed

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();

    SpawnAction splitAction = getGeneratingSpawnAction(getTreeArtifact("dexsplits/top", topTarget));
    assertThat(splitAction.getArguments()).contains("--main-dex-list");
    assertThat(splitAction.getArguments()).contains("--minimal-main-dex");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(splitAction)))
        .containsExactly(
            "shard1.jar.dex.zip", "shard2.jar.dex.zip", "shard3.jar.dex.zip", "main_dex_list.txt");

    SpawnAction shuffleAction =
        getGeneratingSpawnAction(getBinArtifact("_dx/top/shard1.jar", topTarget));
    assertThat(shuffleAction.getArguments()).doesNotContain("--main-dex-list");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(shuffleAction)))
        .containsExactly("top_proguard.jar");

    // --positions dexopt is supported after Proguard, even though not normally otherwise
    assertThat(
            paramFileArgsForAction(
                getGeneratingSpawnAction(getBinArtifact("_dx/top/shard3.jar.dex.zip", topTarget))))
        .contains("--positions=none");
  }

  @Test
  public void testIncrementalDexingAfterProguard_explicitDexShards() throws Exception {
    useConfiguration("--experimental_incremental_dexing_after_proguard=2");
    // Use "legacy" multidex mode so we get a main dex list file and can test that it's passed to
    // the shardAction, not to the subsequent dexMerger action.  Without dex_shards, main dex list
    // file goes to the dexMerger instead (see _multidex test).
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  dex_shards = 25,",
        "  incremental_dexing = 1,",
        "  multidex = 'legacy',",
        "  proguard_specs = ['b.pro'],",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();
    SpawnAction shardAction =
        getGeneratingSpawnAction(getBinArtifact("_dx/top/shard25.jar", topTarget));
    assertThat(shardAction.getArguments()).contains("--main_dex_filter");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(shardAction)))
        .containsExactly("top_proguard.jar", "main_dex_list.txt");
    SpawnAction mergeAction =
        getGeneratingSpawnAction(getBinArtifact("_dx/top/shard1.jar.dex.zip", topTarget));
    assertThat(mergeAction.getArguments()).doesNotContain("--main-dex-list");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(mergeAction)))
        .contains("shard1.jar");
  }

  @Test
  public void testIncrementalDexingAfterProguard_autoShardedMonodex() throws Exception {
    useConfiguration("--experimental_incremental_dexing_after_proguard=3");
    // Use "legacy" multidex mode so we get a main dex list file and can test that it's passed to
    // the splitter action (similar to _withDexShards below), unlike without the dex splitter where
    // the main dex list goes to the merging action.
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'top',",
        "  srcs = ['foo.java', 'bar.srcjar'],",
        "  manifest = 'AndroidManifest.xml',",
        "  incremental_dexing = 1,",
        "  multidex = 'off',",
        "  proguard_specs = ['b.pro'],",
        ")");

    ConfiguredTarget topTarget = getConfiguredTarget("//java/com/google/android:top");
    assertNoEvents();
    SpawnAction mergeAction =
        getGeneratingSpawnAction(getBinArtifact("_dx/top/classes.dex.zip", topTarget));
    assertThat(mergeAction.getArguments()).doesNotContain("--main-dex-list");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(mergeAction)))
        .containsExactly("shard1.jar.dex.zip", "shard2.jar.dex.zip", "shard3.jar.dex.zip");
    SpawnAction shuffleAction =
        getGeneratingSpawnAction(getBinArtifact("_dx/top/shard1.jar", topTarget));
    assertThat(shuffleAction.getArguments()).doesNotContain("--main-dex-list");
    assertThat(ActionsTestUtil.baseArtifactNames(getNonToolInputs(shuffleAction)))
        .containsExactly("top_proguard.jar");
  }

  @Test
  public void testV1SigningMethod() throws Exception {
    actualSignerToolTests("v1", "true", "false");
  }

  @Test
  public void testV2SigningMethod() throws Exception {
    actualSignerToolTests("v2", "false", "true");
  }

  @Test
  public void testV1V2SigningMethod() throws Exception {
    actualSignerToolTests("v1_v2", "true", "true");
  }

  private void actualSignerToolTests(String apkSigningMethod, String signV1, String signV2)
      throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',)");
    useConfiguration("--apk_signing_method=" + apkSigningMethod);
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:hello");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));
    assertThat(getFirstArtifactEndingWith(artifacts, "signed_hello.apk")).isNull();
    SpawnAction unsignedApkAction =
        (SpawnAction)
            actionsTestUtil().getActionForArtifactEndingWith(artifacts, "/hello_unsigned.apk");
    assertThat(
            unsignedApkAction.getInputs().toList().stream()
                .map(Artifact::getFilename)
                .anyMatch(filename -> Ascii.toLowerCase(filename).contains("singlejar")))
        .isTrue();
    SpawnAction compressedUnsignedApkAction =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(artifacts, "compressed_hello_unsigned.apk");
    assertThat(
            compressedUnsignedApkAction.getInputs().toList().stream()
                .map(Artifact::getFilename)
                .anyMatch(filename -> Ascii.toLowerCase(filename).contains("singlejar")))
        .isTrue();
    SpawnAction zipalignAction =
        (SpawnAction)
            actionsTestUtil().getActionForArtifactEndingWith(artifacts, "zipaligned_hello.apk");
    assertThat(zipalignAction.getCommandFilename()).endsWith("zipalign");
    Artifact a = ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "hello.apk");
    assertThat(getGeneratingSpawnAction(a).getCommandFilename()).endsWith("ApkSignerBinary");
    List<String> args = getGeneratingSpawnActionArgs(a);

    assertThat(flagValue("--v1-signing-enabled", args)).isEqualTo(signV1);
    assertThat(flagValue("--v2-signing-enabled", args)).isEqualTo(signV2);
  }

  @Test
  public void testResourcePathShortening_flagEnabledAndCOpt_optimizedApkIsInputToApkBuilderAction()
      throws Exception {
    useConfiguration("--experimental_android_resource_path_shortening", "-c", "opt");

    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    SpawnAction optimizeAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "app_optimized.ap_"));
    assertThat(optimizeAction.getMnemonic()).isEqualTo("Aapt2Optimize");
    assertThat(getGeneratingAction(getFirstArtifactEndingWith(artifacts, "app_resource_paths.map")))
        .isEqualTo(optimizeAction);

    List<String> processingArgs = optimizeAction.getArguments();
    assertThat(processingArgs).contains("--shorten-resource-paths");
    assertThat(flagValue("--resource-path-shortening-map", processingArgs))
        .endsWith("app_resource_paths.map");

    // Verify that the optimized APK is an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "app_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "app_optimized.ap_")).isTrue();
  }

  @Test
  public void testResourcePathShortening_flagEnabledAndCDefault_optimizeArtifactsAbsent()
      throws Exception {
    useConfiguration("--experimental_android_resource_path_shortening");

    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    Artifact resourcePathShorteningMapArtifact =
        getFirstArtifactEndingWith(artifacts, "app_resource_paths.map");
    assertThat(resourcePathShorteningMapArtifact).isNull();

    Artifact optimizedResourceApk = getFirstArtifactEndingWith(artifacts, "app_optimized.ap_");
    assertThat(optimizedResourceApk).isNull();

    // Verify that the optimized APK is not an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "app_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "app_optimized.ap_")).isFalse();
  }

  @Test
  public void testResourcePathShortening_flagNotEnabledAndCOpt_optimizeArtifactsAbsent()
      throws Exception {
    useConfiguration("-c", "opt");

    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    Artifact resourcePathShorteningMapArtifact =
        getFirstArtifactEndingWith(artifacts, "app_resource_paths.map");
    assertThat(resourcePathShorteningMapArtifact).isNull();

    Artifact optimizedResourceApk = getFirstArtifactEndingWith(artifacts, "app_optimized.ap_");
    assertThat(optimizedResourceApk).isNull();

    // Verify that the optimized APK is not an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "app_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "app_optimized.ap_")).isFalse();
  }

  @Test
  public void resourceNameCollapse_flagAndProguardSpecsPresent_optimizedApkIsInputToApkBuilder()
      throws Exception {
    useConfiguration("--experimental_android_resource_name_obfuscation");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               shrink_resources = 1,",
        "               proguard_specs = ['proguard-spec.pro'],)");

    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:hello");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    SpawnAction shrinkerAction =
        getGeneratingSpawnAction(
            getFirstArtifactEndingWith(artifacts, "resource_optimization.cfg"));
    assertThat(shrinkerAction.getMnemonic()).isEqualTo("ResourceShrinker");
    SpawnAction optimizeAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "hello_optimized.ap_"));
    assertThat(optimizeAction.getMnemonic()).isEqualTo("Aapt2Optimize");

    List<String> processingArgs = optimizeAction.getArguments();
    assertThat(processingArgs).contains("--collapse-resource-names");
    assertThat(flagValue("--resources-config-path", processingArgs))
        .endsWith("resource_optimization.cfg");

    // Verify that the optimized APK is an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "hello_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "hello_optimized.ap_")).isTrue();
  }

  @Test
  public void resourceNameCollapse_featureAndProguardSpecsPresent_optimizedApkIsInputToApkBuilder()
      throws Exception {
    useConfiguration("--features=resource_name_obfuscation");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               shrink_resources = 1,",
        "               proguard_specs = ['proguard-spec.pro'],)");

    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:hello");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    SpawnAction shrinkerAction =
        getGeneratingSpawnAction(
            getFirstArtifactEndingWith(artifacts, "resource_optimization.cfg"));
    assertThat(shrinkerAction.getMnemonic()).isEqualTo("ResourceShrinker");
    SpawnAction optimizeAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "hello_optimized.ap_"));
    assertThat(optimizeAction.getMnemonic()).isEqualTo("Aapt2Optimize");

    List<String> processingArgs = optimizeAction.getArguments();
    assertThat(processingArgs).contains("--collapse-resource-names");
    assertThat(flagValue("--resources-config-path", processingArgs))
        .endsWith("resource_optimization.cfg");

    // Verify that the optimized APK is an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "hello_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "hello_optimized.ap_")).isTrue();
  }

  @Test
  public void resourceNameCollapse_flagPresentProguardSpecsAbsent_optimizeArtifactsAbsent()
      throws Exception {
    useConfiguration("--experimental_android_resource_name_obfuscation");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               shrink_resources = 1,)");

    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:hello");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    Artifact optimizedResourceApk = getFirstArtifactEndingWith(artifacts, "hello_optimized.ap_");
    assertThat(optimizedResourceApk).isNull();

    // Verify that the optimized APK is not an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "hello_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "hello_optimized.ap_")).isFalse();
  }

  @Test
  public void resourceNameCollapse_flagAbsentProguardSpecsPresent_optimizeArtifactsAbsent()
      throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               shrink_resources = 1,",
        "               proguard_specs = ['proguard-spec.pro'],)");

    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:hello");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    Artifact optimizedResourceApk = getFirstArtifactEndingWith(artifacts, "hello_optimized.ap_");
    assertThat(optimizedResourceApk).isNull();

    // Verify that the optimized APK is not an input to build the unsigned APK.
    SpawnAction apkAction =
        getGeneratingSpawnAction(getFirstArtifactEndingWith(artifacts, "hello_unsigned.apk"));
    assertThat(apkAction.getMnemonic()).isEqualTo("ApkBuilder");
    assertThat(hasInput(apkAction, "hello_optimized.ap_")).isFalse();
  }

  @Test
  public void testResourceCycleShrinkingWithoutResourceShinking() throws Exception {
    useConfiguration("--experimental_android_resource_cycle_shrinking=true");
    checkError(
        "java/a",
        "a",
        "resource cycle shrinking can only be enabled when resource shrinking is enabled",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        "    shrink_resources = 0,",
        ")");
  }

  @Test
  public void testResourceShrinking_requiresProguard() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               shrink_resources = 1,)");

    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:hello");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    assertThat(artifacts)
        .containsNoneOf(
            getFirstArtifactEndingWith(artifacts, "shrunk.jar"),
            getFirstArtifactEndingWith(artifacts, "shrunk.ap_"));
  }

  @Test
  public void testProguardExtraOutputs() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");
    ConfiguredTarget output = getConfiguredTarget("//java/com/google/android/hello:b");

    // Checks that ProGuard is called with the appropriate options.
    Artifact a = getFirstArtifactEndingWith(getFilesToBuild(output), "_proguard.jar");
    SpawnAction action = getGeneratingSpawnAction(a);
    List<String> args = getGeneratingSpawnActionArgs(a);

    // Assert that the ProGuard executable set in the android_sdk rule appeared in the command-line
    // of the SpawnAction that generated the _proguard.jar.
    assertThat(args)
        .containsAtLeast(
            getProguardBinary().getExecPathString(),
            "-injars",
            execPathEndingWith(action.getInputs(), "b_deploy.jar"),
            "-printseeds",
            execPathEndingWith(action.getOutputs(), "b_proguard.seeds"),
            "-printusage",
            execPathEndingWith(action.getOutputs(), "b_proguard.usage"))
        .inOrder();

    // Checks that the output files are produced.
    assertProguardUsed(output);
    assertThat(getBinArtifact("b_proguard.usage", output)).isNotNull();
    assertThat(getBinArtifact("b_proguard.seeds", output)).isNotNull();
  }

  @Test
  public void testProGuardExecutableMatchesConfiguration() throws Exception {
    scratch.file(
        "java/com/google/devtools/build/jkrunchy/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_binary(name = 'jkrunchy',",
        "            srcs = glob(['*.java']),",
        "            main_class = 'com.google.devtools.build.jkrunchy.JKrunchyMain')");

    useConfiguration("--proguard_top=//java/com/google/devtools/build/jkrunchy:jkrunchy");

    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");

    ConfiguredTarget output = getConfiguredTarget("//java/com/google/android/hello:b_proguard.jar");
    assertProguardUsed(output);

    SpawnAction proguardAction =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(output), "_proguard.jar");
    Artifact jkrunchyExecutable =
        getHostConfiguredTarget("//java/com/google/devtools/build/jkrunchy")
            .getProvider(FilesToRunProvider.class)
            .getExecutable();
    assertWithMessage("ProGuard implementation was not correctly taken from the configuration")
        .that(proguardAction.getCommandFilename())
        .endsWith(jkrunchyExecutable.getOutputDirRelativePathString());
  }

  @Test
  public void enforceProguardFileExtension_disabled_allowsOtherExtensions() throws Exception {
    useConfiguration("--noenforce_proguard_file_extension");
    scratch.file(
        "java/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");

    ConfiguredTarget unused = getConfiguredTarget("//java/android/hello:b");
  }

  @Test
  public void enforceProguardFileExtension_enabled_disallowsOtherExtensions() throws Exception {
    useConfiguration("--enforce_proguard_file_extension");
    scratch.file(
        "java/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");

    AssertionError assertionError =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//java/android/hello:b"));

    assertThat(assertionError).hasMessageThat().contains("These files do not end in .pgcfg");
  }

  @Test
  public void enforceProguardFileExtension_enabled_allowsPgcfg() throws Exception {
    useConfiguration("--enforce_proguard_file_extension");
    scratch.file(
        "java/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = [':proguard.pgcfg'])");

    ConfiguredTarget unused = getConfiguredTarget("//java/android/hello:b");
  }

  @Test
  public void enforceProguardFileExtension_enabled_ignoresThirdParty() throws Exception {
    useConfiguration("--enforce_proguard_file_extension");
    scratch.file(
        "third_party/bar/BUILD", "licenses(['unencumbered'])", "exports_files(['proguard.pro'])");
    scratch.file(
        "java/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['//third_party/bar:proguard.pro'])");

    ConfiguredTarget unused = getConfiguredTarget("//java/android/hello:b");
  }

  @Test
  public void testNeverlinkTransitivity() throws Exception {
    useConfiguration("--android_fixed_resource_neverlinking");

    scratch.file(
        "java/com/google/android/neversayneveragain/BUILD",
        "android_library(name = 'l1',",
        "                srcs = ['l1.java'],",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/values/resource.xml'])",
        "android_library(name = 'l2',",
        "                srcs = ['l2.java'],",
        "                deps = [':l1'],",
        "                neverlink = 1)",
        "android_library(name = 'l3',",
        "                srcs = ['l3.java'],",
        "                deps = [':l2'])",
        "android_library(name = 'l4',",
        "                srcs = ['l4.java'],",
        "                deps = [':l1'])",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               deps = [':l2'],",
        "               manifest = 'AndroidManifest.xml')",
        "android_binary(name = 'b2',",
        "               srcs = ['b2.java'],",
        "               deps = [':l3'],",
        "               manifest = 'AndroidManifest.xml')",
        "android_binary(name = 'b3',",
        "               srcs = ['b3.java'],",
        "               deps = [':l3', ':l4'],",
        "               manifest = 'AndroidManifest.xml')");
    ConfiguredTarget b1 = getConfiguredTarget("//java/com/google/android/neversayneveragain:b1");
    Action b1DeployAction =
        actionsTestUtil()
            .getActionForArtifactEndingWith(
                actionsTestUtil().artifactClosureOf(getFilesToBuild(b1)), "b1_deploy.jar");
    List<String> b1Inputs = prettyArtifactNames(b1DeployAction.getInputs());

    assertThat(b1Inputs)
        .containsNoneOf(
            "java/com/google/android/neversayneveragain/libl1.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libl2.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libl3.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libl4.jar_desugared.jar");
    assertThat(b1Inputs)
        .contains("java/com/google/android/neversayneveragain/libb1.jar_desugared.jar");
    assertThat(
            resourceInputPaths(
                "java/com/google/android/neversayneveragain", getValidatedResources(b1)))
        .doesNotContain("res/values/resource.xml");

    ConfiguredTarget b2 = getConfiguredTarget("//java/com/google/android/neversayneveragain:b2");
    Action b2DeployAction =
        actionsTestUtil()
            .getActionForArtifactEndingWith(
                actionsTestUtil().artifactClosureOf(getFilesToBuild(b2)), "b2_deploy.jar");
    List<String> b2Inputs = prettyArtifactNames(b2DeployAction.getInputs());

    assertThat(b2Inputs)
        .containsNoneOf(
            "java/com/google/android/neversayneveragain/libl1.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libl2.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libl4.jar_desugared.jar");
    assertThat(b2Inputs)
        .containsAtLeast(
            "java/com/google/android/neversayneveragain/_dx/l3/libl3.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libb2.jar_desugared.jar");
    assertThat(
            resourceInputPaths(
                "java/com/google/android/neversayneveragain", getValidatedResources(b2)))
        .doesNotContain("res/values/resource.xml");

    ConfiguredTarget b3 = getConfiguredTarget("//java/com/google/android/neversayneveragain:b3");
    Action b3DeployAction =
        actionsTestUtil()
            .getActionForArtifactEndingWith(
                actionsTestUtil().artifactClosureOf(getFilesToBuild(b3)), "b3_deploy.jar");
    List<String> b3Inputs = prettyArtifactNames(b3DeployAction.getInputs());

    assertThat(b3Inputs)
        .containsAtLeast(
            "java/com/google/android/neversayneveragain/_dx/l1/libl1.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/_dx/l3/libl3.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/_dx/l4/libl4.jar_desugared.jar",
            "java/com/google/android/neversayneveragain/libb3.jar_desugared.jar");
    assertThat(b3Inputs)
        .doesNotContain("java/com/google/android/neversayneveragain/libl2.jar_desugared.jar");
    assertThat(
            resourceInputPaths(
                "java/com/google/android/neversayneveragain", getValidatedResources(b3)))
        .contains("res/values/resource.xml");
  }

  @Test
  public void testDexopts() throws Exception {
    useConfiguration("--noincremental_dexing");
    checkDexopts("[ '--opt1', '--opt2' ]", ImmutableList.of("--opt1", "--opt2"));
  }

  @Test
  public void testDexoptsTokenization() throws Exception {
    useConfiguration("--noincremental_dexing");
    checkDexopts(
        "[ '--opt1', '--opt2 tokenized' ]", ImmutableList.of("--opt1", "--opt2", "tokenized"));
  }

  @Test
  public void testDexoptsMakeVariableSubstitution() throws Exception {
    useConfiguration("--noincremental_dexing");
    checkDexopts("[ '--opt1', '$(COMPILATION_MODE)' ]", ImmutableList.of("--opt1", "fastbuild"));
  }

  private void checkDexopts(String dexopts, List<String> expectedArgs) throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(name = 'b',",
        "    srcs = ['dummy1.java'],",
        "    dexopts = " + dexopts + ",",
        "    manifest = 'AndroidManifest.xml')");

    // Include arguments that are always included.
    List<String> fixedArgs = ImmutableList.of("--num-threads=5");
    expectedArgs =
        new ImmutableList.Builder<String>().addAll(fixedArgs).addAll(expectedArgs).build();

    // Ensure that the args that immediately follow "--dex" match the expectation.
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android:b");
    List<String> args =
        getGeneratingSpawnActionArgs(
            ActionsTestUtil.getFirstArtifactEndingWith(
                actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)), "classes.dex"));
    int start = args.indexOf("--dex") + 1;
    assertThat(start).isNotEqualTo(0);
    int end = Math.min(args.size(), start + expectedArgs.size());
    assertThat(args.subList(start, end)).isEqualTo(expectedArgs);
  }

  @Test
  public void testDexMainListOpts() throws Exception {
    checkDexMainListOpts("[ '--opt1', '--opt2' ]", "--opt1", "--opt2");
  }

  @Test
  public void testDexMainListOptsTokenization() throws Exception {
    checkDexMainListOpts("[ '--opt1', '--opt2 tokenized' ]", "--opt1", "--opt2", "tokenized");
  }

  @Test
  public void testDexMainListOptsMakeVariableSubstitution() throws Exception {
    checkDexMainListOpts("[ '--opt1', '$(COMPILATION_MODE)' ]", "--opt1", "fastbuild");
  }

  private void checkDexMainListOpts(String mainDexListOpts, String... expectedArgs)
      throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(name = 'b',",
        "    srcs = ['dummy1.java'],",
        "    multidex = \"legacy\",",
        "    main_dex_list_opts = " + mainDexListOpts + ",",
        "    manifest = 'AndroidManifest.xml')");

    // Ensure that the args that immediately follow the main class in the shell command
    // match the expectation.
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android:b");
    List<String> args =
        getGeneratingSpawnActionArgs(
            ActionsTestUtil.getFirstArtifactEndingWith(
                actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)), "main_dex_list.txt"));

    // args: [ "bash", "-c", "java -cp dx.jar main opts other" ]
    MoreAsserts.assertContainsSublist(args, expectedArgs);
  }

  @Test
  public void omitResourcesInfoProviderFromAndroidBinary_enabled() throws Exception {
    useConfiguration("--experimental_omit_resources_info_provider_from_android_binary");
    ConfiguredTarget binary =
        scratchConfiguredTarget(
            "java/com/pkg/myapp",
            "myapp",
            "android_binary(",
            "  name = 'myapp',",
            "  manifest = 'AndroidManifest.xml',",
            "  resource_files = glob(['res/**/*']),",
            ")");

    assertThat(binary.get(AndroidResourcesInfo.PROVIDER)).isNull();
  }

  @Test
  public void omitResourcesInfoProviderFromAndroidBinary_disabled() throws Exception {
    useConfiguration("--noexperimental_omit_resources_info_provider_from_android_binary");
    ConfiguredTarget binary =
        scratchConfiguredTarget(
            "java/com/pkg/myapp",
            "myapp",
            "android_binary(",
            "  name = 'myapp',",
            "  manifest = 'AndroidManifest.xml',",
            "  resource_files = glob(['res/**/*']),",
            ")");

    assertThat(binary.get(AndroidResourcesInfo.PROVIDER)).isNotNull();
  }

  @Test
  public void testResourceConfigurationFilters() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(name = 'b',",
        "    srcs = ['dummy1.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_configuration_filters = [ 'en', 'fr'],)");

    // Ensure that the args are present
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android:b");
    List<String> args = resourceArguments(getValidatedResources(binary));
    assertThat(flagValue("--resourceConfigs", args)).contains("en,fr");
  }

  /** Test that resources are not filtered in analysis under aapt2. */
  @Test
  public void testFilteredResourcesFilteringAapt2() throws Exception {
    List<String> resources =
        ImmutableList.of("res/values/foo.xml", "res/values-en/foo.xml", "res/values-fr/foo.xml");
    String dir = "java/r/android";

    ConfiguredTarget binary =
        scratchConfiguredTarget(
            dir,
            "r",
            "android_binary(name = 'r',",
            "  manifest = 'AndroidManifest.xml',",
            "  resource_configuration_filters = ['', 'en, es, '],",
            "  densities = ['hdpi, , ', 'xhdpi'],",
            "  resource_files = ['" + Joiner.on("', '").join(resources) + "'])");
    ValidatedAndroidResources directResources =
        getValidatedResources(binary, /* transitive= */ false);

    // Validate that the AndroidResourceProvider for this binary contains all values.
    assertThat(resourceContentsPaths(dir, directResources)).containsExactlyElementsIn(resources);

    // Validate that the input to resource processing contains all values.
    assertThat(resourceInputPaths(dir, directResources)).containsAtLeastElementsIn(resources);

    // Validate that the filters are correctly passed to the resource processing action
    // This includes trimming whitespace and ignoring empty filters.
    assertThat(resourceArguments(directResources)).contains("en,es");
    assertThat(resourceArguments(directResources)).contains("hdpi,xhdpi");
  }

  @Test
  public void testFilterResourcesPseudolocalesPropagated() throws Exception {
    String dir = "java/r/android";
    ConfiguredTarget binary =
        scratchConfiguredTarget(
            dir,
            "bin",
            "android_binary(name = 'bin',",
            "  resource_files = glob(['res/**']),",
            "  resource_configuration_filters = ['en', 'en-rXA', 'ar-rXB'],",
            "  manifest = 'AndroidManifest.xml')");

    List<String> resourceProcessingArgs =
        getGeneratingSpawnActionArgs(getValidatedResources(binary).getRTxt());

    assertThat(resourceProcessingArgs).containsAtLeast("--resourceConfigs", "ar-rXB,en,en-rXA");
  }

  /**
   * Gets the paths of matching artifacts contained within a resource container
   *
   * @param dir the directory to look for artifacts in
   * @param resource the container that contains eligible artifacts
   * @return the paths to all artifacts from the input that are contained within the given
   *     directory, relative to that directory.
   */
  private List<String> resourceContentsPaths(String dir, ValidatedAndroidResources resource) {
    return pathsToArtifacts(dir, resource.getArtifacts());
  }

  /**
   * Gets the paths of matching artifacts that are used as input to resource processing
   *
   * @param dir the directory to look for artifacts in
   * @param resource the output from the resource processing that uses these artifacts as inputs
   * @return the paths to all artifacts used as inputs to resource processing that are contained
   *     within the given directory, relative to that directory.
   */
  private List<String> resourceInputPaths(String dir, ValidatedAndroidResources resource) {
    return pathsToArtifacts(dir, resourceGeneratingAction(resource).getInputs().toList());
  }

  /**
   * Gets the paths of matching artifacts from an iterable
   *
   * @param dir the directory to look for artifacts in
   * @param artifacts all available artifacts
   * @return the paths to all artifacts from the input that are contained within the given
   *     directory, relative to that directory.
   */
  private List<String> pathsToArtifacts(String dir, Iterable<Artifact> artifacts) {
    List<String> paths = new ArrayList<>();

    Path containingDir = rootDirectory;
    for (String part : dir.split("/")) {
      containingDir = containingDir.getChild(part);
    }

    for (Artifact a : artifacts) {
      if (a.getPath().startsWith(containingDir)) {
        paths.add(a.getPath().relativeTo(containingDir).toString());
      }
    }

    return paths;
  }

  @Test
  public void testInheritedRNotInRuntimeJars() throws Exception {
    String dir = "java/r/android/";
    scratch.file(
        dir + "BUILD",
        "android_library(name = 'sublib',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res3/**']),",
        "                srcs =['sublib.java'],",
        "                )",
        "android_library(name = 'lib',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res2/**']),",
        "                deps = [':sublib'],",
        "                srcs =['lib.java'],",
        "                )",
        "android_binary(name = 'bin',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "               deps = [':lib'],",
        "               srcs =['bin.java'],",
        "               )");

    Action deployJarAction =
        getGeneratingAction(
            getFileConfiguredTarget("//java/r/android:bin_deploy.jar").getArtifact());
    List<String> inputs = ActionsTestUtil.baseArtifactNames(deployJarAction.getInputs());

    assertThat(inputs)
        .containsAtLeast(
            "libsublib.jar_desugared.jar",
            "liblib.jar_desugared.jar",
            "libbin.jar_desugared.jar",
            "bin_resources.jar_desugared.jar");
    assertThat(inputs)
        .containsNoneOf("lib_resources.jar_desugared.jar", "sublib_resources.jar_desugared.jar");
  }

  @Test
  public void testLocalResourcesUseRClassGenerator() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_library(name = 'lib',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res2/**']),",
        "                )",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "               deps = [':lib'],",
        "               )");
    scratch.file(
        "java/r/android/res2/values/strings.xml",
        "<resources><string name = 'lib_string'>Libs!</string></resources>");
    scratch.file(
        "java/r/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    Artifact jar = getResourceClassJar(getConfiguredTargetAndData("//java/r/android:r"));
    assertThat(getGeneratingAction(jar).getMnemonic()).isEqualTo("RClassGenerator");
    assertThat(getGeneratingSpawnActionArgs(jar))
        .containsAtLeast("--primaryRTxt", "--primaryManifest", "--library", "--classJarOutput");
  }

  @Test
  public void testLocalResourcesUseRClassGeneratorNoLibraries() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "               )");
    scratch.file(
        "java/r/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    Artifact jar = getResourceClassJar(getConfiguredTargetAndData("//java/r/android:r"));
    assertThat(getGeneratingAction(jar).getMnemonic()).isEqualTo("RClassGenerator");
    List<String> args = getGeneratingSpawnActionArgs(jar);
    assertThat(args).containsAtLeast("--primaryRTxt", "--primaryManifest", "--classJarOutput");
    assertThat(args).doesNotContain("--libraries");
  }

  @Test
  public void testUseRClassGeneratorCustomPackage() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_library(name = 'lib',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res2/**']),",
        "                custom_package = 'com.lib.custom',",
        "                )",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "               custom_package = 'com.binary.custom',",
        "               deps = [':lib'],",
        "               )");
    scratch.file(
        "java/r/android/res2/values/strings.xml",
        "<resources><string name = 'lib_string'>Libs!</string></resources>");
    scratch.file(
        "java/r/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    ConfiguredTargetAndData binary = getConfiguredTargetAndData("//java/r/android:r");
    Artifact jar = getResourceClassJar(binary);
    assertThat(getGeneratingAction(jar).getMnemonic()).isEqualTo("RClassGenerator");
    List<String> args = getGeneratingSpawnActionArgs(jar);
    assertThat(args)
        .containsAtLeast(
            "--primaryRTxt",
            "--primaryManifest",
            "--library",
            "--classJarOutput",
            "--packageForR",
            "com.binary.custom");
  }

  @Test
  public void testUseRClassGeneratorMultipleDeps() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_library(name = 'lib1',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res1/**']),",
        "                )",
        "android_library(name = 'lib2',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res2/**']),",
        "                )",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "               deps = [':lib1', ':lib2'],",
        "               )");
    ConfiguredTargetAndData binary = getConfiguredTargetAndData("//java/r/android:r");
    Artifact jar = getResourceClassJar(binary);
    assertThat(getGeneratingAction(jar).getMnemonic()).isEqualTo("RClassGenerator");
    List<String> args = getGeneratingSpawnActionArgs(jar);

    AndroidResourcesInfo resourcesInfo =
        binary.getConfiguredTarget().get(AndroidResourcesInfo.PROVIDER);
    assertThat(resourcesInfo.getTransitiveAndroidResources().toList()).hasSize(2);
    ValidatedAndroidResources firstDep =
        resourcesInfo.getTransitiveAndroidResources().toList().get(0);
    ValidatedAndroidResources secondDep =
        resourcesInfo.getTransitiveAndroidResources().toList().get(1);

    assertThat(args)
        .containsAtLeast(
            "--primaryRTxt",
            "--primaryManifest",
            "--library",
            firstDep.getAapt2RTxt().getExecPathString()
                + ","
                + firstDep.getManifest().getExecPathString(),
            "--library",
            secondDep.getAapt2RTxt().getExecPathString()
                + ","
                + secondDep.getManifest().getExecPathString(),
            "--classJarOutput")
        .inOrder();
  }

  @Test
  public void useRTxtFromMergedResourcesForFinalRClasses() throws Exception {
    useConfiguration("--experimental_use_rtxt_from_merged_resources");
    scratch.file(
        "java/pkg/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['B.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")",
        "android_binary(",
        "    name = 'bin',",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [ ':lib' ],",
        ")");

    ConfiguredTargetAndData bin = getConfiguredTargetAndData("//java/pkg:bin");
    ConfiguredTarget lib = getDirectPrerequisite(bin.getConfiguredTarget(), "//java/pkg:lib");
    ValidatedAndroidResources libResources =
        lib.get(AndroidResourcesInfo.PROVIDER).getDirectAndroidResources().toList().get(0);
    SpawnAction topLevelResourceClassAction = getGeneratingSpawnAction(getResourceClassJar(bin));

    // verify that the R.txt from creating the library-level resources.jar is also used for creating
    // the top-level resources.jar
    assertThat(getGeneratingSpawnAction(libResources.getClassJar()).getOutputs())
        .contains(libResources.getAapt2RTxt());
    MoreAsserts.assertContainsSublist(
        topLevelResourceClassAction.getArguments(),
        "--library",
        libResources.getAapt2RTxt().getExecPathString()
            + ","
            + libResources.getManifest().getExecPathString());

    // the "validation artifact" shouldn't be used for creating the top-level resources.jar,
    // but it's still fed as an pseudo-input to trigger validation.
    MoreAsserts.assertDoesNotContainSublist(
        topLevelResourceClassAction.getArguments(),
        "--library",
        libResources.getAapt2ValidationArtifact().getExecPathString()
            + ","
            + libResources.getManifest().getExecPathString());
    assertThat(topLevelResourceClassAction.getInputs().toList())
        .contains(libResources.getAapt2ValidationArtifact());
  }

  // (test for undesired legacy behavior)
  @Test
  public void doNotUseRTxtFromMergedResourcesForFinalRClasses() throws Exception {
    scratch.file(
        "java/pkg/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['B.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")",
        "android_binary(",
        "    name = 'bin',",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [ ':lib' ],",
        ")");

    ConfiguredTargetAndData bin = getConfiguredTargetAndData("//java/pkg:bin");
    ConfiguredTarget lib = getDirectPrerequisite(bin.getConfiguredTarget(), "//java/pkg:lib");
    ValidatedAndroidResources libResources =
        lib.get(AndroidResourcesInfo.PROVIDER).getDirectAndroidResources().toList().get(0);

    // use the "validation artifact" for creating the top-level resources.jar
    MoreAsserts.assertContainsSublist(
        getGeneratingSpawnActionArgs(getResourceClassJar(bin)),
        "--library",
        libResources.getAapt2ValidationArtifact().getExecPathString()
            + ","
            + libResources.getManifest().getExecPathString());
  }

  @Test
  public void testNoCrunchBinaryOnly() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/drawable-hdpi-v4/foo.png',",
        "                                 'res/drawable-hdpi-v4/bar.9.png'],",
        "               crunch_png = 0,",
        "               )");
    ConfiguredTarget binary = getConfiguredTarget("//java/r/android:r");
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(binary));
    assertThat(args).contains("--useAaptCruncher=no");
  }

  @Test
  public void testDoCrunch() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/drawable-hdpi-v4/foo.png',",
        "                                 'res/drawable-hdpi-v4/bar.9.png'],",
        "               crunch_png = 1,",
        "               )");
    ConfiguredTarget binary = getConfiguredTarget("//java/r/android:r");
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(binary));
    assertThat(args).doesNotContain("--useAaptCruncher=no");
  }

  @Test
  public void testDoCrunchDefault() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/drawable-hdpi-v4/foo.png',",
        "                                 'res/drawable-hdpi-v4/bar.9.png'],",
        "               )");
    ConfiguredTarget binary = getConfiguredTarget("//java/r/android:r");
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(binary));
    assertThat(args).doesNotContain("--useAaptCruncher=no");
  }

  @Test
  public void testNoCrunchWithAndroidLibraryNoBinaryResources() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_library(name = 'resources',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/values/strings.xml',",
        "                                  'res/drawable-hdpi-v4/foo.png',",
        "                                  'res/drawable-hdpi-v4/bar.9.png'],",
        "               )",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               deps = [':resources'],",
        "               crunch_png = 0,",
        "               )");
    ConfiguredTarget binary = getConfiguredTarget("//java/r/android:r");
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(binary));
    assertThat(args).contains("--useAaptCruncher=no");
  }

  @Test
  public void testNoCrunchWithMultidexNative() throws Exception {
    scratch.file(
        "java/r/android/BUILD",
        "android_library(name = 'resources',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/values/strings.xml',",
        "                                  'res/drawable-hdpi-v4/foo.png',",
        "                                  'res/drawable-hdpi-v4/bar.9.png'],",
        "               )",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               deps = [':resources'],",
        "               multidex = 'native',",
        "               crunch_png = 0,",
        "               )");
    ConfiguredTarget binary = getConfiguredTarget("//java/r/android:r");
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(binary));
    assertThat(args).contains("--useAaptCruncher=no");
  }

  @Test
  public void testZipaligned() throws Exception {
    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");
    Artifact a =
        ActionsTestUtil.getFirstArtifactEndingWith(
            actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)), "zipaligned_app.apk");
    SpawnAction action = getGeneratingSpawnAction(a);

    assertThat(action.getMnemonic()).isEqualTo("AndroidZipAlign");

    List<String> arguments = getGeneratingSpawnActionArgs(a);
    assertThat(arguments).contains("-p");
    assertThat(arguments).contains("4");

    Artifact zipAlignTool = getFirstArtifactEndingWith(action.getInputs(), "/zipalign");
    assertThat(arguments).contains(zipAlignTool.getExecPathString());

    Artifact unsignedApk = getFirstArtifactEndingWith(action.getInputs(), "/app_unsigned.apk");
    assertThat(arguments).contains(unsignedApk.getExecPathString());

    Artifact zipalignedApk = getFirstArtifactEndingWith(action.getOutputs(), "/zipaligned_app.apk");
    assertThat(arguments).contains(zipalignedApk.getExecPathString());
  }

  @Test
  public void testDeployInfo() throws Exception {
    ConfiguredTarget binary = getConfiguredTarget("//java/android:app");
    NestedSet<Artifact> outputGroup = getOutputGroup(binary, "android_deploy_info");
    Artifact deployInfoArtifact =
        ActionsTestUtil.getFirstArtifactEndingWith(outputGroup, "/deploy_info.deployinfo.pb");
    assertThat(deployInfoArtifact).isNotNull();
    AndroidDeployInfo deployInfo = getAndroidDeployInfo(deployInfoArtifact);
    assertThat(deployInfo).isNotNull();
    assertThat(deployInfo.getMergedManifest().getExecRootPath()).endsWith("/AndroidManifest.xml");
    assertThat(deployInfo.getAdditionalMergedManifestsList()).isEmpty();
    assertThat(deployInfo.getApksToDeploy(0).getExecRootPath()).endsWith("/app.apk");
  }

  /**
   * Internal helper method: checks that dex sharding input and output is correct for different
   * combinations of multidex mode and build with and without proguard.
   */
  private void internalTestDexShardStructure(
      MultidexMode multidexMode, boolean proguard, String nonProguardSuffix) throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//java/a:a");
    assertNoEvents();
    Action shardAction = getGeneratingAction(getBinArtifact("_dx/a/shard1.jar", target));

    // Verify command line arguments
    List<String> arguments = ((SpawnAction) shardAction).getRemainingArguments();
    List<String> expectedArguments = new ArrayList<>();
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(target));
    Artifact shard1 = getFirstArtifactEndingWith(artifacts, "shard1.jar");
    Artifact shard2 = getFirstArtifactEndingWith(artifacts, "shard2.jar");
    Artifact resourceJar = getFirstArtifactEndingWith(artifacts, "/java_resources.jar");
    expectedArguments.add("--output_jar");
    expectedArguments.add(shard1.getExecPathString());
    expectedArguments.add("--output_jar");
    expectedArguments.add(shard2.getExecPathString());
    expectedArguments.add("--output_resources");
    expectedArguments.add(resourceJar.getExecPathString());
    if (multidexMode == MultidexMode.LEGACY) {
      Artifact mainDexList = getFirstArtifactEndingWith(artifacts, "main_dex_list.txt");
      expectedArguments.add("--main_dex_filter");
      expectedArguments.add(mainDexList.getExecPathString());
    }
    if (!proguard) {
      expectedArguments.add("--input_jar");
      expectedArguments.add(
          getFirstArtifactEndingWith(artifacts, "a_resources.jar" + nonProguardSuffix)
              .getExecPathString());
    }
    Artifact inputJar;
    if (proguard) {
      inputJar = getFirstArtifactEndingWith(artifacts, "a_proguard.jar");
    } else {
      inputJar = getFirstArtifactEndingWith(artifacts, "liba.jar" + nonProguardSuffix);
    }
    expectedArguments.add("--input_jar");
    expectedArguments.add(inputJar.getExecPathString());
    assertThat(arguments).containsExactlyElementsIn(expectedArguments).inOrder();

    // Verify input and output artifacts
    List<String> shardOutputs = ActionsTestUtil.baseArtifactNames(shardAction.getOutputs());
    List<String> shardInputs = ActionsTestUtil.baseArtifactNames(shardAction.getInputs());
    assertThat(shardOutputs).containsExactly("shard1.jar", "shard2.jar", "java_resources.jar");
    if (multidexMode == MultidexMode.LEGACY) {
      assertThat(shardInputs).contains("main_dex_list.txt");
    } else {
      assertThat(shardInputs).doesNotContain("main_dex_list.txt");
    }
    if (proguard) {
      assertThat(shardInputs).contains("a_proguard.jar");
      assertThat(shardInputs).doesNotContain("liba.jar" + nonProguardSuffix);
    } else {
      assertThat(shardInputs).contains("liba.jar" + nonProguardSuffix);
      assertThat(shardInputs).doesNotContain("a_proguard.jar");
    }
    assertThat(shardInputs).doesNotContain("a_deploy.jar");

    // Verify that dex compilation is followed by the correct merge operation
    Action apkAction =
        getGeneratingAction(
            getFirstArtifactEndingWith(
                actionsTestUtil().artifactClosureOf(getFilesToBuild(target)),
                "compressed_a_unsigned.apk"));
    Action mergeAction =
        getGeneratingAction(getFirstArtifactEndingWith(apkAction.getInputs(), "classes.dex.zip"));
    Iterable<Artifact> dexShards =
        Iterables.filter(
            mergeAction.getInputs().toList(), ActionsTestUtil.getArtifactSuffixMatcher(".dex.zip"));
    assertThat(ActionsTestUtil.baseArtifactNames(dexShards))
        .containsExactly("shard1.dex.zip", "shard2.dex.zip");
  }

  @Test
  public void testDexShardingNeedsMultidex() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    manifest='AndroidManifest.xml')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//java/a:a");
    assertContainsEvent(".dex sharding is only available in multidex mode");
  }

  @Test
  public void testDexShardingDoesNotWorkWithManualMultidex() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='manual_main_dex',",
        "    main_dex_list='main_dex_list.txt',",
        "    manifest='AndroidManifest.xml')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//java/a:a");
    assertContainsEvent(".dex sharding is not available in manual multidex mode");
  }

  @Test
  public void testDexShardingLegacyStructure() throws Exception {
    useConfiguration("--noincremental_dexing");
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='legacy',",
        "    manifest='AndroidManifest.xml')");

    internalTestDexShardStructure(MultidexMode.LEGACY, false, "_desugared.jar");
  }

  @Test
  public void testDexShardingNativeStructure_withNoDesugaring() throws Exception {
    useConfiguration("--noexperimental_desugar_for_android", "--noincremental_dexing");
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='native',",
        "    manifest='AndroidManifest.xml')");

    internalTestDexShardStructure(MultidexMode.NATIVE, false, "");
  }

  @Test
  public void testDexShardingNativeStructure() throws Exception {
    useConfiguration("--noincremental_dexing");
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='native',",
        "    manifest='AndroidManifest.xml')");

    internalTestDexShardStructure(MultidexMode.NATIVE, false, "_desugared.jar");
  }

  @Test
  public void testDexShardingLegacyAndProguardStructure_withNoDesugaring() throws Exception {
    useConfiguration("--noexperimental_desugar_for_android");
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='legacy',",
        "    manifest='AndroidManifest.xml',",
        "    proguard_specs=['proguard.cfg'])");

    internalTestDexShardStructure(MultidexMode.LEGACY, true, "");
  }

  @Test
  public void testDexShardingLegacyAndProguardStructure() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='legacy',",
        "    manifest='AndroidManifest.xml',",
        "    proguard_specs=['proguard.cfg'])");

    internalTestDexShardStructure(MultidexMode.LEGACY, true, "_desugared.jar");
  }

  @Test
  public void testDexShardingNativeAndProguardStructure() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='native',",
        "    manifest='AndroidManifest.xml',",
        "    proguard_specs=['proguard.cfg'])");

    internalTestDexShardStructure(MultidexMode.NATIVE, true, "");
  }

  @Test
  public void testIncrementalApkAndProguardBuildStructure() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name='a',",
        "    srcs=['A.java'],",
        "    dex_shards=2,",
        "    multidex='native',",
        "    manifest='AndroidManifest.xml',",
        "    proguard_specs=['proguard.cfg'])");

    ConfiguredTarget target = getConfiguredTarget("//java/a:a");
    Action shardAction = getGeneratingAction(getBinArtifact("_dx/a/shard1.jar", target));
    List<String> shardOutputs = ActionsTestUtil.baseArtifactNames(shardAction.getOutputs());
    assertThat(shardOutputs).contains("java_resources.jar");
    assertThat(shardOutputs).doesNotContain("a_deploy.jar");
  }

  @Test
  public void testManualMainDexBuildStructure() throws Exception {
    checkError(
        "java/foo",
        "maindex_nomultidex",
        "Both \"main_dex_list\" and \"multidex='manual_main_dex'\" must be specified",
        "android_binary(",
        "    name = 'maindex_nomultidex',",
        "    srcs = ['a.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'manual_main_dex')");
  }

  @Test
  public void testMainDexListLegacyMultidex() throws Exception {
    checkError(
        "java/foo",
        "maindex_nomultidex",
        "Both \"main_dex_list\" and \"multidex='manual_main_dex'\" must be specified",
        "android_binary(",
        "    name = 'maindex_nomultidex',",
        "    srcs = ['a.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'legacy',",
        "    main_dex_list = 'main_dex_list.txt')");
  }

  @Test
  public void testMainDexListNativeMultidex() throws Exception {
    checkError(
        "java/foo",
        "maindex_nomultidex",
        "Both \"main_dex_list\" and \"multidex='manual_main_dex'\" must be specified",
        "android_binary(",
        "    name = 'maindex_nomultidex',",
        "    srcs = ['a.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'native',",
        "    main_dex_list = 'main_dex_list.txt')");
  }

  @Test
  public void testMainDexListNoMultidex() throws Exception {
    checkError(
        "java/foo",
        "maindex_nomultidex",
        "Both \"main_dex_list\" and \"multidex='manual_main_dex'\" must be specified",
        "android_binary(",
        "    name = 'maindex_nomultidex',",
        "    srcs = ['a.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    main_dex_list = 'main_dex_list.txt')");
  }

  @Test
  public void testMainDexListWithAndroidSdk() throws Exception {
    scratch.file(
        "sdk/BUILD",
        "android_sdk(",
        "    name = 'sdk',",
        "    aapt = 'aapt',",
        "    aapt2 = 'aapt2',",
        "    adb = 'adb',",
        "    aidl = 'aidl',",
        "    android_jar = 'android.jar',",
        "    apksigner = 'apksigner',",
        "    dx = 'dx',",
        "    framework_aidl = 'framework_aidl',",
        "    main_dex_classes = 'main_dex_classes',",
        "    main_dex_list_creator = 'main_dex_list_creator',",
        "    proguard = 'proguard',",
        "    shrinked_android_jar = 'shrinked_android_jar',",
        "    zipalign = 'zipalign',",
        "    tags = ['__ANDROID_RULES_MIGRATION__'])");

    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'legacy',",
        "    main_dex_list_opts = ['--hello', '--world'])");

    useConfiguration("--android_sdk=//sdk:sdk");
    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    Artifact mainDexList =
        ActionsTestUtil.getFirstArtifactEndingWith(
            actionsTestUtil().artifactClosureOf(getFilesToBuild(a)), "main_dex_list.txt");
    List<String> args = getGeneratingSpawnActionArgs(mainDexList);
    assertThat(args).containsAtLeast("--hello", "--world");
  }

  @Test
  public void testMainDexAaptGenerationSupported() throws Exception {
    useConfiguration("--android_sdk=//sdk:sdk", "--noincremental_dexing");
    scratch.file(
        "sdk/BUILD",
        "android_sdk(",
        "    name = 'sdk',",
        "    build_tools_version = '24.0.0',",
        "    aapt = 'aapt',",
        "    aapt2 = 'aapt2',",
        "    adb = 'adb',",
        "    aidl = 'aidl',",
        "    android_jar = 'android.jar',",
        "    apksigner = 'apksigner',",
        "    dx = 'dx',",
        "    framework_aidl = 'framework_aidl',",
        "    main_dex_classes = 'main_dex_classes',",
        "    main_dex_list_creator = 'main_dex_list_creator',",
        "    proguard = 'proguard',",
        "    shrinked_android_jar = 'shrinked_android_jar',",
        "    zipalign = 'zipalign',",
        "    tags = ['__ANDROID_RULES_MIGRATION__'])");

    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'legacy')");

    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    Artifact intermediateJar =
        artifactByPath(
            ImmutableList.of(getCompressedUnsignedApk(a)),
            ".apk",
            ".dex.zip",
            ".dex.zip",
            "main_dex_list.txt",
            "_intermediate.jar");
    List<String> args = getGeneratingSpawnActionArgs(intermediateJar);
    assertContainsSublist(
        args,
        ImmutableList.of(
            "-include",
            targetConfig.getBinFragment() + "/java/a/proguard/a/main_dex_a_proguard.cfg"));
  }

  @Test
  public void testMainDexGenerationWithoutProguardMap() throws Exception {
    useConfiguration("--noincremental_dexing");
    scratchConfiguredTarget(
        "java/foo",
        "abin",
        "android_binary(",
        "    name = 'abin',",
        "    srcs = ['a.java'],",
        "    proguard_specs = [],",
        "    manifest = 'AndroidManifest.xml',",
        "    multidex = 'legacy',)");
    ConfiguredTarget a = getConfiguredTarget("//java/foo:abin");
    Artifact intermediateJar =
        artifactByPath(
            ImmutableList.of(getCompressedUnsignedApk(a)),
            ".apk",
            ".dex.zip",
            ".dex.zip",
            "main_dex_list.txt",
            "_intermediate.jar");
    List<String> args = getGeneratingSpawnActionArgs(intermediateJar);
    MoreAsserts.assertDoesNotContainSublist(args, "-previousobfuscationmap");
  }

  // regression test for b/14288948
  @Test
  public void testEmptyListAsProguardSpec() throws Exception {
    scratch.file(
        "java/foo/BUILD",
        "android_binary(",
        "    name = 'abin',",
        "    srcs = ['a.java'],",
        "    proguard_specs = [],",
        "    manifest = 'AndroidManifest.xml')");
    Rule rule = getTarget("//java/foo:abin").getAssociatedRule();
    assertNoEvents();
    ImmutableList<String> implicitOutputFilenames =
        rule.getOutputFiles().stream().map(FileTarget::getName).collect(toImmutableList());
    assertThat(implicitOutputFilenames).doesNotContain("abin_proguard.jar");
  }

  @Test
  public void testConfigurableProguardSpecsEmptyList() throws Exception {
    scratch.file(
        "java/foo/BUILD",
        "android_binary(",
        "    name = 'abin',",
        "    srcs = ['a.java'],",
        "    proguard_specs = select({",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [],",
        "    }),",
        "    manifest = 'AndroidManifest.xml')");
    Rule rule = getTarget("//java/foo:abin").getAssociatedRule();
    assertNoEvents();
    ImmutableList<String> implicitOutputFilenames =
        rule.getOutputFiles().stream().map(FileTarget::getName).collect(toImmutableList());
    assertThat(implicitOutputFilenames).contains("abin_proguard.jar");
  }

  @Test
  public void testConfigurableProguardSpecsEmptyListWithMapping() throws Exception {
    scratchConfiguredTarget(
        "java/foo",
        "abin",
        "android_binary(",
        "    name = 'abin',",
        "    srcs = ['a.java'],",
        "    proguard_specs = select({",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [],",
        "    }),",
        "    proguard_generate_mapping = 1,",
        "    manifest = 'AndroidManifest.xml')");
    assertNoEvents();
  }

  @Test
  public void testResourcesWithConfigurationQualifier_localResources() throws Exception {
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = glob(['res/**']),",
        "                )");
    scratch.file(
        "java/android/resources/res/values-en/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file(
        "java/android/resources/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources:r");

    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(resource));

    assertPrimaryResourceDirs(ImmutableList.of("java/android/resources/res"), args);
  }

  @Test
  public void testResourcesInOtherPackage_exported_localResources() throws Exception {
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['//java/resources/other:res/values/strings.xml'],",
        "                )");
    scratch.file("java/resources/other/BUILD", "exports_files(['res/values/strings.xml'])");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources:r");

    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(resource));
    assertPrimaryResourceDirs(ImmutableList.of("java/resources/other/res"), args);
    assertNoEvents();
  }

  @Test
  public void testResourcesInOtherPackage_filegroup_localResources() throws Exception {
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['//java/other/resources:fg'],",
        "                )");
    scratch.file(
        "java/other/resources/BUILD",
        "filegroup(name = 'fg',",
        "          srcs = ['res/values/strings.xml'],",
        ")");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources:r");

    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(resource));
    assertPrimaryResourceDirs(ImmutableList.of("java/other/resources/res"), args);
    assertNoEvents();
  }

  @Test
  public void testResourcesInOtherPackage_filegroupWithExternalSources_localResources()
      throws Exception {
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = [':fg'],",
        "               )",
        "filegroup(name = 'fg',",
        "          srcs = ['//java/other/resources:res/values/strings.xml'])");
    scratch.file("java/other/resources/BUILD", "exports_files(['res/values/strings.xml'])");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources:r");

    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(resource));
    assertPrimaryResourceDirs(ImmutableList.of("java/other/resources/res"), args);
    assertNoEvents();
  }

  @Test
  public void testMultipleDependentResourceDirectories_localResources() throws Exception {
    scratch.file(
        "java/android/resources/d1/BUILD",
        "android_library(name = 'd1',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['d1-res/values/strings.xml'],",
        "                )");
    scratch.file(
        "java/android/resources/d2/BUILD",
        "android_library(name = 'd2',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['d2-res/values/strings.xml'],",
        "                )");
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['bin-res/values/strings.xml'],",
        "               deps = [",
        "                   '//java/android/resources/d1:d1','//java/android/resources/d2:d2'",
        "               ])");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources:r");

    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(resource));
    assertPrimaryResourceDirs(ImmutableList.of("java/android/resources/bin-res"), args);
    assertThat(getDirectDependentResourceDirs(args))
        .containsAtLeast("java/android/resources/d1/d1-res", "java/android/resources/d2/d2-res");
    assertNoEvents();
  }

  // Regression test for b/11924769
  @Test
  public void testResourcesInOtherPackage_doubleFilegroup_localResources() throws Exception {
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = [':fg'],",
        "               )",
        "filegroup(name = 'fg',",
        "          srcs = ['//java/other/resources:fg'])");
    scratch.file(
        "java/other/resources/BUILD",
        "filegroup(name = 'fg',",
        "          srcs = ['res/values/strings.xml'],",
        ")");
    ConfiguredTarget resource = getConfiguredTarget("//java/android/resources:r");

    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(resource));
    assertPrimaryResourceDirs(ImmutableList.of("java/other/resources/res"), args);
    assertNoEvents();
  }

  @Test
  public void testManifestMissingFails_localResources() throws Exception {
    checkError(
        "java/android/resources",
        "r",
        "manifest attribute of android_library rule //java/android/resources:r: manifest is "
            + "required when resource_files or assets are defined.",
        "filegroup(name = 'b')",
        "android_library(name = 'r',",
        "                resource_files = [':b'],",
        "                )");
  }

  @Test
  public void testResourcesDoesNotMatchDirectoryLayout_badFile_localResources() throws Exception {
    checkError(
        "java/android/resources",
        "r",
        "'java/android/resources/res/somefile.xml' is not in the expected resource directory "
            + "structure of <resource directory>/{"
            + Joiner.on(',').join(AndroidResources.RESOURCE_DIRECTORY_TYPES)
            + "}",
        "android_binary(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/somefile.xml', 'r/t/f/m/raw/fold']",
        "                )");
  }

  @Test
  public void testResourcesDoesNotMatchDirectoryLayout_badDirectory_localResources()
      throws Exception {
    checkError(
        "java/android/resources",
        "r",
        "'java/android/resources/res/other/somefile.xml' is not in the expected resource directory "
            + "structure of <resource directory>/{"
            + Joiner.on(',').join(AndroidResources.RESOURCE_DIRECTORY_TYPES)
            + "}",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/other/somefile.xml', 'r/t/f/m/raw/fold']",
        "               )");
  }

  @Test
  public void testResourcesNotUnderCommonDirectoryFails_localResources() throws Exception {
    checkError(
        "java/android/resources",
        "r",
        "'java/android/resources/r/t/f/m/raw/fold' (generated by '//java/android/resources:r/t/f/m/"
            + "raw/fold') is not in the same directory 'res' "
            + "(derived from java/android/resources/res/raw/speed). "
            + "All resources must share a common directory",
        "android_binary(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                resource_files = ['res/raw/speed', 'r/t/f/m/raw/fold']",
        "                )");
  }

  @Test
  public void testAssetsAndNoAssetsDirFails_localResources() throws Exception {
    scratch.file(
        "java/android/resources/assets/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    checkError(
        "java/android/resources",
        "r",
        "'assets' and 'assets_dir' should be either both empty or both non-empty",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               assets = glob(['assets/**']),",
        "               )");
  }

  @Test
  public void testAssetsDirAndNoAssetsFails_localResources() throws Exception {
    checkError(
        "java/cpp/android",
        "r",
        "'assets' and 'assets_dir' should be either both empty or both non-empty",
        "android_binary(name = 'r',",
        "               manifest = 'AndroidManifest.xml',",
        "               assets_dir = 'assets',",
        "                )");
  }

  @Test
  public void testAssetsNotUnderAssetsDirFails_localResources() throws Exception {
    checkError(
        "java/android/resources",
        "r",
        "'java/android/resources/r/t/f/m' (generated by '//java/android/resources:r/t/f/m') "
            + "is not beneath 'assets'",
        "android_binary(name = 'r',",
        "                manifest = 'AndroidManifest.xml',",
        "                assets_dir = 'assets',",
        "                assets = ['assets/valuable', 'r/t/f/m']",
        "                )");
  }

  @Test
  public void testFileLocation_localResources() throws Exception {
    scratch.file(
        "java/android/resources/BUILD",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               )");
    ConfiguredTarget r = getConfiguredTarget("//java/android/resources:r");
    assertThat(getFirstArtifactEndingWith(getFilesToBuild(r), ".apk").getRoot())
        .isEqualTo(getTargetConfiguration().getBinDirectory(RepositoryName.MAIN));
  }

  @Test
  public void testCustomPackage_localResources() throws Exception {
    scratch.file(
        "a/r/BUILD",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               custom_package = 'com.google.android.bar',",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               )");
    ConfiguredTarget r = getConfiguredTarget("//a/r:r");
    assertNoEvents();
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(r));
    assertContainsSublist(args, ImmutableList.of("--packageForR", "com.google.android.bar"));
  }

  @Test
  public void testCustomJavacopts() throws Exception {
    scratch.file("java/foo/A.java", "foo");
    scratch.file(
        "java/foo/BUILD",
        "android_binary(name = 'a', manifest = 'AndroidManifest.xml', ",
        "  srcs = ['A.java'], javacopts = ['-g:lines,source'])");

    Artifact deployJar = getFileConfiguredTarget("//java/foo:a_deploy.jar").getArtifact();
    Action deployAction = getGeneratingAction(deployJar);
    JavaCompileAction javacAction =
        (JavaCompileAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(deployAction.getInputs()), "liba.jar");

    assertThat(getJavacArguments(javacAction)).contains("-g:lines,source");
  }

  @Test
  public void testFixDepsToolFlag() throws Exception {
    useConfiguration("--experimental_fix_deps_tool=autofixer");

    scratch.file("java/foo/A.java", "foo");
    scratch.file(
        "java/foo/BUILD",
        "android_binary(name = 'a', manifest = 'AndroidManifest.xml', ",
        "  srcs = ['A.java'])");

    Iterable<String> commandLine =
        getJavacArguments(
            ((JavaCompileAction)
                actionsTestUtil()
                    .getActionForArtifactEndingWith(
                        actionsTestUtil()
                            .artifactClosureOf(
                                getGeneratingAction(
                                        getFileConfiguredTarget("//java/foo:a_deploy.jar")
                                            .getArtifact())
                                    .getInputs()),
                        "liba.jar")));

    assertThat(commandLine).containsAtLeast("--experimental_fix_deps_tool", "autofixer").inOrder();
  }

  @Test
  public void testFixDepsToolFlagEmpty() throws Exception {
    scratch.file("java/foo/A.java", "foo");
    scratch.file(
        "java/foo/BUILD",
        "android_binary(name = 'a', manifest = 'AndroidManifest.xml', ",
        "  srcs = ['A.java'])");

    Iterable<String> commandLine =
        getJavacArguments(
            ((JavaCompileAction)
                actionsTestUtil()
                    .getActionForArtifactEndingWith(
                        actionsTestUtil()
                            .artifactClosureOf(
                                getGeneratingAction(
                                        getFileConfiguredTarget("//java/foo:a_deploy.jar")
                                            .getArtifact())
                                    .getInputs()),
                        "liba.jar")));

    assertThat(commandLine).containsAtLeast("--experimental_fix_deps_tool", "add_dep").inOrder();
  }

  @Test
  public void testAndroidBinaryExportsJavaCompilationArgsProvider() throws Exception {

    scratch.file("java/foo/A.java", "foo");
    scratch.file(
        "java/foo/BUILD",
        "android_binary(name = 'a', manifest = 'AndroidManifest.xml', ",
        "  srcs = ['A.java'], javacopts = ['-g:lines,source'])");

    final JavaCompilationArgsProvider provider =
        JavaInfo.getProvider(
            JavaCompilationArgsProvider.class, getConfiguredTarget("//java/foo:a"));

    assertThat(provider).isNotNull();
  }

  @Test
  public void testNoApplicationId_localResources() throws Exception {
    scratch.file(
        "java/a/r/BUILD",
        "android_binary(name = 'r',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               )");
    ConfiguredTarget r = getConfiguredTarget("//java/a/r:r");
    assertNoEvents();
    List<String> args = getGeneratingSpawnActionArgs(getResourceApk(r));
    Truth.assertThat(args).doesNotContain("--applicationId");
  }

  @Test
  public void testDisallowPrecompiledJars() throws Exception {
    checkError(
        "java/precompiled",
        "binary",
        // messages:
        "does not produce any android_binary srcs files (expected .java or .srcjar)",
        // build file:
        "android_binary(name = 'binary',",
        "    manifest='AndroidManifest.xml',",
        "    srcs = [':jar'])",
        "filegroup(name = 'jar',",
        "    srcs = ['lib.jar'])");
  }

  @Test
  public void testDesugarJava8Libs_noProguard() throws Exception {
    useConfiguration("--experimental_desugar_java8_libs");
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  manifest = 'AndroidManifest.xml',",
        "  multidex = 'native',",
        ")");

    ConfiguredTarget top = getConfiguredTarget("//java/com/google/android:foo");
    Artifact artifact = getBinArtifact("_dx/foo/_final_classes.dex.zip", top);
    assertWithMessage("_final_classes.dex.zip").that(artifact).isNotNull();
    Action generatingAction = getGeneratingAction(artifact);
    assertThat(ActionsTestUtil.baseArtifactNames(generatingAction.getInputs()))
        .containsAtLeast("classes.dex.zip", /*canned*/ "java8_legacy.dex.zip");
  }

  @Test
  public void testDesugarJava8Libs_withProguard() throws Exception {
    useConfiguration("--experimental_desugar_java8_libs");
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  manifest = 'AndroidManifest.xml',",
        "  multidex = 'native',",
        "  proguard_specs = ['foo.cfg'],",
        ")");

    ConfiguredTarget top = getConfiguredTarget("//java/com/google/android:foo");
    Artifact artifact = getBinArtifact("_dx/foo/_final_classes.dex.zip", top);
    assertWithMessage("_final_classes.dex.zip").that(artifact).isNotNull();
    Action generatingAction = getGeneratingAction(artifact);
    assertThat(ActionsTestUtil.baseArtifactNames(generatingAction.getInputs()))
        .containsAtLeast("classes.dex.zip", /*built*/ "_java8_legacy.dex.zip");
  }

  @Test
  public void testDesugarJava8Libs_noMultidexError() throws Exception {
    useConfiguration("--experimental_desugar_java8_libs");
    checkError(
        /*packageName=*/ "java/com/google/android",
        /*ruleName=*/ "foo",
        /*expectedErrorMessage=*/ "multidex",
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  manifest = 'AndroidManifest.xml',",
        ")");
  }

  @Test
  public void testApplyProguardMapping() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  proguard_apply_mapping = 'proguard.map',",
        "  proguard_specs = ['foo.pro'],",
        "  manifest = 'AndroidManifest.xml',",
        ")");

    ConfiguredTarget ct = getConfiguredTarget("//java/com/google/android:foo");

    Artifact artifact = artifactByPath(getFilesToBuild(ct), "_proguard.jar");
    Action generatingAction = getGeneratingAction(artifact);
    assertThat(Artifact.asExecPaths(generatingAction.getInputs()))
        .contains("java/com/google/android/proguard.map");
    // Cannot use assertThat().containsAllOf().inOrder() as that does not assert that the elements
    // are consecutive.
    MoreAsserts.assertContainsSublist(
        getGeneratingSpawnActionArgs(artifact),
        "-applymapping",
        "java/com/google/android/proguard.map");
  }

  @Test
  public void testApplyProguardMappingWithNoSpec() throws Exception {
    checkError(
        "java/com/google/android",
        "foo",
        // messages:
        "'proguard_apply_mapping' can only be used when 'proguard_specs' is also set",
        // build file:
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  proguard_apply_mapping = 'proguard.map',",
        "  manifest = 'AndroidManifest.xml',",
        ")");
  }

  @Test
  public void testApplyProguardDictionary() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  proguard_apply_dictionary = 'dictionary.txt',",
        "  proguard_specs = ['foo.pro'],",
        "  manifest = 'AndroidManifest.xml',",
        ")");

    ConfiguredTarget ct = getConfiguredTarget("//java/com/google/android:foo");

    Artifact artifact = artifactByPath(getFilesToBuild(ct), "_proguard.jar");
    Action generatingAction = getGeneratingAction(artifact);
    assertThat(Artifact.asExecPaths(generatingAction.getInputs()))
        .contains("java/com/google/android/dictionary.txt");
    // Cannot use assertThat().containsAllOf().inOrder() as that does not assert that the elements
    // are consecutive.
    MoreAsserts.assertContainsSublist(
        getGeneratingSpawnActionArgs(artifact),
        "-obfuscationdictionary",
        "java/com/google/android/dictionary.txt");
    MoreAsserts.assertContainsSublist(
        getGeneratingSpawnActionArgs(artifact),
        "-classobfuscationdictionary",
        "java/com/google/android/dictionary.txt");
    MoreAsserts.assertContainsSublist(
        getGeneratingSpawnActionArgs(artifact),
        "-packageobfuscationdictionary",
        "java/com/google/android/dictionary.txt");
  }

  @Test
  public void testApplyProguardDictionaryWithNoSpec() throws Exception {
    checkError(
        "java/com/google/android",
        "foo",
        // messages:
        "'proguard_apply_dictionary' can only be used when 'proguard_specs' is also set",
        // build file:
        "android_binary(",
        "  name = 'foo',",
        "  srcs = ['foo.java'],",
        "  proguard_apply_dictionary = 'dictionary.txt',",
        "  manifest = 'AndroidManifest.xml',",
        ")");
  }

  @Test
  public void testFeatureFlagsAttributeSetsSelectInDependency() throws Exception {
    useConfiguration(
        "--experimental_dynamic_configs=notrim",
        "--enforce_transitive_configs_for_config_feature_flag");
    scratch.file(
        "java/com/foo/BUILD",
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
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  deps = [':lib'],",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        "  transitive_configs = [':flag1', ':flag2'],",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/foo");
    List<String> inputs =
        prettyArtifactNames(actionsTestUtil().artifactClosureOf(getFinalUnsignedApk(binary)));

    assertThat(inputs).containsAtLeast("java/com/foo/Flag1On.java", "java/com/foo/Flag2Off.java");
    assertThat(inputs).containsNoneOf("java/com/foo/Flag1Off.java", "java/com/foo/Flag2On.java");
  }

  @Test
  public void testFeatureFlagsAttributeSetsSelectInBinary() throws Exception {
    useConfiguration(
        "--experimental_dynamic_configs=notrim",
        "--enforce_transitive_configs_for_config_feature_flag");
    scratch.file(
        "java/com/foo/BUILD",
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
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = select({",
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
        prettyArtifactNames(actionsTestUtil().artifactClosureOf(getFinalUnsignedApk(binary)));

    assertThat(inputs).containsAtLeast("java/com/foo/Flag1On.java", "java/com/foo/Flag2Off.java");
    assertThat(inputs).containsNoneOf("java/com/foo/Flag1Off.java", "java/com/foo/Flag2On.java");
  }

  @Test
  public void testFeatureFlagsAttributeSetsSelectInBinaryAlias() throws Exception {
    useConfiguration(
        "--experimental_dynamic_configs=notrim",
        "--enforce_transitive_configs_for_config_feature_flag");
    scratch.file(
        "java/com/foo/BUILD",
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
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = select({",
        "    ':flag1@on': ['Flag1On.java'],",
        "    '//conditions:default': ['Flag1Off.java'],",
        "  }),",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        "  transitive_configs = [':flag1'],",
        ")",
        "alias(",
        "  name = 'alias',",
        "  actual = ':foo',",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/foo:alias");
    List<String> inputs =
        prettyArtifactNames(actionsTestUtil().artifactClosureOf(getFinalUnsignedApk(binary)));

    assertThat(inputs).contains("java/com/foo/Flag1On.java");
    assertThat(inputs).doesNotContain("java/com/foo/Flag1Off.java");
  }

  @Test
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagValueIsInvalid() throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration(
        "--experimental_dynamic_configs=on",
        "--enforce_transitive_configs_for_config_feature_flag");
    scratch.file(
        "java/com/foo/BUILD",
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
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  deps = [':lib'],",
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
    scratch.file(
        "java/com/foo/BUILD",
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
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
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
  public void testFeatureFlagsAttributeFailsAnalysisIfFlagIsAliased() throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration(
        "--experimental_dynamic_configs=notrim",
        "--enforce_transitive_configs_for_config_feature_flag");
    scratch.file(
        "java/com/foo/BUILD",
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
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  feature_flags = {",
        "    'alias': 'on',",
        "  },",
        "  transitive_configs = [':flag1'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/foo")).isNull();
    assertContainsEvent(
        "in feature_flags attribute of android_binary rule //java/com/foo:foo: "
            + "Feature flags must be named directly, not through aliases; "
            + "use '//java/com/foo:flag1', not '//java/com/foo:alias'");
  }

  @Test
  public void testFeatureFlagsAttributeSetsFeatureFlagProviderValues() throws Exception {
    useConfiguration(
        "--experimental_dynamic_configs=notrim",
        "--enforce_transitive_configs_for_config_feature_flag");
    scratch.file(
        "java/com/foo/reader.bzl",
        "def _impl(ctx):",
        "  ctx.actions.write(",
        "      ctx.outputs.java,",
        "      '\\n'.join([",
        "          str(target.label) + ': ' + target[config_common.FeatureFlagInfo].value",
        "          for target in ctx.attr.flags]))",
        "  return [DefaultInfo(files=depset([ctx.outputs.java]))]",
        "flag_reader = rule(",
        "  implementation=_impl,",
        "  attrs={'flags': attr.label_list(providers=[config_common.FeatureFlagInfo])},",
        "  outputs={'java': '%{name}.java'},",
        ")");
    scratch.file(
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
        "  transitive_configs = [':flag1', ':flag2'],",
        ")",
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = [':FooFlags.java'],",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        "  transitive_configs = [':flag1', ':flag2'],",
        ")");
    Artifact flagList =
        getFirstArtifactEndingWith(
            actionsTestUtil()
                .artifactClosureOf(getFinalUnsignedApk(getConfiguredTarget("//java/com/foo"))),
            "/FooFlags.java");
    FileWriteAction action = (FileWriteAction) getGeneratingAction(flagList);
    assertThat(action.getFileContents())
        .isEqualTo("//java/com/foo:flag1: on\n//java/com/foo:flag2: off");
  }

  @Test
  public void featureFlagsSetByAndroidBinaryAreInRequiredFragments() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct");
    scratch.file(
        "java/com/foo/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = [':FooFlags.java'],",
        "  feature_flags = {",
        "    'flag1': 'on',",
        "  },",
        ")");
    ConfiguredTarget ct = getConfiguredTarget("//java/com/foo:foo");
    assertThat(ct.getProvider(RequiredConfigFragmentsProvider.class).getRequiredConfigFragments())
        .contains("//java/com/foo:flag1");
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
    List<String> args = resourceArguments(resource);
    Artifact inputManifest =
        getFirstArtifactEndingWith(
            getGeneratingSpawnAction(resource.getManifest()).getInputs(), "AndroidManifest.xml");
    assertContainsSublist(
        args,
        ImmutableList.of(
            "--primaryData", "java/r/android/res::" + inputManifest.getExecPathString()));
    assertThat(args).contains("--uncompressedExtensions");
    assertThat(args.get(args.indexOf("--uncompressedExtensions") + 1)).isEqualTo(".apk,.so");
    assertThat(getGeneratingSpawnActionArgs(getCompressedUnsignedApk(binary)))
        .containsAtLeast("--nocompress_suffixes", ".apk", ".so")
        .inOrder();
    assertThat(getGeneratingSpawnActionArgs(getFinalUnsignedApk(binary)))
        .containsAtLeast("--nocompress_suffixes", ".apk", ".so")
        .inOrder();
  }

  @Test
  public void testFeatureFlagPolicyMustContainRuleToUseFeatureFlags() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratch.overwriteFile(
        "tools/allowlists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['//flag'])");
    scratch.file(
        "flag/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['right', 'wrong'],",
        "    default_value = 'right',",
        "    visibility = ['//java/com/google/android/foo:__pkg__'],",
        ")");
    scratch.file(
        "java/com/google/android/foo/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = [':FooFlags.java'],",
        "  feature_flags = {",
        "    '//flag:flag': 'right',",
        "  },",
        "  transitive_configs = ['//flag:flag'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNull();
    assertContainsEvent(
        "in feature_flags attribute of android_binary rule //java/com/google/android/foo:foo: "
            + "the feature_flags attribute is not available in package "
            + "'java/com/google/android/foo'");
  }

  @Test
  public void testFeatureFlagPolicyDoesNotBlockRuleIfInPolicy() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratch.overwriteFile(
        "tools/allowlists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['//flag', '//java/com/google/android/foo'])");
    scratch.file(
        "flag/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['right', 'wrong'],",
        "    default_value = 'right',",
        "    visibility = ['//java/com/google/android/foo:__pkg__'],",
        ")");
    scratch.file(
        "java/com/google/android/foo/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = [':FooFlags.java'],",
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
    scratch.overwriteFile(
        "tools/allowlists/config_feature_flag/BUILD",
        "package_group(",
        "    name = 'config_feature_flag',",
        "    packages = ['*super* busted package group'])");
    scratch.file(
        "java/com/google/android/foo/BUILD",
        "android_binary(",
        "  name = 'foo',",
        "  manifest = 'AndroidManifest.xml',",
        "  srcs = [':FooFlags.java'],",
        ")");
    assertThat(getConfiguredTarget("//java/com/google/android/foo:foo")).isNotNull();
    // the package_group is busted, so we would have failed to get this far if we depended on it
    assertNoEvents();
    // Check time: does this test actually test what we're testing for?
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//tools/allowlists/config_feature_flag:config_feature_flag"))
        .isNull();
    assertContainsEvent("*super* busted package group");
  }

  @Test
  public void testAapt2WithAndroidSdk() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")");

    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    Artifact apk = getImplicitOutputArtifact(a, AndroidRuleClasses.ANDROID_RESOURCES_APK);

    assertThat(getGeneratingSpawnActionArgs(apk))
        .containsAtLeast(
            "--aapt2",
            // The path to aapt2 is different between Blaze and Bazel, so we omit it here.
            // It's safe to do so as we've already checked for the `--aapt2` flag.
            "--tool",
            "AAPT2_PACKAGE");
  }

  @Test
  public void testAapt2WithAndroidSdkAndDependencies() throws Exception {
    scratch.file(
        "java/b/BUILD",
        "android_library(",
        "    name = 'b',",
        "    srcs = ['B.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")");

    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [ '//java/b:b' ],",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")");

    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    ConfiguredTarget b = getDirectPrerequisite(a, "//java/b:b");

    Artifact classJar =
        getImplicitOutputArtifact(a, AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);
    Artifact apk = getImplicitOutputArtifact(a, AndroidRuleClasses.ANDROID_RESOURCES_APK);

    SpawnAction apkAction = getGeneratingSpawnAction(apk);
    assertThat(getGeneratingSpawnActionArgs(apk))
        .containsAtLeast(
            "--aapt2",
            // The path to aapt2 is different between Blaze and Bazel, so we omit it here.
            // It's safe to do so as we've already checked for the `--aapt2` flag.
            "--tool",
            "AAPT2_PACKAGE");

    assertThat(apkAction.getInputs().toList())
        .contains(getImplicitOutputArtifact(b, AndroidRuleClasses.ANDROID_COMPILED_SYMBOLS));

    SpawnAction classAction = getGeneratingSpawnAction(classJar);
    assertThat(classAction.getInputs().toList())
        .containsAtLeast(
            getImplicitOutputArtifact(a, AndroidRuleClasses.ANDROID_R_TXT),
            getImplicitOutputArtifact(b, AndroidRuleClasses.ANDROID_R_TXT));
  }

  @Test
  public void testAapt2ResourceShrinkingAction() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               shrink_resources = 1,",
        "               proguard_specs = ['proguard-spec.pro'],)");

    ConfiguredTargetAndData targetAndData =
        getConfiguredTargetAndData("//java/com/google/android/hello:hello");
    ConfiguredTarget binary = targetAndData.getConfiguredTarget();

    Artifact jar = getResourceClassJar(targetAndData);
    assertThat(getGeneratingAction(jar).getMnemonic()).isEqualTo("RClassGenerator");
    assertThat(getGeneratingSpawnActionArgs(jar)).contains("--finalFields");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    assertThat(artifacts)
        .containsAtLeast(
            getFirstArtifactEndingWith(artifacts, "resource_files.zip"),
            getFirstArtifactEndingWith(artifacts, "proguard.jar"),
            getFirstArtifactEndingWith(artifacts, "shrunk.ap_"));

    List<String> processingArgs =
        getGeneratingSpawnActionArgs(getFirstArtifactEndingWith(artifacts, "resource_files.zip"));

    assertThat(flagValue("--resourcesOutput", processingArgs))
        .endsWith("hello_files/resource_files.zip");

    List<String> proguardArgs =
        getGeneratingSpawnActionArgs(getFirstArtifactEndingWith(artifacts, "proguard.jar"));

    assertThat(flagValue("-outjars", proguardArgs)).endsWith("hello_proguard.jar");

    List<String> shrinkingArgs =
        getGeneratingSpawnActionArgs(getFirstArtifactEndingWith(artifacts, "shrunk.ap_"));

    assertThat(flagValue("--tool", shrinkingArgs)).isEqualTo("SHRINK_AAPT2");

    assertThat(flagValue("--aapt2", shrinkingArgs)).isEqualTo(flagValue("--aapt2", processingArgs));
    assertThat(flagValue("--resources", shrinkingArgs))
        .isEqualTo(flagValue("--resourcesOutput", processingArgs));
    assertThat(flagValue("--shrunkJar", shrinkingArgs))
        .isEqualTo(flagValue("-outjars", proguardArgs));
    assertThat(flagValue("--proguardMapping", shrinkingArgs))
        .isEqualTo(flagValue("-printmapping", proguardArgs));
    assertThat(flagValue("--rTxt", shrinkingArgs))
        .isEqualTo(flagValue("--rOutput", processingArgs));
    assertThat(flagValue("--resourcesConfigOutput", shrinkingArgs))
        .endsWith("resource_optimization.cfg");

    List<String> packageArgs =
        getGeneratingSpawnActionArgs(getFirstArtifactEndingWith(artifacts, "_hello_proguard.cfg"));

    assertThat(flagValue("--tool", packageArgs)).isEqualTo("AAPT2_PACKAGE");
    assertThat(packageArgs).doesNotContain("--conditionalKeepRules");
  }

  @Test
  public void testAapt2ResourceShrinking_proguardSpecsAbsent_noShrunkApk() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               shrink_resources = 1,)");

    ConfiguredTargetAndData targetAndData =
        getConfiguredTargetAndData("//java/com/google/android/hello:hello");
    ConfiguredTarget binary = targetAndData.getConfiguredTarget();

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    assertThat(getFirstArtifactEndingWith(artifacts, "shrunk.ap_")).isNull();
  }

  @Test
  public void testAapt2ResourceCycleShrinking() throws Exception {
    useConfiguration("--experimental_android_resource_cycle_shrinking=true");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'hello',",
        "               srcs = ['Foo.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               shrink_resources = 1,",
        "               proguard_specs = ['proguard-spec.pro'],)");

    ConfiguredTargetAndData targetAndData =
        getConfiguredTargetAndData("//java/com/google/android/hello:hello");
    ConfiguredTarget binary = targetAndData.getConfiguredTarget();

    Artifact jar = getResourceClassJar(targetAndData);
    assertThat(getGeneratingAction(jar).getMnemonic()).isEqualTo("RClassGenerator");
    assertThat(getGeneratingSpawnActionArgs(jar)).contains("--nofinalFields");

    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    List<String> packageArgs =
        getGeneratingSpawnActionArgs(getFirstArtifactEndingWith(artifacts, "_hello_proguard.cfg"));

    assertThat(flagValue("--tool", packageArgs)).isEqualTo("AAPT2_PACKAGE");
    assertThat(packageArgs).contains("--conditionalKeepRules");
  }

  @Test
  public void testAapt2ResourceCycleShinkingWithoutResourceShrinking() throws Exception {
    useConfiguration("--experimental_android_resource_cycle_shrinking=true");
    checkError(
        "java/a",
        "a",
        "resource cycle shrinking can only be enabled when resource shrinking is enabled",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        "    shrink_resources = 0,",
        ")");
  }

  @Test
  public void testOnlyProguardSpecs() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l2',",
        "                srcs = ['MoreMaps.java'],",
        "                neverlink = 1)",
        "android_library(name = 'l3',",
        "                idl_srcs = ['A.aidl'],",
        "                deps = [':l2'])",
        "android_library(name = 'l4',",
        "                srcs = ['SubMoreMaps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l3', ':l4'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");
    checkProguardUse(
        "//java/com/google/android/hello:b",
        "b_proguard.jar",
        false,
        null,
        /*splitOptimizationPass=*/ false,
        targetConfig.getBinFragment()
            + "/java/com/google/android/hello/proguard/b/legacy_b_combined_library_jars.jar");
  }

  @Test
  public void testOnlyProguardSpecsProguardJar() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l2',",
        "                srcs = ['MoreMaps.java'],",
        "                neverlink = 1)",
        "android_library(name = 'l3',",
        "                idl_srcs = ['A.aidl'],",
        "                deps = [':l2'])",
        "android_library(name = 'l4',",
        "                srcs = ['SubMoreMaps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l3', ':l4'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_generate_mapping = 1,",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");

    ConfiguredTarget output = getConfiguredTarget("//java/com/google/android/hello:b_proguard.jar");
    assertProguardUsed(output);

    output = getConfiguredTarget("//java/com/google/android/hello:b_proguard.map");
    assertWithMessage("proguard.map is not in the rule output")
        .that(
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(output), "_proguard.map"))
        .isNotNull();
  }

  @Test
  public void testCommandLineForMultipleProguardSpecs() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l1',",
        "                srcs = ['Maps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l1'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:b");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "_proguard.jar");

    assertWithMessage("Proguard action does not contain expected inputs.")
        .that(prettyArtifactNames(action.getInputs()))
        .containsAtLeast(
            "java/com/google/android/hello/proguard-spec.pro",
            "java/com/google/android/hello/proguard-spec1.pro",
            "java/com/google/android/hello/proguard-spec2.pro");

    assertThat(action.getArguments())
        .containsExactly(
            getProguardBinary().getExecPathString(),
            "-forceprocessing",
            "-injars",
            execPathEndingWith(action.getInputs(), "b_deploy.jar"),
            "-outjars",
            execPathEndingWith(action.getOutputs(), "b_proguard.jar"),
            // Only one combined library jar
            "-libraryjars",
            execPathEndingWith(action.getInputs(), "legacy_b_combined_library_jars.jar"),
            "@" + execPathEndingWith(action.getInputs(), "b_proguard.cfg"),
            "@java/com/google/android/hello/proguard-spec.pro",
            "@java/com/google/android/hello/proguard-spec1.pro",
            "@java/com/google/android/hello/proguard-spec2.pro",
            "-printseeds",
            execPathEndingWith(action.getOutputs(), "_proguard.seeds"),
            "-printusage",
            execPathEndingWith(action.getOutputs(), "_proguard.usage"),
            "-printconfiguration",
            execPathEndingWith(action.getOutputs(), "_proguard.config"))
        .inOrder();
  }

  /** Regression test for b/17790639 */
  @Test
  public void testNoDuplicatesInProguardCommand() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l1',",
        "                srcs = ['Maps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l1'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:b");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "_proguard.jar");
    assertThat(action.getArguments())
        .containsExactly(
            getProguardBinary().getExecPathString(),
            "-forceprocessing",
            "-injars",
            execPathEndingWith(action.getInputs(), "b_deploy.jar"),
            "-outjars",
            execPathEndingWith(action.getOutputs(), "b_proguard.jar"),
            // Only one combined library jar
            "-libraryjars",
            execPathEndingWith(action.getInputs(), "legacy_b_combined_library_jars.jar"),
            "@" + execPathEndingWith(action.getInputs(), "b_proguard.cfg"),
            "@java/com/google/android/hello/proguard-spec.pro",
            "@java/com/google/android/hello/proguard-spec1.pro",
            "@java/com/google/android/hello/proguard-spec2.pro",
            "-printseeds",
            execPathEndingWith(action.getOutputs(), "_proguard.seeds"),
            "-printusage",
            execPathEndingWith(action.getOutputs(), "_proguard.usage"),
            "-printconfiguration",
            execPathEndingWith(action.getOutputs(), "_proguard.config"))
        .inOrder();
  }

  @Test
  public void testProguardMapping() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'],",
        "               proguard_generate_mapping = 1)");
    checkProguardUse(
        "//java/com/google/android/hello:b",
        "b_proguard.jar",
        true,
        null,
        /*splitOptimizationPass=*/ false,
        getAndroidJarPath());
  }

  @Test
  public void testProguardMappingProvider() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l2',",
        "                srcs = ['MoreMaps.java'],",
        "                neverlink = 1)",
        "android_library(name = 'l3',",
        "                idl_srcs = ['A.aidl'],",
        "                deps = [':l2'])",
        "android_library(name = 'l4',",
        "                srcs = ['SubMoreMaps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b1',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l3', ':l4'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_generate_mapping = 1,",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])",
        "android_binary(name = 'b2',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l3', ':l4'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");

    ConfiguredTarget output = getConfiguredTarget("//java/com/google/android/hello:b1");
    assertProguardUsed(output);
    Artifact mappingArtifact = getBinArtifact("b1_proguard.map", output);
    ProguardMappingProvider mappingProvider = output.get(ProguardMappingProvider.PROVIDER);
    assertThat(mappingProvider.getProguardMapping()).isEqualTo(mappingArtifact);

    output = getConfiguredTarget("//java/com/google/android/hello:b2");
    assertProguardUsed(output);
    assertThat(output.get(ProguardMappingProvider.PROVIDER)).isNull();
  }

  @Test
  public void testLegacyOptimizationModeUsesExtraProguardSpecs() throws Exception {
    useConfiguration("--extra_proguard_specs=java/com/google/android/hello:extra.pro");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "exports_files(['extra.pro'])",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");
    checkProguardUse(
        "//java/com/google/android/hello:b",
        "b_proguard.jar",
        false,
        null,
        /*splitOptimizationPass=*/ false,
        getAndroidJarPath());

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    getFilesToBuild(getConfiguredTarget("//java/com/google/android/hello:b")),
                    "_proguard.jar");
    assertThat(prettyArtifactNames(action.getInputs())).containsNoDuplicates();
    assertThat(Collections2.filter(action.getArguments(), arg -> arg.startsWith("@")))
        .containsExactly(
            "@" + execPathEndingWith(action.getInputs(), "/proguard-spec.pro"),
            "@" + execPathEndingWith(action.getInputs(), "/_b_proguard.cfg"),
            "@java/com/google/android/hello/extra.pro");
  }

  @Test
  public void testExtraProguardSpecsDontDuplicateProguardInputFiles() throws Exception {
    useConfiguration("--extra_proguard_specs=java/com/google/android/hello:proguard-spec.pro");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'])");
    checkProguardUse(
        "//java/com/google/android/hello:b",
        "b_proguard.jar",
        false,
        null,
        /*splitOptimizationPass=*/ false,
        getAndroidJarPath());

    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    getFilesToBuild(getConfiguredTarget("//java/com/google/android/hello:b")),
                    "_proguard.jar");
    assertThat(prettyArtifactNames(action.getInputs())).containsNoDuplicates();
    assertThat(Collections2.filter(action.getArguments(), arg -> arg.startsWith("@")))
        .containsExactly(
            "@java/com/google/android/hello/proguard-spec.pro",
            "@" + execPathEndingWith(action.getInputs(), "/_b_proguard.cfg"));
  }

  @Test
  public void testProguardSpecFromLibraryUsedInBinary() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l2',",
        "                srcs = ['MoreMaps.java'],",
        "                proguard_specs = ['library_spec.cfg'])",
        "android_library(name = 'l3',",
        "                idl_srcs = ['A.aidl'],",
        "                proguard_specs = ['library_spec.cfg'],",
        "                deps = [':l2'])",
        "android_library(name = 'l4',",
        "                srcs = ['SubMoreMaps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l3', ':l4'],",
        "               proguard_specs = ['proguard-spec.pro'],",
        "               manifest = 'AndroidManifest.xml',)");
    assertProguardUsed(getConfiguredTarget("//java/com/google/android/hello:b"));
    assertProguardGenerated(getConfiguredTarget("//java/com/google/android/hello:b"));
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    getFilesToBuild(getConfiguredTarget("//java/com/google/android/hello:b")),
                    "_proguard.jar");
    assertThat(prettyArtifactNames(action.getInputs()))
        .contains("java/com/google/android/hello/proguard-spec.pro");
    assertThat(prettyArtifactNames(action.getInputs()))
        .contains(
            "java/com/google/android/hello/validated_proguard/l2/java/com/google/android/hello/library_spec.cfg_valid");
    assertThat(prettyArtifactNames(action.getInputs())).containsNoDuplicates();
  }

  @Test
  public void testResourcesUsedInProguardGenerate() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = ['res/values/strings.xml'],",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");
    scratch.file(
        "java/com/google/android/hello/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:b");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(
                    actionsTestUtil().artifactClosureOf(getFilesToBuild(binary)), "_proguard.cfg");

    assertProguardGenerated(binary);
    assertWithMessage("Generate proguard action does not contain expected input.")
        .that(prettyArtifactNames(action.getInputs()))
        .contains("java/com/google/android/hello/res/values/strings.xml");
  }

  @Test
  public void testUseSingleJarForLibraryJars() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l1',",
        "                srcs = ['Maps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l1'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:b");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "_proguard.jar");

    checkProguardLibJars(
        action,
        targetConfig.getBinFragment()
            + "/java/com/google/android/hello/proguard/b/legacy_b_combined_library_jars.jar");
  }

  @Test
  public void testUseSingleJarForFilteredLibraryJars() throws Exception {
    useConfiguration("--experimental_filter_library_jar_with_program_jar=true");
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_library(name = 'l1',",
        "                srcs = ['Maps.java'],",
        "                neverlink = 1)",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               deps = [':l1'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro', 'proguard-spec1.pro',",
        "                                 'proguard-spec2.pro'])");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:b");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "_proguard.jar");

    checkProguardLibJars(
        action,
        targetConfig.getBinFragment()
            + "/java/com/google/android/hello/proguard/b/legacy_b_combined_library_jars_filtered.jar");
  }

  @Test
  public void testOnlyOneLibraryJar() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "android_binary(name = 'b',",
        "               srcs = ['HelloApp.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               proguard_specs = ['proguard-spec.pro'],",
        "               proguard_generate_mapping = 1)");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:b");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "_proguard.jar");

    checkProguardLibJars(action, getAndroidJarPath());
  }

  @Test
  public void testApkInfoAccessibleFromStarlark() throws Exception {
    scratch.file(
        "java/com/google/android/BUILD",
        "load(':postprocess.bzl', 'postprocess')",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               manifest = 'AndroidManifest.xml')",
        "postprocess(name = 'postprocess', dep = ':b1')");
    scratch.file(
        "java/com/google/android/postprocess.bzl",
        "def _impl(ctx):",
        "  return [DefaultInfo(files=depset([ctx.attr.dep[ApkInfo].signed_apk]))]",
        "postprocess = rule(implementation=_impl,",
        "              attrs={'dep': attr.label(providers=[ApkInfo])})");
    ConfiguredTarget postprocess = getConfiguredTarget("//java/com/google/android:postprocess");
    assertThat(postprocess).isNotNull();
    assertThat(
            prettyArtifactNames(postprocess.getProvider(FilesToRunProvider.class).getFilesToRun()))
        .containsExactly("java/com/google/android/b1.apk");
  }

  @Test
  public void testInstrumentationInfoAccessibleFromStarlark() throws Exception {
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "load(':instr.bzl', 'instr')",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               instruments = ':b2',",
        "               manifest = 'AndroidManifest.xml')",
        "android_binary(name = 'b2',",
        "               srcs = ['b2.java'],",
        "               manifest = 'AndroidManifest.xml')",
        "instr(name = 'instr', dep = ':b1')");
    scratch.file(
        "java/com/google/android/instr/instr.bzl",
        "def _impl(ctx):",
        "  target = ctx.attr.dep[AndroidInstrumentationInfo].target.signed_apk",
        "  return [DefaultInfo(files=depset([target]))]",
        "instr = rule(implementation=_impl,",
        "             attrs={'dep': attr.label(providers=[AndroidInstrumentationInfo])})");
    ConfiguredTarget instr = getConfiguredTarget("//java/com/google/android/instr");
    assertThat(instr).isNotNull();
    assertThat(prettyArtifactNames(instr.getProvider(FilesToRunProvider.class).getFilesToRun()))
        .containsExactly("java/com/google/android/instr/b2.apk");
  }

  @Test
  public void testInstrumentationInfoCreatableFromStarlark() throws Exception {
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "load(':instr.bzl', 'instr')",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               instruments = ':b2',",
        "               manifest = 'AndroidManifest.xml')",
        "android_binary(name = 'b2',",
        "               srcs = ['b2.java'],",
        "               manifest = 'AndroidManifest.xml')",
        "instr(name = 'instr', dep = ':b1')");
    scratch.file(
        "java/com/google/android/instr/instr.bzl",
        "def _impl(ctx):",
        "  target = ctx.attr.dep[AndroidInstrumentationInfo].target",
        "  return [AndroidInstrumentationInfo(target=target)]",
        "instr = rule(implementation=_impl,",
        "             attrs={'dep': attr.label(providers=[AndroidInstrumentationInfo])})");
    ConfiguredTarget instr = getConfiguredTarget("//java/com/google/android/instr");
    assertThat(instr).isNotNull();
    assertThat(instr.get(AndroidInstrumentationInfo.PROVIDER).getTarget().getApk().prettyPrint())
        .isEqualTo("java/com/google/android/instr/b2.apk");
  }

  @Test
  public void testInstrumentationInfoProviderHasApks() throws Exception {
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               instruments = ':b2',",
        "               manifest = 'AndroidManifest.xml')",
        "android_binary(name = 'b2',",
        "               srcs = ['b2.java'],",
        "               manifest = 'AndroidManifest.xml')");
    ConfiguredTarget b1 = getConfiguredTarget("//java/com/google/android/instr:b1");
    AndroidInstrumentationInfo provider = b1.get(AndroidInstrumentationInfo.PROVIDER);
    assertThat(provider.getTarget()).isNotNull();
    assertThat(provider.getTarget().getApk().prettyPrint())
        .isEqualTo("java/com/google/android/instr/b2.apk");
  }

  @Test
  public void testNoInstrumentationInfoProviderIfNotInstrumenting() throws Exception {
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               manifest = 'AndroidManifest.xml')");
    ConfiguredTarget b1 = getConfiguredTarget("//java/com/google/android/instr:b1");
    AndroidInstrumentationInfo provider = b1.get(AndroidInstrumentationInfo.PROVIDER);
    assertThat(provider).isNull();
  }

  @Test
  public void testFilterActionWithInstrumentedBinary() throws Exception {
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "android_binary(name = 'b1',",
        "               srcs = ['b1.java'],",
        "               instruments = ':b2',",
        "               manifest = 'AndroidManifest.xml')",
        "android_binary(name = 'b2',",
        "               srcs = ['b2.java'],",
        "               manifest = 'AndroidManifest.xml')");
    ConfiguredTarget b1 = getConfiguredTarget("//java/com/google/android/instr:b1");
    SpawnAction action =
        (SpawnAction)
            actionsTestUtil().getActionForArtifactEndingWith(getFilesToBuild(b1), "_filtered.jar");
    assertThat(action.getArguments())
        .containsAtLeast(
            "--inputZip",
            getFirstArtifactEndingWith(action.getInputs(), "b1_deploy.jar").getExecPathString(),
            "--filterZips",
            getFirstArtifactEndingWith(action.getInputs(), "b2_deploy.jar").getExecPathString(),
            "--outputZip",
            getFirstArtifactEndingWith(action.getOutputs(), "b1_filtered.jar").getExecPathString(),
            "--filterTypes",
            ".class",
            "--checkHashMismatch",
            "IGNORE",
            "--explicitFilters",
            "/BR\\.class$,/databinding/[^/]+Binding\\.class$,R\\.class,R\\$.*\\.class",
            "--outputMode",
            "DONT_CARE");
  }

  /**
   * 'proguard_specs' attribute gets read by an implicit outputs function: the current heuristic is
   * that if this attribute is configurable, we assume its contents are non-empty and thus create
   * the mybinary_proguard.jar output. Test that here.
   */
  @Test
  public void testConfigurableProguardSpecs() throws Exception {
    scratch.file(
        "conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})",
        "config_setting(",
        "    name = 'b',",
        "    values = {'test_arg': 'b'})");
    scratchConfiguredTarget(
        "java/foo",
        "abin",
        "android_binary(",
        "    name = 'abin',",
        "    srcs = ['a.java'],",
        "    proguard_specs = select({",
        "        '//conditions:a': [':file1.pro'],",
        "        '//conditions:b': [],",
        "        '//conditions:default': [':file3.pro'],",
        "    }) + [",
        // Add a long list here as a regression test for b/68238721
        "        'file4.pro',",
        "        'file5.pro',",
        "        'file6.pro',",
        "        'file7.pro',",
        "        'file8.pro',",
        "    ],",
        "    manifest = 'AndroidManifest.xml')");
    checkProguardUse(
        "//java/foo:abin",
        "abin_proguard.jar",
        /*expectMapping=*/ false,
        /*passes=*/ null,
        /*splitOptimizationPass=*/ false,
        getAndroidJarPath());
  }

  @Test
  public void alwaysSkipParsingActionWithAapt2() throws Exception {
    scratch.file(
        "java/b/BUILD",
        "android_library(",
        "    name = 'b',",
        "    srcs = ['B.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")");

    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [ '//java/b:b' ],",
        "    resource_files = [ 'res/values/values.xml' ], ",
        ")");

    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    ConfiguredTarget b = getDirectPrerequisite(a, "//java/b:b");

    List<String> resourceProcessingArgs =
        getGeneratingSpawnActionArgs(getValidatedResources(a).getApk());

    assertThat(resourceProcessingArgs).contains("AAPT2_PACKAGE");
    String directData =
        resourceProcessingArgs.get(resourceProcessingArgs.indexOf("--directData") + 1);
    assertThat(directData).contains("symbols.zip");
    assertThat(directData).doesNotContain("merged.bin");

    List<String> resourceMergingArgs =
        getGeneratingSpawnActionArgs(getValidatedResources(b).getJavaClassJar());

    assertThat(resourceMergingArgs).contains("MERGE_COMPILED");
  }

  @Test
  public void starlarkJavaInfoToAndroidBinaryAttributes() throws Exception {
    scratch.file(
        "java/r/android/extension.bzl",
        "def _impl(ctx):",
        "  dep_params = ctx.attr.dep[JavaInfo]",
        "  return [dep_params]",
        "my_rule = rule(",
        "    _impl,",
        "    attrs = {",
        "        'dep': attr.label(),",
        "    },",
        ")");
    scratch.file(
        "java/r/android/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "android_library(",
        "    name = 'al_bottom_for_deps',",
        "    srcs = ['java/A.java'],",
        ")",
        "my_rule(",
        "    name = 'mya',",
        "    dep = ':al_bottom_for_deps',",
        ")",
        "android_binary(",
        "    name = 'foo_app',",
        "    srcs = ['java/B.java'],",
        "    deps = [':mya'],",
        "    manifest = 'AndroidManifest.xml',",
        // TODO(b/75051107): Remove the following line when fixed.
        "    incremental_dexing = 0,",
        ")");
    // Test that all bottom jars are on the runtime classpath of the app.
    ConfiguredTarget target = getConfiguredTarget("//java/r/android:foo_app");
    ImmutableList<Artifact> transitiveSrcJars =
        OutputGroupInfo.get(target).getOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP).toList();
    assertThat(ActionsTestUtil.baseArtifactNames(transitiveSrcJars))
        .containsExactly("libal_bottom_for_deps-src.jar", "libfoo_app-src.jar");
  }

  @Test
  public void androidManifestMergerOrderAlphabetical_MergeesSortedByExecPath() throws Exception {
    // Hack: Avoid the Android split transition by turning off fat_apk_cpu/android_cpu.
    // This is necessary because the transition would change the configuration directory, causing
    // the manifest paths in the assertion not to match.
    // TODO(b/140634666): Get the library manifests in the same configuration as the binary gets
    // them.
    useConfiguration(
        "--fat_apk_cpu=", "--android_cpu=", "--android_manifest_merger_order=alphabetical");
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
    Artifact androidCoreManifest = getLibraryManifest(getConfiguredTarget("//java/android:core"));
    Artifact androidUtilityManifest =
        getLibraryManifest(getConfiguredTarget("//java/android:utility"));
    Artifact binaryLibraryManifest =
        getLibraryManifest(getConfiguredTarget("//java/binary:library"));
    Artifact commonManifest = getLibraryManifest(getConfiguredTarget("//java/common:common"));
    Artifact commonThemeManifest = getLibraryManifest(getConfiguredTarget("//java/common:theme"));

    assertThat(getBinaryMergeeManifests(getConfiguredTarget("//java/binary:application")))
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                androidCoreManifest.getExecPath().toString(), "//java/android:core",
                androidUtilityManifest.getExecPath().toString(), "//java/android:utility",
                binaryLibraryManifest.getExecPath().toString(), "//java/binary:library",
                commonManifest.getExecPath().toString(), "//java/common:common",
                commonThemeManifest.getExecPath().toString(), "//java/common:theme"))
        .inOrder();
  }

  @Test
  public void androidManifestMergerOrderAlphabeticalByConfiguration_MergeesSortedByPathInBinOrGen()
      throws Exception {
    // Hack: Avoid the Android split transition by turning off fat_apk_cpu/android_cpu.
    // This is necessary because the transition would change the configuration directory, causing
    // the manifest paths in the assertion not to match.
    // TODO(b/140634666): Get the library manifests in the same configuration as the binary gets
    // them.
    useConfiguration(
        "--fat_apk_cpu=",
        "--android_cpu=",
        "--android_manifest_merger_order=alphabetical_by_configuration");
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
        "    transitive_configs = ['//flags:a', '//flags:b'],",
        ")");
    scratch.file(
        "java/binary/BUILD",
        "android_binary(",
        "    name = 'application',",
        "    srcs = ['App.java'],",
        "    manifest = 'app/AndroidManifest.xml',",
        "    deps = [':library'],",
        "    feature_flags = {",
        "        '//flags:a': 'on',",
        "        '//flags:b': 'on',",
        "        '//flags:c': 'on',",
        "    },",
        "    transitive_configs = ['//flags:a', '//flags:b', '//flags:c'],",
        ")",
        "android_library(",
        "    name = 'library',",
        "    manifest = 'library/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    deps = ['//java/common:theme', '//java/android:utility'],",
        "    transitive_configs = ['//flags:a', '//flags:b', '//flags:c'],",
        ")");
    scratch.file(
        "java/common/BUILD",
        "android_library(",
        "    name = 'common',",
        "    manifest = 'common/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['common/res/values/common.xml'],",
        "    deps = ['//java/android:core'],",
        "    transitive_configs = ['//flags:a'],",
        ")",
        "android_library(",
        "    name = 'theme',",
        "    manifest = 'theme/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['theme/res/values/values.xml'],",
        "    transitive_configs = ['//flags:a', '//flags:b', '//flags:c'],",
        ")");
    scratch.file(
        "flags/BUILD",
        "config_feature_flag(",
        "    name = 'a',",
        "    allowed_values = ['on', 'off'],",
        "    default_value = 'off',",
        ")",
        "config_feature_flag(",
        "    name = 'b',",
        "    allowed_values = ['on', 'off'],",
        "    default_value = 'off',",
        ")",
        "config_feature_flag(",
        "    name = 'c',",
        "    allowed_values = ['on', 'off'],",
        "    default_value = 'off',",
        ")");

    assertThat(getBinaryMergeeManifests(getConfiguredTarget("//java/binary:application")).values())
        .containsExactly(
            "//java/android:core",
            "//java/android:utility",
            "//java/binary:library",
            "//java/common:common",
            "//java/common:theme")
        .inOrder();
  }

  @Test
  public void androidBinaryResourceInjection() throws Exception {
    // ResourceApk.toResourceInfo() is not compatible with the AndroidResourcesInfo provider.
    useConfiguration("--experimental_omit_resources_info_provider_from_android_binary");

    scratch.file(
        "java/com/app/android_resource_injection.bzl",
        "def _android_application_resources_impl(ctx):",
        "    resource_proguard_config = ctx.actions.declare_file(ctx.label.name + '/proguard.cfg')",
        "    ctx.actions.write(resource_proguard_config, '# Empty proguard.cfg')",
        "",
        "    resource_apk = ctx.actions.declare_file(ctx.label.name + '/injected_resource.ap_')",
        "    ctx.actions.write(resource_apk, 'empty ap_')",
        "",
        "    manifest = ctx.actions.declare_file(ctx.label.name + '/AndroidManifest.xml')",
        "    ctx.actions.write(manifest, 'empty manifest')",
        "",
        "    resource_java_src_jar = ctx.actions.declare_file(",
        "        ctx.label.name + '/resources.srcjar')",
        "    ctx.actions.write(resource_java_src_jar, 'empty manifest')",
        "",
        "    resource_java_class_jar = ctx.actions.declare_file(",
        "        ctx.label.name + '/resources.jar')",
        "    ctx.actions.write(resource_java_class_jar, 'empty manifest')",
        "",
        "    r_txt = ctx.actions.declare_file(ctx.label.name + '/r.txt')",
        "    ctx.actions.write(r_txt, 'empty r_txt')",
        "",
        "    resources_zip = ctx.actions.declare_file(ctx.label.name + '/resource_files.zip')",
        "    ctx.actions.write(resources_zip, 'empty resources zip')",
        "",
        "    return [",
        "        DefaultInfo(files = depset([resource_apk, resource_proguard_config])),",
        "        AndroidApplicationResourceInfo(",
        "            resource_apk = resource_apk,",
        "            resource_java_src_jar = resource_java_src_jar,",
        "            resource_java_class_jar = resource_java_class_jar,",
        "            manifest = manifest,",
        "            resource_proguard_config = resource_proguard_config,",
        "            main_dex_proguard_config = None,",
        "            r_txt = r_txt,",
        "            resources_zip = resources_zip,",
        "        ),",
        "    ]",
        "",
        "android_application_resources = rule(",
        "    implementation = _android_application_resources_impl,",
        "    provides = [AndroidApplicationResourceInfo],",
        ")",
        "",
        "def resource_injected_android_binary(**attrs):",
        "  android_application_resources(name = 'application_resources')",
        "  attrs['application_resources'] = ':application_resources'",
        "  native.android_binary(**attrs)");

    scratch.file(
        "java/com/app/BUILD",
        "load(':android_resource_injection.bzl', 'resource_injected_android_binary')",
        "resource_injected_android_binary(",
        "  name = 'app',",
        "  manifest = 'AndroidManifest.xml')");

    ConfiguredTarget app = getConfiguredTarget("//java/com/app:app");
    assertThat(app).isNotNull();

    // Assert that the injected resource apk is the only resource apk being merged into the final
    // apk.
    Action singleJarAction = getGeneratingAction(getFinalUnsignedApk(app));
    List<Artifact> resourceApks =
        getArtifactsEndingWith(singleJarAction.getInputs().toList(), ".ap_");
    assertThat(resourceApks).hasSize(1);
    assertThat(resourceApks.get(0).getExecPathString())
        .endsWith("java/com/app/application_resources/injected_resource.ap_");
  }

  @Test
  public void testAndroidStarlarkApiNativeLibs() throws Exception {
    scratch.file(
        "java/a/fetch_native_libs.bzl",
        "def _impl(ctx):",
        "  libs = ctx.attr.android_binary.android.native_libs",
        "  return [DefaultInfo(files = libs.values()[0])]",
        "fetch_native_libs = rule(implementation = _impl,",
        "    attrs = {",
        "        'android_binary': attr.label(),",
        "    },",
        ")");
    scratch.file(
        "java/a/BUILD",
        "load('//java/a:fetch_native_libs.bzl', 'fetch_native_libs')",
        "android_binary(",
        "    name = 'app',",
        "    srcs=['Main.java'],",
        "    manifest='AndroidManifest.xml',",
        "    deps=[':cc'],",
        ")",
        "cc_library(",
        "    name = 'cc',",
        "    srcs = ['cc.cc'],",
        ")",
        "fetch_native_libs(",
        "    name = 'clibs',",
        "    android_binary = 'app',",
        ")");
    ConfiguredTarget clibs = getConfiguredTarget("//java/a:clibs");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                clibs.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("libapp.so");
  }

  @Test
  public void testInstrumentsManifestMergeEnabled() throws Exception {
    // This is the incorrect behavior where dependency manifests are merged into the test apk.
    useConfiguration("--noexperimental_disable_instrumentation_manifest_merge");
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "android_binary(",
        "    name = 'b1',",
        "    srcs = ['b1.java'],",
        "    instruments = ':b2',",
        "    deps = [':lib'],",
        "    manifest = 'test/AndroidManifest.xml',",
        ")",
        "android_library(",
        "    name = 'lib',",
        "    manifest = 'lib/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['lib/res/values/strings.xml'],",
        ")",
        "android_binary(",
        "    name = 'b2',",
        "    srcs = ['b2.java'],",
        "    deps = [':lib'],",
        "    manifest = 'bin/AndroidManifest.xml',",
        ")");
    assertThat(
            getBinaryMergeeManifests(getConfiguredTarget("//java/com/google/android/instr:b1"))
                .values())
        .containsExactly("//java/com/google/android/instr:lib");
  }

  @Test
  public void testInstrumentsManifestMergeDisabled() throws Exception {
    // This is the correct behavior where dependency manifests are not merged into the test apk.
    useConfiguration("--experimental_disable_instrumentation_manifest_merge");
    scratch.file(
        "java/com/google/android/instr/BUILD",
        "android_binary(",
        "    name = 'b1',",
        "    srcs = ['b1.java'],",
        "    instruments = ':b2',",
        "    deps = [':lib'],",
        "    manifest = 'test/AndroidManifest.xml',",
        ")",
        "android_library(",
        "    name = 'lib',",
        "    manifest = 'lib/AndroidManifest.xml',",
        "    exports_manifest = 1,",
        "    resource_files = ['lib/res/values/strings.xml'],",
        ")",
        "android_binary(",
        "    name = 'b2',",
        "    srcs = ['b2.java'],",
        "    deps = [':lib'],",
        "    manifest = 'bin/AndroidManifest.xml',",
        ")");
    assertThat(
            getBinaryMergeeManifests(getConfiguredTarget("//java/com/google/android/instr:b1"))
                .values())
        .isEmpty();
  }

  // DEPENDENCY order is not tested; the incorrect order of dependencies means the test would
  // have to enforce incorrect behavior.
  // TODO(b/117338320): Add a test when dependency order is fixed.
}
