// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static java.util.stream.Collectors.toList;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingV2Provider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.ImportDepsCheckingLevel;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.rules.android.AarImport}. */
@RunWith(JUnit4.class)
public class AarImportTest extends BuildViewTestCase {

  @Before
  public void setup() throws Exception {
    scratch.file(
        "aapt2/sdk/BUILD",
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
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")");
    scratch.file(
        "a/BUILD",
        "java_import(",
        "    name = 'foo_src',",
        "    jars = ['foo-src.jar'],",
        ")",
        "aar_import(",
        "    name = 'foo',",
        "    aar = 'foo.aar',",
        "    srcjar = ':foo_src',",
        ")",
        "aar_import(",
        "    name = 'baz',",
        "    aar = 'baz.aar',",
        ")",
        "java_import(",
        "    name = 'bar_src',",
        "    jars = ['bar-src.jar'],",
        ")",
        "aar_import(",
        "    name = 'bar',",
        "    aar = 'bar.aar',",
        "    srcjar = ':bar_src',",
        "    deps = [':baz'],",
        "    exports = [':foo', '//java:baz'],",
        ")",
        "aar_import(",
        "    name = 'intermediate',",
        "    aar = 'intermediate.aar',",
        "    deps = [':bar']",
        ")",
        "aar_import(",
        "    name = 'last',",
        "    aar = 'last.aar',",
        "    deps = [':intermediate'],",
        ")",
        "android_library(",
        "    name = 'library',",
        "    manifest = 'AndroidManifest.xml',",
        "    custom_package = 'com.google.arrimport',",
        "    resource_files = ['res/values/values.xml'],",
        "    srcs = ['App.java'],",
        "    deps = [':foo'],",
        ")");
    scratch.file(
        "java/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['App.java'],",
        "    deps = ['//a:bar'],",
        ")",
        "android_library(",
        "    name = 'lib',",
        "    exports = ['//a:bar'],",
        ")",
        "java_import(",
        "    name = 'baz',",
        "    jars = ['baz.jar'],",
        "    constraints = ['android'],",
        ")");
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  @Test
  public void aapt2RTxtProvided() throws Exception {
    useConfiguration("--android_sdk=//aapt2/sdk:sdk");

    ConfiguredTarget libTarget = getConfiguredTarget("//a:library");

    NestedSet<Artifact> transitiveCompiledSymbols =
        libTarget.get(AndroidResourcesInfo.PROVIDER).getTransitiveCompiledSymbols();

    assertThat(
            transitiveCompiledSymbols.toSet().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toSet()))
        .containsExactly("a/foo_symbols/symbols.zip", "a/library_symbols/symbols.zip");

    NestedSet<ValidatedAndroidResources> directResources =
        libTarget.get(AndroidResourcesInfo.PROVIDER).getDirectAndroidResources();

    assertThat(directResources.toList()).hasSize(1);

    ValidatedAndroidResources resourceContainer = directResources.toList().get(0);
    assertThat(resourceContainer.getAapt2RTxt()).isNotNull();
  }

  @Test
  public void testResourcesProvided() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:foo");

    NestedSet<ValidatedAndroidResources> directResources =
        aarImportTarget.get(AndroidResourcesInfo.PROVIDER).getDirectAndroidResources();
    assertThat(directResources.toList()).hasSize(1);

    ValidatedAndroidResources resourceContainer = directResources.toList().get(0);
    assertThat(resourceContainer.getManifest()).isNotNull();

    Artifact resourceTreeArtifact = Iterables.getOnlyElement(resourceContainer.getResources());
    assertThat(resourceTreeArtifact.isTreeArtifact()).isTrue();
    assertThat(resourceTreeArtifact.getExecPathString()).endsWith("_aar/unzipped/resources/foo");

    NestedSet<ParsedAndroidAssets> directAssets =
        aarImportTarget.get(AndroidAssetsInfo.PROVIDER).getDirectParsedAssets();
    assertThat(directAssets.toList()).hasSize(1);

    ParsedAndroidAssets assets = directAssets.toList().get(0);
    assertThat(assets.getSymbols()).isNotNull();

    Artifact assetsTreeArtifact = Iterables.getOnlyElement(assets.getAssets());
    assertThat(assetsTreeArtifact.isTreeArtifact()).isTrue();
    assertThat(assetsTreeArtifact.getExecPathString()).endsWith("_aar/unzipped/assets/foo");
  }

  @Test
  public void testDatabindingInfoProvided() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:last");

    DataBindingV2Provider provider = aarImportTarget.get(DataBindingV2Provider.PROVIDER);

    Artifact setterStore = Iterables.getOnlyElement(provider.getSetterStores());
    assertThat(setterStore.isTreeArtifact()).isTrue();
    assertThat(setterStore.getExecPathString())
        .endsWith("_aar/unzipped/data-binding-setter_store/last");

    assertThat(
            provider.getTransitiveBRFiles().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(toList()))
        .containsExactly(
            "a/_aar/unzipped/data-binding-br/baz",
            "a/_aar/unzipped/data-binding-br/foo",
            "a/_aar/unzipped/data-binding-br/bar",
            "a/_aar/unzipped/data-binding-br/intermediate",
            "a/_aar/unzipped/data-binding-br/last");
  }

  @Test
  public void testSourceJarsProvided() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:foo");

    Iterable<Artifact> srcJars =
        JavaInfo.getProvider(JavaSourceJarsProvider.class, aarImportTarget).getSourceJars();
    assertThat(srcJars).hasSize(1);
    Artifact srcJar = Iterables.getOnlyElement(srcJars);
    assertThat(srcJar.getExecPathString()).endsWith("foo-src.jar");

    Iterable<Artifact> srcInfoJars =
        JavaInfo.getProvider(JavaSourceInfoProvider.class, aarImportTarget)
            .getSourceJarsForJarFiles();
    assertThat(srcInfoJars).hasSize(1);
    Artifact srcInfoJar = Iterables.getOnlyElement(srcInfoJars);
    assertThat(srcInfoJar.getExecPathString()).endsWith("foo-src.jar");
  }

  @Test
  public void testSourceJarsCollectedTransitively() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:bar");

    NestedSet<Artifact> srcJars =
        JavaInfo.getProvider(JavaSourceJarsProvider.class, aarImportTarget)
            .getTransitiveSourceJars();
    assertThat(ActionsTestUtil.baseArtifactNames(srcJars))
        .containsExactly("foo-src.jar", "bar-src.jar");

    Iterable<Artifact> srcInfoJars =
        JavaInfo.getProvider(JavaSourceInfoProvider.class, aarImportTarget)
            .getSourceJarsForJarFiles();
    assertThat(srcInfoJars).hasSize(1);
    Artifact srcInfoJar = Iterables.getOnlyElement(srcInfoJars);
    assertThat(srcInfoJar.getExecPathString()).endsWith("bar-src.jar");
  }

  @Test
  public void testResourcesExtractor() throws Exception {
    ValidatedAndroidResources resourceContainer =
        getConfiguredTarget("//a:foo")
            .get(AndroidResourcesInfo.PROVIDER)
            .getDirectAndroidResources()
            .toList()
            .get(0);

    Artifact resourceTreeArtifact = resourceContainer.getResources().get(0);
    Artifact aarResourcesExtractor =
        getHostConfiguredTarget(
                ruleClassProvider.getToolsRepository() + "//tools/android:aar_resources_extractor")
            .getProvider(FilesToRunProvider.class)
            .getExecutable();

    ParsedAndroidAssets assets =
        getConfiguredTarget("//a:foo")
            .get(AndroidAssetsInfo.PROVIDER)
            .getDirectParsedAssets()
            .toList()
            .get(0);
    Artifact assetsTreeArtifact = assets.getAssets().get(0);

    DataBindingV2Provider dataBindingV2Provider =
        getConfiguredTarget("//a:foo").get(DataBindingV2Provider.PROVIDER);
    Artifact databindingBrTreeArtifact =
        dataBindingV2Provider.getTransitiveBRFiles().toList().get(0);
    Artifact databindingSetterStoreTreeArtifact = dataBindingV2Provider.getSetterStores().get(0);

    assertThat(getGeneratingSpawnAction(resourceTreeArtifact).getArguments())
        .containsExactly(
            aarResourcesExtractor.getExecPathString(),
            "--input_aar",
            "a/foo.aar",
            "--output_res_dir",
            resourceTreeArtifact.getExecPathString(),
            "--output_assets_dir",
            assetsTreeArtifact.getExecPathString(),
            "--output_databinding_br_dir",
            databindingBrTreeArtifact.getExecPathString(),
            "--output_databinding_setter_store_dir",
            databindingSetterStoreTreeArtifact.getExecPathString());
  }

  @Test
  public void testDepsCheckerActionExistsForLevelError() throws Exception {
    useConfiguration("--experimental_import_deps_checking=ERROR");
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:last");
    OutputGroupInfo outputGroupInfo = aarImportTarget.get(OutputGroupInfo.SKYLARK_CONSTRUCTOR);
    NestedSet<Artifact> outputGroup =
        outputGroupInfo.getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    assertThat(outputGroup.toList()).hasSize(2);

    // We should force asset merging to happen
    Artifact mergedAssetsZip =
        aarImportTarget.get(AndroidAssetsInfo.PROVIDER).getValidationResult();
    assertThat(outputGroup.toList()).contains(mergedAssetsZip);

    // Get the other artifact from the output group
    Artifact artifact = ActionsTestUtil.getFirstArtifactEndingWith(outputGroup, "jdeps.proto");

    assertThat(artifact.isTreeArtifact()).isFalse();
    assertThat(artifact.getExecPathString()).endsWith("_aar/last/jdeps.proto");

    SpawnAction checkerAction = getGeneratingSpawnAction(artifact);
    List<String> arguments = checkerAction.getArguments();
    assertThat(arguments)
        .containsAtLeast(
            "--bootclasspath_entry",
            "--classpath_entry",
            "--directdep",
            "--input",
            "--checking_mode=error",
            "--rule_label",
            "//a:last",
            "--jdeps_output");
    ensureArgumentsHaveClassEntryOptionWithSuffix(
        arguments, "/intermediate/classes_and_libs_merged.jar");
    assertThat(arguments.stream().filter(arg -> "--classpath_entry".equals(arg)).count())
        .isEqualTo(9); // transitive classpath
    assertThat(arguments.stream().filter(arg -> "--directdep".equals(arg)).count())
        .isEqualTo(2); // 1 declared dep
  }

  @Test
  public void testDepsCheckerActionExistsForLevelWarning() throws Exception {
    checkDepsCheckerActionExistsForLevel(ImportDepsCheckingLevel.WARNING, "warning");
  }

  @Test
  public void testDepsCheckerActionDoesNotExistsForLevelOff() throws Exception {
    useConfiguration("--experimental_import_deps_checking=off");
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:bar");
    OutputGroupInfo outputGroupInfo = aarImportTarget.get(OutputGroupInfo.SKYLARK_CONSTRUCTOR);
    NestedSet<Artifact> outputGroup =
        outputGroupInfo.getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    assertThat(outputGroup.toList()).hasSize(1);
    assertThat(ActionsTestUtil.getFirstArtifactEndingWith(outputGroup, "jdeps.proto")).isNull();
  }

  private void checkDepsCheckerActionExistsForLevel(
      ImportDepsCheckingLevel level, String expectedCheckingMode) throws Exception {
    useConfiguration("--experimental_import_deps_checking=" + level.name());
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:bar");
    OutputGroupInfo outputGroupInfo = aarImportTarget.get(OutputGroupInfo.SKYLARK_CONSTRUCTOR);
    NestedSet<Artifact> outputGroup =
        outputGroupInfo.getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    assertThat(outputGroup.toList()).hasSize(2);

    // We should force asset merging to happen
    Artifact mergedAssetsZip =
        aarImportTarget.get(AndroidAssetsInfo.PROVIDER).getValidationResult();
    assertThat(outputGroup.toList()).contains(mergedAssetsZip);

    // Get the other artifact from the output group
    Artifact artifact = ActionsTestUtil.getFirstArtifactEndingWith(outputGroup, "jdeps.proto");
    checkDepsCheckerOutputArtifact(artifact, expectedCheckingMode);
  }

  private void checkDepsCheckerOutputArtifact(Artifact artifact, String expectedCheckingMode)
      throws CommandLineExpansionException {
    assertThat(artifact.isTreeArtifact()).isFalse();
    assertThat(artifact.getExecPathString()).endsWith("_aar/bar/jdeps.proto");

    SpawnAction checkerAction = getGeneratingSpawnAction(artifact);
    List<String> arguments = checkerAction.getArguments();
    assertThat(arguments)
        .containsAtLeast(
            "--bootclasspath_entry",
            "--classpath_entry",
            "--input",
            "--rule_label",
            "--jdeps_output",
            "--checking_mode=" + expectedCheckingMode);
  }

  /**
   * Tests whether the given argument list contains an argument nameds "--classpath_entry" with a
   * value that ends with the given suffix.
   */
  private static void ensureArgumentsHaveClassEntryOptionWithSuffix(
      List<String> arguments, String suffix) {
    assertThat(arguments).isNotEmpty();
    Iterator<String> iterator = arguments.iterator();
    assertThat(iterator.hasNext()).isTrue();
    String prev = iterator.next();
    while (iterator.hasNext()) {
      String current = iterator.next();
      if ("--classpath_entry".equals(prev) && current.endsWith(suffix)) {
        return; // Success.
      }
      prev = current;
    }
    Assert.fail(
        "The arguments does not have the expected --classpath_entry: The arguments are "
            + arguments
            + ", and the expected class entry suffix is "
            + suffix);
  }

  @Test
  public void testNativeLibsProvided() throws Exception {
    ConfiguredTarget androidLibraryTarget = getConfiguredTarget("//java:lib");

    NestedSet<Artifact> nativeLibs =
        androidLibraryTarget.get(AndroidNativeLibsInfo.PROVIDER).getNativeLibs();
    assertThat(nativeLibs.toList())
        .containsExactly(
            ActionsTestUtil.getFirstArtifactEndingWith(nativeLibs, "foo/native_libs.zip"),
            ActionsTestUtil.getFirstArtifactEndingWith(nativeLibs, "bar/native_libs.zip"),
            ActionsTestUtil.getFirstArtifactEndingWith(nativeLibs, "baz/native_libs.zip"));
  }

  @Test
  public void testNativeLibsMakesItIntoApk() throws Exception {
    scratch.file(
        "java/com/google/android/hello/BUILD",
        "aar_import(",
        "    name = 'my_aar',",
        "    aar = 'my_aar.aar',",
        ")",
        "android_binary(",
        "    name = 'my_app',",
        "    srcs = ['HelloApp.java'],",
        "    deps = [':my_aar'],",
        "    manifest = 'AndroidManifest.xml',",
        ")");
    ConfiguredTarget binary = getConfiguredTarget("//java/com/google/android/hello:my_app");
    SpawnAction apkBuilderAction =
        (SpawnAction)
            actionsTestUtil()
                .getActionForArtifactEndingWith(getFilesToBuild(binary), "my_app_unsigned.apk");
    assertThat(
            Iterables.find(
                apkBuilderAction.getArguments(),
                Predicates.containsPattern("_aar/my_aar/native_libs.zip$")))
        .isNotEmpty();
  }

  @Test
  public void testClassesJarProvided() throws Exception {
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:foo");

    Iterable<OutputJar> outputJars =
        JavaInfo.getProvider(JavaRuleOutputJarsProvider.class, aarImportTarget).getOutputJars();
    assertThat(outputJars).hasSize(1);

    Artifact classesJar = outputJars.iterator().next().getClassJar();
    assertThat(classesJar.getFilename()).isEqualTo("classes_and_libs_merged.jar");

    SpawnAction jarMergingAction = ((SpawnAction) getGeneratingAction(classesJar));
    assertThat(jarMergingAction.getArguments()).contains("--dont_change_compression");
  }

  @Test
  public void testNoCustomJavaPackage() throws Exception {
    ValidatedAndroidResources resourceContainer =
        getConfiguredTarget("//a:foo")
            .get(AndroidResourcesInfo.PROVIDER)
            .getDirectAndroidResources()
            .toList()
            .get(0);

    // aar_import should not set a custom java package. Instead aapt will read the
    // java package from the manifest.
    assertThat(resourceContainer.getJavaPackage()).isNull();
  }

  @Test
  public void testDepsPropagatesMergedAarJars() throws Exception {
    Action appCompileAction =
        getGeneratingAction(
            ActionsTestUtil.getFirstArtifactEndingWith(
                actionsTestUtil()
                    .artifactClosureOf(getFileConfiguredTarget("//java:app.apk").getArtifact()),
                "libapp.jar"));
    assertThat(appCompileAction).isNotNull();
    assertThat(ActionsTestUtil.prettyArtifactNames(appCompileAction.getInputs()))
        .containsAtLeast(
            "a/_aar/foo/classes_and_libs_merged.jar",
            "a/_aar/bar/classes_and_libs_merged.jar",
            "a/_aar/baz/classes_and_libs_merged.jar");
  }

  @Test
  public void testExportsPropagatesMergedAarJars() throws Exception {
    FileConfiguredTarget appTarget = getFileConfiguredTarget("//java:app.apk");
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(appTarget.getArtifact());

    ConfiguredTarget bar = getConfiguredTarget("//a:bar");
    Artifact barClassesJar =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "bar/classes_and_libs_merged.jar");
    // Verify that bar/classes_and_libs_merged.jar was in the artifact closure.
    assertThat(barClassesJar).isNotNull();
    assertThat(barClassesJar.getArtifactOwner().getLabel()).isEqualTo(bar.getLabel());

    ConfiguredTarget foo = getConfiguredTarget("//a:foo");
    Artifact fooClassesJar =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "foo/classes_and_libs_merged.jar");
    // Verify that foo/classes_and_libs_merged.jar was in the artifact closure.
    assertThat(fooClassesJar).isNotNull();
    assertThat(fooClassesJar.getArtifactOwner().getLabel()).isEqualTo(foo.getLabel());

    ConfiguredTarget baz = getConfiguredTarget("//java:baz.jar");
    Artifact bazJar = ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "baz.jar");
    // Verify that baz.jar was in the artifact closure
    assertThat(bazJar).isNotNull();
    assertThat(bazJar.getArtifactOwner().getLabel()).isEqualTo(baz.getLabel());
  }

  @Test
  public void testExportsPropagatesResources() throws Exception {
    FileConfiguredTarget appTarget = getFileConfiguredTarget("//java:app.apk");
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(appTarget.getArtifact());

    ConfiguredTarget bar = getConfiguredTarget("//a:bar");
    Artifact barResources =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "_aar/unzipped/resources/bar");
    assertThat(barResources).isNotNull();
    assertThat(barResources.getArtifactOwner().getLabel()).isEqualTo(bar.getLabel());

    ConfiguredTarget foo = getConfiguredTarget("//a:foo");
    Artifact fooResources =
        ActionsTestUtil.getFirstArtifactEndingWith(artifacts, "_aar/unzipped/resources/foo");
    assertThat(fooResources).isNotNull();
    assertThat(fooResources.getArtifactOwner().getLabel()).isEqualTo(foo.getLabel());
  }

  @Test
  public void testJavaCompilationArgsProvider() throws Exception {
    useConfiguration("--experimental_import_deps_checking=ERROR");
    ConfiguredTarget aarImportTarget = getConfiguredTarget("//a:bar");

    JavaCompilationArgsProvider provider =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, aarImportTarget);
    assertThat(provider).isNotNull();
    assertThat(artifactsToStrings(provider.getRuntimeJars()))
        .containsExactly(
            "bin a/_aar/bar/classes_and_libs_merged.jar",
            "bin a/_aar/foo/classes_and_libs_merged.jar",
            "bin a/_aar/baz/classes_and_libs_merged.jar",
            "src java/baz.jar");
    List<Artifact> compileTimeJavaDependencyArtifacts =
        provider.getCompileTimeJavaDependencyArtifacts().toList();
    assertThat(compileTimeJavaDependencyArtifacts).hasSize(2);
    assertThat(
            compileTimeJavaDependencyArtifacts.stream()
                .filter(artifact -> artifact.getExecPathString().endsWith("/_aar/foo/jdeps.proto"))
                .collect(Collectors.toList()))
        .hasSize(1);
    assertThat(
            compileTimeJavaDependencyArtifacts.stream()
                .filter(artifact -> artifact.getExecPathString().endsWith("/_aar/bar/jdeps.proto"))
                .collect(Collectors.toList()))
        .hasSize(1);
  }

  @Test
  public void testFailsWithoutAndroidSdk() throws Exception {
    scratch.file("sdk/BUILD", "alias(", "    name = 'sdk',", "    actual = 'doesnotexist',", ")");
    useConfiguration("--android_sdk=//sdk");
    checkError(
        "aar",
        "aar",
        "No Android SDK found. Use the --android_sdk command line option to specify one.",
        "aar_import(",
        "    name = 'aar',",
        "    aar = 'a.aar',",
        ")");
  }

  @Test
  public void testExportsManifest() throws Exception {
    Artifact binaryMergedManifest =
        getConfiguredTarget("//java:app").get(ApkInfo.PROVIDER).getMergedManifest();
    // Compare root relative path strings instead of artifacts due to difference in configuration
    // caused by the Android split transition.
    assertThat(
            Iterables.transform(
                getGeneratingAction(binaryMergedManifest).getInputs().toList(),
                Artifact::getRootRelativePathString))
        .containsAtLeast(getAndroidManifest("//a:foo"), getAndroidManifest("//a:bar"));
  }

  private String getAndroidManifest(String aarImport) throws Exception {
    return getConfiguredTarget(aarImport)
        .get(AndroidResourcesInfo.PROVIDER)
        .getDirectAndroidResources()
        .toList()
        .get(0)
        .getManifest()
        .getRootRelativePathString();
  }

  @Test
  public void testTransitiveExports() throws Exception {
    assertThat(
            getConfiguredTarget("//a:bar")
                .get(JavaInfo.PROVIDER)
                .getTransitiveExports()
                .toList(Label.class))
        .containsExactly(
            Label.parseAbsolute("//a:foo", ImmutableMap.of()),
            Label.parseAbsolute("//java:baz", ImmutableMap.of()));
  }

  @Test
  public void testRClassFromAarImportInCompileClasspath() throws Exception {
    Collection<Artifact> compilationClasspath =
        JavaInfo.getProvider(JavaCompilationInfoProvider.class, getConfiguredTarget("//a:library"))
            .getCompilationClasspath()
            .toList(Artifact.class);

    assertThat(
            compilationClasspath.stream()
                .filter(artifact -> artifact.getFilename().equalsIgnoreCase("foo_resources.jar"))
                .count())
        .isEqualTo(1);
  }
}
