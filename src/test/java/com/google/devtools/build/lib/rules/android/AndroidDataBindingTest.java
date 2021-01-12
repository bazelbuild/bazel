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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getJavacArguments;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.android.AndroidDataBindingTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidDataBindingTest.WithoutPlatforms;
import com.google.devtools.build.lib.rules.android.databinding.UsesDataBindingProvider;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests for Bazel's Android data binding support. */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidDataBindingTest extends AndroidBuildViewTestCase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidDataBindingTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidDataBindingTest {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  private void writeDataBindingFiles() throws Exception {
    scratch.file(
        "java/android/library/BUILD",
        "android_library(",
        "    name = 'lib_with_data_binding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib.java'],",
        "    resource_files = [],",
        ")");
    scratch.file(
        "java/android/library/MyLib.java", "package android.library; public class MyLib {};");
    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/library:lib_with_data_binding'],",
        ")");
    scratch.file(
        "java/android/binary/MyApp.java", "package android.binary; public class MyApp {};");
  }

  private void writeDataBindingFilesWithNoResourcesDep() throws Exception {
    scratch.file(
        "java/android/lib_with_resource_files/BUILD",
        "android_library(",
        "    name = 'lib_with_resource_files',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['LibWithResourceFiles.java'],",
        "    resource_files = glob(['res/**']),",
        ")");
    scratch.file(
        "java/android/lib_with_resource_files/LibWithResourceFiles.java",
        "package android.lib_with_resource_files; public class LibWithResourceFiles {};");

    scratch.file(
        "java/android/lib_no_resource_files/BUILD",
        "android_library(",
        "    name = 'lib_no_resource_files',",
        "    enable_data_binding = 1,",
        "    srcs = ['LibNoResourceFiles.java'],",
        "    deps = ['//java/android/lib_with_resource_files'],",
        ")");
    scratch.file(
        "java/android/lib_no_resource_files/LibNoResourceFiles.java",
        "package android.lib_no_resource_files; public class LibNoResourceFiles {};");

    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/lib_no_resource_files'],",
        ")");
    scratch.file(
        "java/android/binary/MyApp.java", "package android.binary; public class MyApp {};");
  }

  @Test
  public void basicDataBindingIntegration() throws Exception {
    writeDataBindingFiles();
    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    // "Data binding"-enabled targets invoke resource processing with a request for data binding
    // output:
    Artifact libResourceInfoOutput =
        getFirstArtifactEndingWith(
            allArtifacts, "databinding/lib_with_data_binding/layout-info.zip");
    assertThat(getGeneratingSpawnActionArgs(libResourceInfoOutput))
        .containsAtLeast("--dataBindingInfoOut", libResourceInfoOutput.getExecPathString())
        .inOrder();

    Artifact binResourceInfoOutput =
        getFirstArtifactEndingWith(allArtifacts, "databinding/app/layout-info.zip");
    assertThat(getGeneratingSpawnActionArgs(binResourceInfoOutput))
        .containsAtLeast("--dataBindingInfoOut", binResourceInfoOutput.getExecPathString())
        .inOrder();

    // Java compilation includes the data binding annotation processor, the resource processor's
    // output, and the auto-generated DataBindingInfo.java the annotation processor uses to figure
    // out what to do:
    JavaCompileAction libCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "lib_with_data_binding.jar"));
    assertThat(getProcessorNames(libCompileAction))
        .contains("android.databinding.annotationprocessor.ProcessDataBinding");
    assertThat(prettyArtifactNames(libCompileAction.getInputs()))
        .containsAtLeast(
            "java/android/library/databinding/lib_with_data_binding/layout-info.zip",
            "java/android/library/databinding/lib_with_data_binding/DataBindingInfo.java");

    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    assertThat(getProcessorNames(binCompileAction))
        .contains("android.databinding.annotationprocessor.ProcessDataBinding");
    assertThat(prettyArtifactNames(binCompileAction.getInputs()))
        .containsAtLeast(
            "java/android/binary/databinding/app/layout-info.zip",
            "java/android/binary/databinding/app/DataBindingInfo.java");
  }

  @Test
  public void dataBindingCompilationUsesMetadataFromDeps() throws Exception {
    writeDataBindingFiles();
    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    // The library's compilation doesn't include any of the -setter_store.bin, layoutinfo.bin, etc.
    // files that store a dependency's data binding results (since the library has no deps).
    // We check that they don't appear as compilation inputs.
    JavaCompileAction libCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "lib_with_data_binding.jar"));
    assertThat(
            Iterables.filter(
                libCompileAction.getInputs().toList(),
                ActionsTestUtil.getArtifactSuffixMatcher(".bin")))
        .isEmpty();

    // The binary's compilation includes the library's data binding results.
    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    Iterable<Artifact> depMetadataInputs =
        Iterables.filter(
            binCompileAction.getInputs().toList(),
            ActionsTestUtil.getArtifactSuffixMatcher(".bin"));
    final String depMetadataBaseDir =
        Iterables.getFirst(depMetadataInputs, null).getExecPath().getParentDirectory().toString();
    ActionsTestUtil.execPaths(
        Iterables.filter(
            binCompileAction.getInputs().toList(),
            ActionsTestUtil.getArtifactSuffixMatcher(".bin")));
    assertThat(ActionsTestUtil.execPaths(depMetadataInputs))
        .containsExactly(
            depMetadataBaseDir + "/android.library-android.library-setter_store.bin",
            depMetadataBaseDir + "/android.library-android.library-layoutinfo.bin",
            depMetadataBaseDir + "/android.library-android.library-br.bin");
  }

  @Test
  public void dataBindingAnnotationProcessorFlags() throws Exception {
    writeDataBindingFiles();
    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));
    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    String dataBindingFilesDir =
        targetConfig
            .getBinDirectory(RepositoryName.MAIN)
            .getExecPath()
            .getRelative("java/android/binary/databinding/app")
            .getPathString();
    ImmutableList<String> expectedJavacopts =
        ImmutableList.of(
            "-Aandroid.databinding.bindingBuildFolder=" + dataBindingFilesDir,
            "-Aandroid.databinding.generationalFileOutDir=" + dataBindingFilesDir,
            "-Aandroid.databinding.sdkDir=/not/used",
            "-Aandroid.databinding.artifactType=APPLICATION",
            "-Aandroid.databinding.xmlOutDir=" + dataBindingFilesDir,
            "-Aandroid.databinding.exportClassListTo=/tmp/exported_classes",
            "-Aandroid.databinding.modulePackage=android.binary",
            "-Aandroid.databinding.minApi=14");
    assertThat(getJavacArguments(binCompileAction)).containsAtLeastElementsIn(expectedJavacopts);

    // Regression test for b/63134122
    JavaCompileInfo javaCompileInfo =
        binCompileAction
            .getExtraActionInfo(actionKeyContext)
            .getExtension(JavaCompileInfo.javaCompileInfo);
    assertThat(javaCompileInfo.getJavacOptList()).containsAtLeastElementsIn(expectedJavacopts);
  }

  @Test
  public void dataBindingIncludesTransitiveDepsForLibsWithNoResources() throws Exception {
    writeDataBindingFilesWithNoResourcesDep();
    ConfiguredTarget ct = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ct));

    // Data binding resource processing outputs are expected for the app and libs with resources.
    assertThat(
            getFirstArtifactEndingWith(
                allArtifacts, "databinding/lib_with_resource_files/layout-info.zip"))
        .isNotNull();
    assertThat(getFirstArtifactEndingWith(allArtifacts, "databinding/app/layout-info.zip"))
        .isNotNull();

    // Compiling the app's Java source includes data binding metadata from the resource-equipped
    // lib, but not the resource-empty one.
    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    List<String> appJarInputs = prettyArtifactNames(binCompileAction.getInputs());
    String libWithResourcesMetadataBaseDir =
        "java/android/binary/databinding/app/"
            + "dependent-lib-artifacts/java/android/lib_with_resource_files/databinding/"
            + "lib_with_resource_files/bin-files/android.lib_with_resource_files-";
    assertThat(appJarInputs)
        .containsAtLeast(
            "java/android/binary/databinding/app/layout-info.zip",
            libWithResourcesMetadataBaseDir + "android.lib_with_resource_files-setter_store.bin",
            libWithResourcesMetadataBaseDir + "android.lib_with_resource_files-layoutinfo.bin",
            libWithResourcesMetadataBaseDir + "android.lib_with_resource_files-br.bin");
    for (String compileInput : appJarInputs) {
      assertThat(compileInput).doesNotMatch(".*lib_no_resource_files.*.bin");
    }
  }

  @Test
  public void libsWithNoResourcesOnlyRunAnnotationProcessor() throws Exception {
    // Bazel skips resource processing because there are no new resources to process. But it still
    // runs the annotation processor to ensure the Java compiler reads Java sources referenced by
    // the deps' resources (e.g. "<variable type="some.package.SomeClass" />"). Without this,
    // JavaBuilder's --reduce_classpath feature would strip out those sources as "unused" and fail
    // the binary's compilation with unresolved symbol errors.
    writeDataBindingFilesWithNoResourcesDep();
    ConfiguredTarget ct = getConfiguredTarget("//java/android/lib_no_resource_files");
    NestedSet<Artifact> libArtifacts = getFilesToBuild(ct);

    assertThat(getFirstArtifactEndingWith(libArtifacts, "_resources.jar")).isNull();
    assertThat(getFirstArtifactEndingWith(libArtifacts, "layout-info.zip")).isNull();

    JavaCompileAction libCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(libArtifacts, "lib_no_resource_files.jar"));
    // The annotation processor is attached to the Java compilation:
    assertThat(getJavacArguments(libCompileAction))
        .containsAtLeast(
            "--processors", "android.databinding.annotationprocessor.ProcessDataBinding");
    // The dummy .java file with annotations that trigger the annotation process is present:
    assertThat(prettyArtifactNames(libCompileAction.getInputs()))
        .contains(
            "java/android/lib_no_resource_files/databinding/lib_no_resource_files/"
                + "DataBindingInfo.java");
  }

  @Test
  public void missingDataBindingAttributeStillAnalyzes() throws Exception {
    // When a library is missing enable_data_binding = 1, we expect it to fail in execution (because
    // aapt doesn't know how to read the data binding expressions). But analysis should work.
    scratch.file(
        "java/android/library/BUILD",
        "android_library(",
        "    name = 'lib_with_data_binding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib.java'],",
        "    resource_files = [],",
        ")");
    scratch.file(
        "java/android/library/MyLib.java", "package android.library; public class MyLib {};");
    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 0,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/library:lib_with_data_binding'],",
        ")");
    scratch.file(
        "java/android/binary/MyApp.java", "package android.binary; public class MyApp {};");
    assertThat(getConfiguredTarget("//java/android/binary:app")).isNotNull();
  }

  @Test
  public void dataBindingProviderIsProvided() throws Exception {
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
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")");
    scratch.file(
        "java/a/BUILD",
        "android_library(",
        "    name = 'a', ",
        "    srcs = ['A.java'],",
        "    enable_data_binding = 1,",
        "    manifest = 'a/AndroidManifest.xml',",
        "    resource_files = ['res/values/a.xml'],",
        ")");

    useConfiguration("--android_sdk=//sdk:sdk");
    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    final UsesDataBindingProvider usesDataBindingProvider = a.get(UsesDataBindingProvider.PROVIDER);

    assertWithMessage(UsesDataBindingProvider.NAME).that(usesDataBindingProvider).isNotNull();

    Truth.assertThat(
            usesDataBindingProvider.getMetadataOutputs().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly(
            "java/a/databinding/a/bin-files/a-a-setter_store.bin",
            "java/a/databinding/a/bin-files/a-a-layoutinfo.bin",
            "java/a/databinding/a/bin-files/a-a-br.bin");
  }

  @Test
  public void ensureDataBindingProviderIsPropagatedThroughNonDataBindingLibs() throws Exception {
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
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")");
    scratch.file(
        "java/a/BUILD",
        "android_library(",
        "    name = 'a', ",
        "    srcs = ['A.java'],",
        "    enable_data_binding = 1,",
        "    manifest = 'a/AndroidManifest.xml',",
        "    resource_files = ['res/values/a.xml'],",
        ")");
    scratch.file(
        "java/b/BUILD",
        "android_library(",
        "    name = 'b', ",
        "    srcs = ['A.java'],",
        "    deps = ['//java/a:a'],",
        ")");
    useConfiguration("--android_sdk=//sdk:sdk");
    ConfiguredTarget b = getConfiguredTarget("//java/b:b");
    assertWithMessage("UsesDataBindingProvider")
        .that(b.get(UsesDataBindingProvider.PROVIDER))
        .isNotNull();
  }
}
