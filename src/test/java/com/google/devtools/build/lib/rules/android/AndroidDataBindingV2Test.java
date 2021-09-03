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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactMatching;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getJavacArguments;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.android.AndroidDataBindingV2Test.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidDataBindingV2Test.WithoutPlatforms;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingV2Provider;
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

/** Tests for Bazel's Android data binding v2 support. */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidDataBindingV2Test extends AndroidBuildViewTestCase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidDataBindingV2Test {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidDataBindingV2Test {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

  @Before
  public void setDataBindingV2Flag() throws Exception {
    useConfiguration("--experimental_android_databinding_v2");
  }

  private void writeDataBindingLibrariesFiles() throws Exception {

    scratch.file(
        "java/android/library2/BUILD",
        "android_library(",
        "    name = 'lib2_with_databinding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib2.java'],",
        "    resource_files = [],",
        ")");

    scratch.file(
        "java/android/library/BUILD",
        "android_library(",
        "    name = 'lib_with_databinding',",
        "    deps = ['//java/android/library2:lib2_with_databinding'],",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib.java'],",
        "    resource_files = [],",
        ")");

    scratch.file(
        "java/android/library/MyLib.java", "package android.library; public class MyLib {};");
  }

  private void writeNonDataBindingLocalTestFiles() throws Exception {

    scratch.file(
        "javatests/android/test/BUILD",
        "android_local_test(",
        "    name = 'databinding_enabled_test',",
        "    deps = ['//java/android/library:lib_with_databinding'],",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyTest.java'],",
        ")");

    scratch.file(
        "javatests/android/test/MyTest.java", "package android.test; public class MyTest {};");
  }

  private void writeDataBindingLocalTestFiles() throws Exception {

    scratch.file(
        "javatests/android/test/BUILD",
        "android_local_test(",
        "    name = 'databinding_enabled_test',",
        "    deps = ['//java/android/library:lib_with_databinding'],",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyTest.java'],",
        ")");

    scratch.file(
        "javatests/android/test/MyTest.java", "package android.test; public class MyTest {};");
  }

  private void writeDataBindingFiles() throws Exception {

    writeDataBindingLibrariesFiles();

    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/library:lib_with_databinding'],",
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

  private void writeDataBindingFilesWithShrinkage() throws Exception {

    writeDataBindingLibrariesFiles();

    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    shrink_resources = 1,",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/library:lib_with_databinding'],",
        "    proguard_specs = ['proguard-spec.pro'],",
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
            allArtifacts, "databinding/lib_with_databinding/layout-info.zip");
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
                getFirstArtifactEndingWith(allArtifacts, "lib_with_databinding.jar"));
    assertThat(getProcessorNames(libCompileAction))
        .contains("android.databinding.annotationprocessor.ProcessDataBinding");
    assertThat(prettyArtifactNames(libCompileAction.getInputs()))
        .containsAtLeast(
            "java/android/library/databinding/lib_with_databinding/layout-info.zip",
            "java/android/library/databinding/lib_with_databinding/DataBindingInfo.java");

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
                getFirstArtifactEndingWith(allArtifacts, "lib2_with_databinding.jar"));
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

    final String appDependentLibArtifacts =
        Iterables.getFirst(depMetadataInputs, null).getRoot().getExecPathString()
            + "/java/android/binary/databinding/app/dependent-lib-artifacts/";
    ActionsTestUtil.execPaths(
        Iterables.filter(
            binCompileAction.getInputs().toList(),
            ActionsTestUtil.getArtifactSuffixMatcher(".bin")));
    assertThat(ActionsTestUtil.execPaths(depMetadataInputs))
        .containsExactly(
            appDependentLibArtifacts
                + "java/android/library/databinding/"
                + "lib_with_databinding/bin-files/android.library-android.library-br.bin",
            appDependentLibArtifacts
                + "java/android/library/databinding/"
                + "lib_with_databinding/bin-files/android.library-android.library-setter_store.bin",
            appDependentLibArtifacts
                + "java/android/library2/databinding/"
                + "lib2_with_databinding/bin-files/android.library2-android.library2-br.bin");
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
            "-Aandroid.databinding.exportClassListTo=/tmp/exported_classes",
            "-Aandroid.databinding.modulePackage=android.binary",
            "-Aandroid.databinding.minApi=14",
            "-Aandroid.databinding.enableV2=1",
            // Note that this includes only android.library and not android.library2
            "-Aandroid.databinding.directDependencyPkgs=[android.library]");
    assertThat(getJavacArguments(binCompileAction)).containsAtLeastElementsIn(expectedJavacopts);

    // Regression test for b/63134122
    JavaCompileInfo javaCompileInfo =
        binCompileAction
            .getExtraActionInfo(actionKeyContext)
            .getExtension(JavaCompileInfo.javaCompileInfo);
    assertThat(javaCompileInfo.getJavacOptList()).containsAtLeastElementsIn(expectedJavacopts);
  }

  @Test
  public void dataBindingAnnotationProcessorFlags_v3_4() throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
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
    String inputDir = dataBindingFilesDir + "/" + DataBinding.DEP_METADATA_INPUT_DIR;
    String outputDir = dataBindingFilesDir + "/" + DataBinding.METADATA_OUTPUT_DIR;
    ImmutableList<String> expectedJavacopts =
        ImmutableList.of(
            "-Aandroid.databinding.dependencyArtifactsDir=" + inputDir,
            "-Aandroid.databinding.aarOutDir=" + outputDir,
            "-Aandroid.databinding.sdkDir=/not/used",
            "-Aandroid.databinding.artifactType=APPLICATION",
            "-Aandroid.databinding.exportClassListOutFile=/tmp/exported_classes",
            "-Aandroid.databinding.modulePackage=android.binary",
            "-Aandroid.databinding.minApi=14",
            "-Aandroid.databinding.enableV2=1",
            // Note that this includes only android.library and not android.library2
            "-Aandroid.databinding.directDependencyPkgs=[android.library]");
    assertThat(getJavacArguments(binCompileAction)).containsAtLeastElementsIn(expectedJavacopts);

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
        "    name = 'lib_with_databinding',",
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
        "    deps = ['//java/android/library:lib_with_databinding'],",
        ")");

    scratch.file(
        "java/android/binary/MyApp.java", "package android.binary; public class MyApp {};");

    assertThat(getConfiguredTarget("//java/android/binary:app")).isNotNull();
  }

  @Test
  public void dataBindingProviderIsProvided() throws Exception {

    useConfiguration("--android_sdk=//sdk:sdk", "--experimental_android_databinding_v2");

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
        "    srcs = ['B.java'],",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = ['res/values/a.xml'],",
        ")");

    ConfiguredTarget a = getConfiguredTarget("//java/a:a");
    final DataBindingV2Provider dataBindingV2Provider = a.get(DataBindingV2Provider.PROVIDER);

    assertWithMessage(DataBindingV2Provider.NAME).that(dataBindingV2Provider).isNotNull();

    assertThat(
            dataBindingV2Provider.getSetterStores().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly("java/a/databinding/a/bin-files/a-a-setter_store.bin");

    assertThat(
            dataBindingV2Provider.getClassInfos().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly("java/a/databinding/a/class-info.zip");

    assertThat(
            dataBindingV2Provider.getTransitiveBRFiles().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly("java/a/databinding/a/bin-files/a-a-br.bin");
  }

  @Test
  public void ensureDataBindingProviderIsPropagatedThroughNonDataBindingLibs() throws Exception {

    useConfiguration("--android_sdk=//sdk:sdk", "--experimental_android_databinding_v2");

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
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = ['res/values/a.xml'],",
        ")");
    scratch.file(
        "java/b/BUILD",
        "android_library(",
        "    name = 'b', ",
        "    srcs = ['B.java'],",
        "    deps = ['//java/a:a'],",
        ")");

    ConfiguredTarget b = getConfiguredTarget("//java/b:b");
    assertWithMessage("DataBindingV2Info").that(b.get(DataBindingV2Provider.PROVIDER)).isNotNull();
  }

  @Test
  public void testDataBindingCollectedThroughExports() throws Exception {

    useConfiguration("--android_sdk=//sdk:sdk", "--experimental_android_databinding_v2");

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
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = ['res/values/a.xml'],",
        ")");

    scratch.file(
        "java/b/BUILD",
        "android_library(",
        "    name = 'b', ",
        "    srcs = ['B.java'],",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = ['res/values/a.xml'],",
        ")");

    scratch.file(
        "java/c/BUILD",
        "android_library(",
        "    name = 'c', ",
        "    exports = ['//java/a:a', '//java/b:b']",
        ")");

    ConfiguredTarget c = getConfiguredTarget("//java/c:c");
    DataBindingV2Provider provider = c.get(DataBindingV2Provider.PROVIDER);

    assertThat(
            provider.getClassInfos().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly(
            "java/a/databinding/a/class-info.zip", "java/b/databinding/b/class-info.zip");

    assertThat(
            provider.getSetterStores().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly(
            "java/a/databinding/a/bin-files/a-a-setter_store.bin",
            "java/b/databinding/b/bin-files/b-b-setter_store.bin");

    assertThat(
            provider.getTransitiveBRFiles().toList().stream()
                .map(Artifact::getRootRelativePathString)
                .collect(Collectors.toList()))
        .containsExactly(
            "java/a/databinding/a/bin-files/a-a-br.bin",
            "java/b/databinding/b/bin-files/b-b-br.bin");
  }

  @Test
  public void testMultipleAndroidLibraryDepsWithSameJavaPackageRaisesError() throws Exception {

    String databindingRuntime =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/databinding"
            + ":runtime";
    String supportAnnotations =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/annotations"
            + ":annotations";

    scratch.file(
        "java/com/lib/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['User.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")",
        "android_library(",
        "    name = 'lib2',",
        "    srcs = ['User2.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res2/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    scratch.file(
        "java/com/bin/BUILD",
        "android_binary(",
        "    name = 'bin',",
        "    srcs = ['MyActivity.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '//java/com/lib',",
        "        '//java/com/lib:lib2',",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    checkError(
        "//java/com/bin:bin",
        "Java package com.lib:\n" + "    //java/com/lib:lib\n" + "    //java/com/lib:lib2");
  }

  @Test
  public void testMultipleAndroidLibraryDepsWithSameJavaPackageThroughDiamondRaisesError()
      throws Exception {

    String databindingRuntime =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/databinding"
            + ":runtime";
    String supportAnnotations =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/annotations"
            + ":annotations";

    // The bin target depends on these target indirectly and separately through the libraries
    // in middleA and middleB.
    scratch.file(
        "java/com/bottom/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['User.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")",
        "android_library(",
        "    name = 'lib2',",
        "    srcs = ['User2.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res2/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    scratch.file(
        "java/com/middleA/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['UserMiddleA.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [",
        "        '//java/com/bottom:lib',",
        "    ],",
        ")");
    scratch.file(
        "java/com/middleB/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['UserMiddleB.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [",
        "        '//java/com/bottom:lib2',",
        "    ],",
        ")");

    scratch.file(
        "java/com/bin/BUILD",
        "android_binary(",
        "    name = 'bin',",
        "    srcs = ['MyActivity.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '//java/com/middleA:lib',",
        "        '//java/com/middleB:lib',",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    checkError(
        "//java/com/bin:bin",
        "Java package com.bottom:\n"
            + "    //java/com/bottom:lib\n"
            + "    //java/com/bottom:lib2");
  }

  @Test
  public void testMultipleAndroidLibraryDepsWithSameJavaPackageThroughCustomPackageAttrRaisesError()
      throws Exception {

    String databindingRuntime =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/databinding"
            + ":runtime";
    String supportAnnotations =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/annotations"
            + ":annotations";

    // The bin target depends on these target indirectly and separately through the libraries
    // in middleA and middleB.
    scratch.file(
        "libA/BUILD",
        "android_library(",
        "    name = 'libA',",
        "    srcs = ['UserA.java'],",
        "    custom_package = 'com.foo',",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    scratch.file(
        "libB/BUILD",
        "android_library(",
        "    name = 'libB',",
        "    srcs = ['UserB.java'],",
        "    custom_package = 'com.foo',",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    scratch.file(
        "java/com/bin/BUILD",
        "android_binary(",
        "    name = 'bin',",
        "    srcs = ['MyActivity.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '//libA:libA',",
        "        '//libB:libB',",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    checkError(
        "//java/com/bin:bin", "Java package com.foo:\n" + "    //libA:libA\n" + "    //libB:libB");
  }

  @Test
  public void testAndroidBinaryAndroidLibraryWithDatabindingSamePackageRaisesError()
      throws Exception {

    String databindingRuntime =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/databinding"
            + ":runtime";
    String supportAnnotations =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/annotations"
            + ":annotations";

    // The android_binary and android_library are in the same java package and have
    // databinding.
    scratch.file(
        "java/com/bin/BUILD",
        "android_binary(",
        "    name = 'bin',",
        "    srcs = ['MyActivity.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        ':lib',",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['User.java'],",
        "    manifest = 'LibManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    checkError(
        "//java/com/bin:bin",
        "Java package com.bin:\n" + "    //java/com/bin:bin\n" + "    //java/com/bin:lib");
  }

  @Test
  public void testSameAndroidLibraryMultipleTimesThroughDiamondDoesNotRaiseSameJavaPackageError()
      throws Exception {

    String databindingRuntime =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/databinding"
            + ":runtime";
    String supportAnnotations =
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/annotations"
            + ":annotations";

    // The bin target depends on this target twice: indirectly and separately through the libraries
    // in middleA and middleB, but this should not be a problem because it's the same library.
    scratch.file(
        "java/com/bottom/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['User.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    resource_files = glob(['res/**']),",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    scratch.file(
        "java/com/middleA/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['UserMiddleA.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [",
        "        '//java/com/bottom:lib',",
        "    ],",
        ")");
    scratch.file(
        "java/com/middleB/BUILD",
        "android_library(",
        "    name = 'lib',",
        "    srcs = ['UserMiddleB.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    deps = [",
        "        '//java/com/bottom:lib',",
        "    ],",
        ")");

    scratch.file(
        "java/com/bin/BUILD",
        "android_binary(",
        "    name = 'bin',",
        "    srcs = ['MyActivity.java'],",
        "    manifest = 'AndroidManifest.xml',",
        "    enable_data_binding = 1,",
        "    deps = [",
        "        '//java/com/middleA:lib',",
        "        '//java/com/middleB:lib',",
        "        '" + databindingRuntime + "',",
        "        '" + supportAnnotations + "',",
        "    ],",
        ")");

    // Should not throw error.
    getConfiguredTarget("//java/com/bin:bin");
  }

  private void writeDataBindingFilesWithExports() throws Exception {

    scratch.file(
        "java/android/library1/BUILD",
        "android_library(",
        "    name = 'lib1_with_databinding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib1.java'],",
        ")");

    scratch.file(
        "java/android/library2/BUILD",
        "android_library(",
        "    name = 'lib2_with_databinding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib2.java'],",
        ")");

    scratch.file(
        "java/android/library3/BUILD",
        "android_library(",
        "    name = 'lib3',",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib3.java'],",
        ")");

    scratch.file(
        "java/android/lib_with_exports/BUILD",
        "android_library(",
        "    name = 'lib_with_exports_no_databinding',",
        "    exports = [",
        "        '//java/android/library1:lib1_with_databinding',",
        "        '//java/android/library2:lib2_with_databinding',",
        "        '//java/android/library3:lib3',",
        "    ],",
        "    manifest = 'AndroidManifest.xml',",
        ")",
        "",
        "android_library(",
        "    name = 'lib_with_exports_and_databinding',",
        "    exports = [",
        "        '//java/android/library1:lib1_with_databinding',",
        "        '//java/android/library2:lib2_with_databinding',",
        "        '//java/android/library3:lib3',",
        "    ],",
        "    manifest = 'AndroidManifest.xml',",
        "    enable_data_binding = 1,",
        ")");

    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app_dep_on_exports_no_databinding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/lib_with_exports:lib_with_exports_no_databinding'],",
        ")",
        "",
        "android_binary(",
        "    name = 'app_dep_on_exports_and_databinding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/lib_with_exports:lib_with_exports_and_databinding'],",
        ")");
  }

  @Test
  public void testDependentLibraryJavaPackagesPassedFromLibraryWithExportsNoDatabinding()
      throws Exception {

    writeDataBindingFilesWithExports();

    ConfiguredTarget ctapp =
        getConfiguredTarget("//java/android/binary:app_dep_on_exports_no_databinding");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));
    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "app_dep_on_exports_no_databinding.jar"));

    ImmutableList<String> expectedJavacopts =
        ImmutableList.of(
            "-Aandroid.databinding.directDependencyPkgs=[android.library1,android.library2]");
    assertThat(getJavacArguments(binCompileAction)).containsAtLeastElementsIn(expectedJavacopts);
  }

  @Test
  public void testDependentLibraryJavaPackagesPassedFromLibraryWithExportsAndDatabinding()
      throws Exception {

    writeDataBindingFilesWithExports();

    ConfiguredTarget ctapp =
        getConfiguredTarget("//java/android/binary:app_dep_on_exports_and_databinding");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));
    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "app_dep_on_exports_and_databinding.jar"));

    ImmutableList<String> expectedJavacopts =
        ImmutableList.of(
            "-Aandroid.databinding.directDependencyPkgs="
                + "[android.lib_with_exports,android.library1,android.library2]");
    assertThat(getJavacArguments(binCompileAction)).containsAtLeastElementsIn(expectedJavacopts);
  }

  @Test
  public void testNoDependentLibraryJavaPackagesIsEmptyBrackets() throws Exception {

    scratch.file(
        "java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app_databinding_no_deps',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = [],",
        ")");

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app_databinding_no_deps");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));
    JavaCompileAction binCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "app_databinding_no_deps.jar"));

    ImmutableList<String> expectedJavacopts =
        ImmutableList.of("-Aandroid.databinding.directDependencyPkgs=[]");
    assertThat(getJavacArguments(binCompileAction)).containsAtLeastElementsIn(expectedJavacopts);
  }

  @Test
  public void dataBinding_aapt2PackageAction_withoutAndroidX_doesNotPassAndroidXFlag()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
    writeDataBindingFiles();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact aapt2PackageArtifact = getAapt2PackgeActionArtifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(aapt2PackageArtifact))
        .doesNotContain("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_aapt2PackageAction_withAndroidX_passesAndroidXFlag() throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2",
        "--android_databinding_use_v3_4_args",
        "--android_databinding_use_androidx");
    writeDataBindingFiles();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact aapt2PackageArtifact = getAapt2PackgeActionArtifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(aapt2PackageArtifact))
        .contains("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_processingDatabindingAction_withoutAndroidX_doesNotPassAndroidXFlag()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
    writeDataBindingFiles();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact processingDatabindingArtifact = getProcessingDatabindingArtifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(processingDatabindingArtifact))
        .doesNotContain("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_processingDatabindingAction_withAndroidX_passesAndroidXFlag()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2",
        "--android_databinding_use_v3_4_args",
        "--android_databinding_use_androidx");
    writeDataBindingFiles();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact processingDatabindingArtifact = getProcessingDatabindingArtifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(processingDatabindingArtifact))
        .contains("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_compileLibraryResourcesAction_withoutAndroidX_doesNotPassAndroidXFlag()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
    writeDataBindingFiles();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact compileLibraryResourcesArtifact = getCompileLibraryResourcesArtifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(compileLibraryResourcesArtifact))
        .doesNotContain("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_compileLibraryResourcesAction_withAndroidX_passesAndroidXFlag()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2",
        "--android_databinding_use_v3_4_args",
        "--android_databinding_use_androidx");
    writeDataBindingFiles();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact compileLibraryResourcesArtifact = getCompileLibraryResourcesArtifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(compileLibraryResourcesArtifact))
        .contains("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_shrinkAapt2Action_withoutAndroidX_doesNotPassAndroidXFlag()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
    writeDataBindingFilesWithShrinkage();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact shrinkAapt2Artifact = getShrinkAapt2Artifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(shrinkAapt2Artifact))
        .doesNotContain("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_shrinkAapt2Action_withAndroidX_passesAndroidXFlag() throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2",
        "--android_databinding_use_v3_4_args",
        "--android_databinding_use_androidx");
    writeDataBindingFilesWithShrinkage();

    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    Artifact shrinkAapt2Artifact = getShrinkAapt2Artifact(allArtifacts);
    assertThat(getGeneratingSpawnActionArgs(shrinkAapt2Artifact))
        .contains("--useDataBindingAndroidX");
  }

  @Test
  public void dataBinding_androidLocalTest_dataBindingDisabled_doesNotUseDataBindingFlags()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
    writeDataBindingFiles();
    writeNonDataBindingLocalTestFiles();

    if (platformBasedToolchains()) {
      // TODO(b/161709111): With platforms, the below fails with
      // "no attribute `$android_sdk_toolchain_type`" on AspectAwareAttributeMapper.
      return;
    }

    ConfiguredTarget testTarget =
        getConfiguredTarget("//javatests/android/test:databinding_enabled_test");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(testTarget));
    JavaCompileAction testCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "databinding_enabled_test-class.jar"));
    ImmutableList<String> expectedMissingJavacopts =
        ImmutableList.of(
            "-Aandroid.databinding.sdkDir=/not/used",
            "-Aandroid.databinding.artifactType=APPLICATION",
            "-Aandroid.databinding.exportClassListOutFile=/tmp/exported_classes",
            "-Aandroid.databinding.modulePackage=android.test",
            "-Aandroid.databinding.minApi=14",
            "-Aandroid.databinding.enableV2=1",
            "-Aandroid.databinding.directDependencyPkgs=[android.library]");
    assertThat(getJavacArguments(testCompileAction)).containsNoneIn(expectedMissingJavacopts);

    JavaCompileInfo javaCompileInfo =
        testCompileAction
            .getExtraActionInfo(actionKeyContext)
            .getExtension(JavaCompileInfo.javaCompileInfo);
    assertThat(javaCompileInfo.getJavacOptList()).containsNoneIn(expectedMissingJavacopts);
  }

  @Test
  public void dataBinding_androidLocalTest_dataBindingEnabled_usesDataBindingFlags()
      throws Exception {
    useConfiguration(
        "--experimental_android_databinding_v2", "--android_databinding_use_v3_4_args");
    writeDataBindingFiles();
    writeDataBindingLocalTestFiles();

    if (platformBasedToolchains()) {
      // TODO(b/161709111): With platforms, the below fails with
      // "no attribute `$android_sdk_toolchain_type`" on AspectAwareAttributeMapper.
      return;
    }

    ConfiguredTarget testTarget =
        getConfiguredTarget("//javatests/android/test:databinding_enabled_test");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(testTarget));
    JavaCompileAction testCompileAction =
        (JavaCompileAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(allArtifacts, "databinding_enabled_test-class.jar"));
    String dataBindingFilesDir =
        targetConfig
            .getBinDirectory(RepositoryName.MAIN)
            .getExecPath()
            .getRelative("javatests/android/test/databinding/databinding_enabled_test")
            .getPathString();
    String inputDir = dataBindingFilesDir + "/" + DataBinding.DEP_METADATA_INPUT_DIR;
    String outputDir = dataBindingFilesDir + "/" + DataBinding.METADATA_OUTPUT_DIR;
    ImmutableList<String> expectedJavacopts =
        ImmutableList.of(
            "-Aandroid.databinding.dependencyArtifactsDir=" + inputDir,
            "-Aandroid.databinding.aarOutDir=" + outputDir,
            "-Aandroid.databinding.sdkDir=/not/used",
            "-Aandroid.databinding.artifactType=APPLICATION",
            "-Aandroid.databinding.exportClassListOutFile=/tmp/exported_classes",
            "-Aandroid.databinding.modulePackage=android.test",
            "-Aandroid.databinding.minApi=14",
            "-Aandroid.databinding.enableV2=1",
            "-Aandroid.databinding.directDependencyPkgs=[android.library]");
    assertThat(getJavacArguments(testCompileAction)).containsAtLeastElementsIn(expectedJavacopts);

    JavaCompileInfo javaCompileInfo =
        testCompileAction
            .getExtraActionInfo(actionKeyContext)
            .getExtension(JavaCompileInfo.javaCompileInfo);
    assertThat(javaCompileInfo.getJavacOptList()).containsAtLeastElementsIn(expectedJavacopts);
  }

  private Artifact getAapt2PackgeActionArtifact(Set<Artifact> allArtifacts) {
    return getArtifactForTool(allArtifacts, /* toolName= */ "AAPT2_PACKAGE");
  }

  private Artifact getProcessingDatabindingArtifact(Set<Artifact> allArtifacts) {
    return getArtifactForTool(allArtifacts, /* toolName= */ "PROCESS_DATABINDING");
  }

  private Artifact getCompileLibraryResourcesArtifact(Set<Artifact> allArtifacts) {
    return getArtifactForTool(allArtifacts, /* toolName= */ "COMPILE_LIBRARY_RESOURCES");
  }

  private Artifact getShrinkAapt2Artifact(Set<Artifact> allArtifacts) {
    return getArtifactForTool(allArtifacts, /* toolName= */ "SHRINK_AAPT2");
  }

  private Artifact getArtifactForTool(Set<Artifact> allArtifacts, String toolName) {
    Artifact artifact = getFirstArtifactMatching(allArtifacts, isSpawnActionWithTool(toolName));
    assertWithMessage("Expected to find an artifact using tool: %s", toolName)
        .that(artifact)
        .isNotNull();
    return artifact;
  }

  private Predicate<Artifact> isSpawnActionWithTool(String toolName) {
    return artifact -> {
      List<String> actionArgs;
      try {
        actionArgs = getGeneratingSpawnActionArgs(artifact);
      } catch (Exception e) {
        // Some artifacts are not compatible spawn artifacts.
        return false;
      }
      int toolIndicatorIndex = actionArgs.indexOf("--tool");
      return toolIndicatorIndex > -1
          && toolIndicatorIndex + 1 < actionArgs.size()
          && actionArgs.get(toolIndicatorIndex + 1).equals(toolName);
    };
  }
}
