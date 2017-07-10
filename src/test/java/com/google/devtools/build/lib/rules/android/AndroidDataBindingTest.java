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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Bazel's Android data binding support.
 */
@RunWith(JUnit4.class)
public class AndroidDataBindingTest extends AndroidBuildViewTestCase {
  private void writeDataBindingFiles() throws Exception {
    scratch.file("java/android/library/BUILD",
        "android_library(",
        "    name = 'lib_with_data_binding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib.java'],",
        "    resource_files = [],",
        ")");
    scratch.file("java/android/library/MyLib.java",
        "package android.library; public class MyLib {};");
    scratch.file("java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/library:lib_with_data_binding'],",
        ")");
    scratch.file("java/android/binary/MyApp.java",
        "package android.binary; public class MyApp {};");
  }

  private void writeDataBindingFilesWithNoResourcesDep() throws Exception {
    scratch.file("java/android/lib_with_resource_files/BUILD",
        "android_library(",
        "    name = 'lib_with_resource_files',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['LibWithResourceFiles.java'],",
        "    resource_files = glob(['res/**']),",
        ")");
    scratch.file("java/android/lib_with_resource_files/LibWithResourceFiles.java",
        "package android.lib_with_resource_files; public class LibWithResourceFiles {};");

    scratch.file("java/android/lib_no_resource_files/BUILD",
        "android_library(",
        "    name = 'lib_no_resource_files',",
        "    enable_data_binding = 1,",
        "    srcs = ['LibNoResourceFiles.java'],",
        "    deps = ['//java/android/lib_with_resource_files'],",
        ")");
    scratch.file("java/android/lib_no_resource_files/LibNoResourceFiles.java",
        "package android.lib_no_resource_files; public class LibNoResourceFiles {};");

    scratch.file("java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/lib_no_resource_files'],",
        ")");
    scratch.file("java/android/binary/MyApp.java",
        "package android.binary; public class MyApp {};");
  }

  /**
   * Returns the .params file contents of a {@link JavaCompileAction}
   */
  private Iterable<String> getParamFileContents(JavaCompileAction action) {
    Artifact paramFile = getFirstArtifactEndingWith(action.getInputs(), "-2.params");
    return ((ParameterFileWriteAction) getGeneratingAction(paramFile)).getContents();
  }

  @Test
  public void basicDataBindingIntegrationParallelResourceProcessing() throws Exception {
    useConfiguration("--experimental_use_parallel_android_resource_processing");
    basicDataBindingIntegration();
  }

  @Test
  public void basicDataBindingIntegrationLegacyResourceProcessing() throws Exception {
    useConfiguration("--noexperimental_use_parallel_android_resource_processing");
    basicDataBindingIntegration();
  }

  private void basicDataBindingIntegration() throws Exception {
    writeDataBindingFiles();
    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    // "Data binding"-enabled targets invoke resource processing with a request for data binding
    // output:
    Artifact libResourceInfoOutput = getFirstArtifactEndingWith(allArtifacts,
        "databinding/lib_with_data_binding/layout-info.zip");
    assertThat(getGeneratingSpawnActionArgs(libResourceInfoOutput))
        .containsAllOf("--dataBindingInfoOut", libResourceInfoOutput.getExecPathString())
        .inOrder();

    Artifact binResourceInfoOutput = getFirstArtifactEndingWith(allArtifacts,
        "databinding/app/layout-info.zip");
    assertThat(getGeneratingSpawnActionArgs(binResourceInfoOutput))
        .containsAllOf("--dataBindingInfoOut", binResourceInfoOutput.getExecPathString())
        .inOrder();

    // Java compilation includes the data binding annotation processor, the resource processor's
    // output, and the auto-generated DataBindingInfo.java the annotation processor uses to figure
    // out what to do:
    JavaCompileAction libCompileAction = (JavaCompileAction) getGeneratingAction(
        getFirstArtifactEndingWith(allArtifacts, "lib_with_data_binding.jar"));
    assertThat(libCompileAction.getProcessorNames())
        .contains("android.databinding.annotationprocessor.ProcessDataBinding");
    assertThat(ActionsTestUtil.prettyArtifactNames(libCompileAction.getInputs()))
        .containsAllOf(
            "java/android/library/databinding/lib_with_data_binding/layout-info.zip",
            "java/android/library/databinding/lib_with_data_binding/DataBindingInfo.java");

    JavaCompileAction binCompileAction = (JavaCompileAction) getGeneratingAction(
        getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    assertThat(binCompileAction.getProcessorNames())
        .contains("android.databinding.annotationprocessor.ProcessDataBinding");
    assertThat(ActionsTestUtil.prettyArtifactNames(binCompileAction.getInputs()))
        .containsAllOf(
            "java/android/binary/databinding/app/layout-info.zip",
            "java/android/binary/databinding/app/DataBindingInfo.java");
  }

  @Test
  public void dataBindingCompilationUsesMetadataFromDepsParallelResourceProcessing()
      throws Exception {
    useConfiguration("--experimental_use_parallel_android_resource_processing");
    dataBindingCompilationUsesMetadataFromDeps();
  }

  @Test
  public void dataBindingCompilationUsesMetadataFromDepsLegacyResourceProcessing()
      throws Exception {
    useConfiguration("--noexperimental_use_parallel_android_resource_processing");
    dataBindingCompilationUsesMetadataFromDeps();
  }

  private void dataBindingCompilationUsesMetadataFromDeps() throws Exception {
    writeDataBindingFiles();
    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));

    // The library's compilation doesn't include any of the -setter_store.bin, layoutinfo.bin, etc.
    // files that store a dependency's data binding results (since the library has no deps).
    // We check that they don't appear as compilation inputs.
    JavaCompileAction libCompileAction = (JavaCompileAction)
        getGeneratingAction(getFirstArtifactEndingWith(allArtifacts, "lib_with_data_binding.jar"));
    assertThat(
        Iterables.filter(libCompileAction.getInputs(),
            ActionsTestUtil.getArtifactSuffixMatcher(".bin")))
        .isEmpty();

    // The binary's compilation includes the library's data binding results.
    JavaCompileAction binCompileAction = (JavaCompileAction) getGeneratingAction(
        getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    Iterable<Artifact> depMetadataInputs = Iterables.filter(
        binCompileAction.getInputs(), ActionsTestUtil.getArtifactSuffixMatcher(".bin"));
    final String depMetadataBaseDir = Iterables.getFirst(depMetadataInputs, null).getExecPath()
        .getParentDirectory().toString();
        ActionsTestUtil.execPaths(Iterables.filter(
        binCompileAction.getInputs(), ActionsTestUtil.getArtifactSuffixMatcher(".bin")));
    assertThat(ActionsTestUtil.execPaths(depMetadataInputs)).containsExactly(
        depMetadataBaseDir + "/android.library-android.library-setter_store.bin",
        depMetadataBaseDir + "/android.library-android.library-layoutinfo.bin",
        depMetadataBaseDir + "/android.library-android.library-br.bin");
  }

  @Test
  public void dataBindingAnnotationProcessorFlags() throws Exception {
    writeDataBindingFiles();
    ConfiguredTarget ctapp = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ctapp));
    JavaCompileAction binCompileAction = (JavaCompileAction) getGeneratingAction(
        getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    String dataBindingFilesDir = targetConfig.getBinDirectory(RepositoryName.MAIN).getExecPath()
        .getRelative("java/android/binary/databinding/app").getPathString();
    assertThat(getParamFileContents(binCompileAction))
         .containsAllOf(
             "-Aandroid.databinding.bindingBuildFolder=" + dataBindingFilesDir,
             "-Aandroid.databinding.generationalFileOutDir=" + dataBindingFilesDir,
             "-Aandroid.databinding.sdkDir=/not/used",
             "-Aandroid.databinding.artifactType=APPLICATION",
             "-Aandroid.databinding.xmlOutDir=" + dataBindingFilesDir,
             "-Aandroid.databinding.exportClassListTo=/tmp/exported_classes",
             "-Aandroid.databinding.modulePackage=android.binary",
             "-Aandroid.databinding.minApi=14",
             "-Aandroid.databinding.printEncodedErrors=0");
  }

  @Test
  public void dataBindingIncludesTransitiveDepsForLibsWithNoResources() throws Exception {
    writeDataBindingFilesWithNoResourcesDep();
    ConfiguredTarget ct = getConfiguredTarget("//java/android/binary:app");
    Set<Artifact> allArtifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(ct));

    // Data binding resource processing outputs are expected for the app and libs with resources.
    assertThat(getFirstArtifactEndingWith(allArtifacts,
        "databinding/lib_with_resource_files/layout-info.zip")).isNotNull();
    assertThat(getFirstArtifactEndingWith(allArtifacts, "databinding/app/layout-info.zip"))
        .isNotNull();

    // Compiling the app's Java source includes data binding metadata from the resource-equipped
    // lib, but not the resource-empty one.
    JavaCompileAction binCompileAction = (JavaCompileAction) getGeneratingAction(
        getFirstArtifactEndingWith(allArtifacts, "app.jar"));
    List<String> appJarInputs = ActionsTestUtil.prettyArtifactNames(binCompileAction.getInputs());
    String libWithResourcesMetadataBaseDir = "java/android/binary/databinding/app/"
        + "dependent-lib-artifacts/java/android/lib_with_resource_files/databinding/"
        + "lib_with_resource_files/bin-files/android.lib_with_resource_files-";
    assertThat(appJarInputs).containsAllOf(
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
    Iterable<Artifact> libArtifacts = getFilesToBuild(ct);

    assertThat(getFirstArtifactEndingWith(libArtifacts, "_resources.jar")).isNull();
    assertThat(getFirstArtifactEndingWith(libArtifacts, "layout-info.zip")).isNull();

    JavaCompileAction libCompileAction = (JavaCompileAction) getGeneratingAction(
        getFirstArtifactEndingWith(libArtifacts, "lib_no_resource_files.jar"));
    // The annotation processor is attached to the Java compilation:
    assertThat(getParamFileContents(libCompileAction)).containsAllOf(
        "--processors",
        "android.databinding.annotationprocessor.ProcessDataBinding");
    // The dummy .java file with annotations that trigger the annotation process is present:
    assertThat(ActionsTestUtil.prettyArtifactNames(libCompileAction.getInputs()))
        .contains("java/android/lib_no_resource_files/databinding/lib_no_resource_files/"
            + "DataBindingInfo.java");
  }

  @Test
  public void missingDataBindingAttributeStillAnalyzes() throws Exception {
    // When a library is missing enable_data_binding = 1, we expect it to fail in execution (because
    // aapt doesn't know how to read the data binding expressions). But analysis should work.
    scratch.file("java/android/library/BUILD",
        "android_library(",
        "    name = 'lib_with_data_binding',",
        "    enable_data_binding = 1,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyLib.java'],",
        "    resource_files = [],",
        ")");
    scratch.file("java/android/library/MyLib.java",
        "package android.library; public class MyLib {};");
    scratch.file("java/android/binary/BUILD",
        "android_binary(",
        "    name = 'app',",
        "    enable_data_binding = 0,",
        "    manifest = 'AndroidManifest.xml',",
        "    srcs = ['MyApp.java'],",
        "    deps = ['//java/android/library:lib_with_data_binding'],",
        ")");
    scratch.file("java/android/binary/MyApp.java",
        "package android.binary; public class MyApp {};");
    assertThat(getConfiguredTarget("//java/android/binary:app")).isNotNull();
  }
}
