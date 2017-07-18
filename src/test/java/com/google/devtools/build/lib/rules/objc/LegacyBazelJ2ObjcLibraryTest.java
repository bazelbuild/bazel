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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Legacy test: These tests test --experimental_objc_crosstool=off. See README in
 * devtools.build.lib.rules.objc
 */
@RunWith(JUnit4.class)
public class LegacyBazelJ2ObjcLibraryTest extends BazelJ2ObjcLibraryTest {
  @Override
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.OFF;
  }

  @Override
  protected BuildConfiguration getGenfilesConfig() {
    return targetConfig;
  }

  @Test
  public void testObjcCompileAction() throws Exception {
    Artifact archive = j2objcArchive("//java/com/google/dummy/test:transpile", "test");
    CommandAction compileAction = getObjcCompileAction(archive, "test.o");
    assertThat(Artifact.toRootRelativePaths(compileAction.getInputs())).containsExactly(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.m",
        "tools/objc/xcrunwrapper");
    assertThat(compileAction.getArguments()).containsAllOf("-fno-objc-arc", "-fno-strict-overflow");
  }

  @Test
  public void testJ2ObjcSourcesCompilationAndLinking() throws Exception {
    addSimpleBinaryTarget("//java/com/google/dummy/test:transpile");

    checkObjcArchiveAndLinkActions("//app:app", "libtest_j2objc.a", "test.o", ImmutableList.of(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.m"));
  }

  @Test
  public void testNestedJ2ObjcLibraryDeps() throws Exception {
    scratch.file("java/com/google/dummy/dummy.java");
    scratch.file("java/com/google/dummy/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(",
        "    name = 'dummy',",
        "    srcs = ['dummy.java'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = [",
        "        ':dummy',",
        "        '//java/com/google/dummy/test:transpile',",
        "    ])");
    addSimpleBinaryTarget("//java/com/google/dummy:transpile");

    checkObjcArchiveAndLinkActions("//app:app", "libtest_j2objc.a", "test.o", ImmutableList.of(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.m"));

    checkObjcArchiveAndLinkActions("//app:app", "libdummy_j2objc.a", "dummy.o", ImmutableList.of(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/dummy/_j2objc/dummy/java/com/google/dummy/dummy.h",
        "java/com/google/dummy/_j2objc/dummy/java/com/google/dummy/dummy.m"));
  }

  @Test
  public void testJ2ObjcTranspiledHeaderInCompilationAction() throws Exception {
    scratch.file("app/lib.m");
    scratch.file("app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = ['//java/com/google/dummy/test:transpile'])");

    checkObjcCompileActions(getBinArtifact("liblib.a", getConfiguredTarget("//app:lib")), "lib.o",
        ImmutableList.of(
            "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h"));
  }

  @Test
  public void testJ2ObjcDeadCodeRemovalActionWithOptFlag() throws Exception {
    useConfiguration("--j2objc_dead_code_removal");
    addSimpleJ2ObjcLibraryWithEntryClasses();
    addSimpleBinaryTarget("//java/com/google/app/test:transpile");

    ConfiguredTarget appTarget = getConfiguredTarget("//app:app", getAppleCrosstoolConfiguration());
    Artifact prunedArchive =
        getBinArtifact(
            "_j2objc_pruned/app/java/com/google/app/test/libtest_j2objc_pruned.a", appTarget);
    Artifact paramFile =
        getBinArtifact("_j2objc_pruned/app/java/com/google/app/test/test.param.j2objc", appTarget);

    ConfiguredTarget javaTarget =
        getConfiguredTarget("//java/com/google/app/test:test", getAppleCrosstoolConfiguration());
    Artifact inputArchive = getBinArtifact("libtest_j2objc.a", javaTarget);
    Artifact headerMappingFile = getBinArtifact("test.mapping.j2objc", javaTarget);
    Artifact dependencyMappingFile = getBinArtifact("test.dependency_mapping.j2objc", javaTarget);
    Artifact archiveSourceMappingFile =
        getBinArtifact("test.archive_source_mapping.j2objc", javaTarget);
    String execPath =
        javaTarget.getConfiguration().getBinDirectory(RepositoryName.MAIN).getExecPath() + "/";

    ParameterFileWriteAction paramFileAction =
        (ParameterFileWriteAction) getGeneratingAction(paramFile);
    assertContainsSublist(
        ImmutableList.copyOf(paramFileAction.getContents()),
        new ImmutableList.Builder<String>()
            .add("--input_archive")
            .add(inputArchive.getExecPathString())
            .add("--output_archive")
            .add(prunedArchive.getExecPathString())
            .add("--dummy_archive")
            .add(execPath + "tools/objc/libdummy_lib.a")
            .add("--xcrunwrapper")
            .add("tools/objc/xcrunwrapper")
            .add("--dependency_mapping_files")
            .add(dependencyMappingFile.getExecPathString())
            .add("--header_mapping_files")
            .add(headerMappingFile.getExecPathString())
            .add("--archive_source_mapping_files")
            .add(archiveSourceMappingFile.getExecPathString())
            .add("--entry_classes")
            .add("com.google.app.test.test")
            .build());

    SpawnAction deadCodeRemovalAction = (SpawnAction) getGeneratingAction(prunedArchive);
    assertContainsSublist(
        deadCodeRemovalAction.getArguments(),
        new ImmutableList.Builder<String>()
            .add("tools/objc/j2objc_dead_code_pruner.py")
            .add("@" + paramFile.getExecPathString())
            .build());
    assertThat(deadCodeRemovalAction.getOutputs()).containsExactly(prunedArchive);
  }

  @Test
  public void testCompileActionTemplateFromGenJar() throws Exception {
    useConfiguration("--cpu=ios_i386", "--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    Artifact archive = j2objcArchive("//java/com/google/app/test:transpile", "test");
    CommandAction archiveAction = (CommandAction) getGeneratingAction(archive);
    Artifact objectFilesFromGenJar =
        getFirstArtifactEndingWith(archiveAction.getInputs(), "source_files");

    assertThat(objectFilesFromGenJar.isTreeArtifact()).isTrue();
    assertThat(objectFilesFromGenJar.getRootRelativePathString())
        .isEqualTo(
            "java/com/google/app/test/_objs/test/java/com/google/app/test/_j2objc/"
                + "src_jar_files/test/source_files");

    SpawnActionTemplate actionTemplate =
        (SpawnActionTemplate) getActionGraph().getGeneratingAction(objectFilesFromGenJar);
    Artifact sourceFilesFromGenJar =
        getFirstArtifactEndingWith(actionTemplate.getInputs(), "source_files");
    Artifact headerFilesFromGenJar =
        getFirstArtifactEndingWith(actionTemplate.getInputs(), "header_files");
    assertThat(sourceFilesFromGenJar.getRootRelativePathString())
        .isEqualTo("java/com/google/app/test/_j2objc/src_jar_files/test/source_files");
    assertThat(headerFilesFromGenJar.getRootRelativePathString())
        .isEqualTo("java/com/google/app/test/_j2objc/src_jar_files/test/header_files");

    // The files contained inside the tree artifacts are not known until execution time.
    // Therefore we need to fake some files inside them to test the action template in this
    // analysis-time test.
    TreeFileArtifact oneSourceFileFromGenJar =
        ActionInputHelper.treeFileArtifact(sourceFilesFromGenJar, "children1.m");
    TreeFileArtifact oneObjFileFromGenJar =
        ActionInputHelper.treeFileArtifact(objectFilesFromGenJar, "children1.o");
    Iterable<SpawnAction> compileActions =
        actionTemplate.generateActionForInputArtifacts(
            ImmutableList.of(oneSourceFileFromGenJar), ArtifactOwner.NULL_OWNER);
    SpawnAction compileAction = Iterables.getOnlyElement(compileActions);
    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/dummy/test:transpile");
    String genfilesFragment =
        j2objcLibraryTarget.getConfiguration().getGenfilesFragment().toString();
    String binFragment = j2objcLibraryTarget.getConfiguration().getBinFragment().toString();
    AppleConfiguration appleConfiguration =
        j2objcLibraryTarget.getConfiguration().getFragment(AppleConfiguration.class);

    assertThat(compileAction.getArguments())
        .containsExactlyElementsIn(
            new ImmutableList.Builder<String>()
                .add(MOCK_XCRUNWRAPPER_PATH)
                .add("clang")
                .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
                .add("-fexceptions")
                .add("-fasm-blocks")
                .add("-fobjc-abi-version=2")
                .add("-fobjc-legacy-dispatch")
                .add("-DOS_IOS")
                .add("-mios-simulator-version-min=1.0")
                .add("-arch", "i386")
                .add("-isysroot")
                .add(AppleToolchain.sdkDir())
                .add("-F")
                .add(AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")
                .add("-F")
                .add(AppleToolchain.platformDeveloperFrameworkDir(appleConfiguration))
                .add("-O0")
                .add("-DDEBUG=1")
                .add("-iquote")
                .add(".")
                .add("-iquote")
                .add(genfilesFragment)
                .add("-I")
                .add(binFragment + "/java/com/google/app/test/_j2objc/test")
                .add("-I")
                .add(headerFilesFromGenJar.getExecPathString())
                .add("-fno-strict-overflow")
                .add("-fno-objc-arc")
                .add("-c")
                .add(oneSourceFileFromGenJar.getExecPathString())
                .add("-o")
                .add(oneObjFileFromGenJar.getExecPathString())
                .build())
        .inOrder();
  }

}
