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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.cpp.UmbrellaHeaderAction;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayOutputStream;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit test for Java source file translation into ObjC in {@link
 * com.google.devtools.build.lib.rules.java.JavaLibrary} and translated file compilation and linking
 * in {@link ObjcBinary} and {@link J2ObjcLibrary}.
 */
@RunWith(JUnit4.class)
public class BazelJ2ObjcLibraryTest extends J2ObjcLibraryTest {
  @Test
  public void testJ2ObjCInformationExportedFromJ2ObjcLibrary() throws Exception {
    ConfiguredTarget j2objcLibraryTarget = getConfiguredTarget(
        "//java/com/google/dummy/test:transpile");
    ObjcProvider provider = j2objcLibraryTarget.getProvider(ObjcProvider.class);
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY))).containsExactly(
        "third_party/java/j2objc/libjre_core_lib.a",
        "java/com/google/dummy/test/libtest_j2objc.a");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.HEADER))).containsExactly(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h");

    String execPath = j2objcLibraryTarget.getConfiguration().getBinDirectory(RepositoryName.MAIN)
        .getExecPath() + "/";
    assertThat(PathFragment.safePathStrings(provider.get(ObjcProvider.INCLUDE)))
        .containsExactly(execPath + "java/com/google/dummy/test/_j2objc/test");
  }

  @Test
  public void testJ2ObjCInformationExportedWithGeneratedJavaSources() throws Exception {
    scratch.file("java/com/google/test/in.txt");
    scratch.file("java/com/google/test/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "genrule(",
        "    name = 'dummy_gen',",
        "    srcs = ['in.txt'],",
        "    outs = ['test.java'],",
        "    cmd = 'dummy')",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = [':test.java'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = ['test'])");

    ConfiguredTarget target = getConfiguredTarget("//java/com/google/test:transpile");
    ObjcProvider provider = target.getProvider(ObjcProvider.class);
    String genfilesFragment = target.getConfiguration().getGenfilesFragment().toString();
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY))).containsExactly(
        "third_party/java/j2objc/libjre_core_lib.a",
        "java/com/google/test/libtest_j2objc.a");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.HEADER))).containsExactly(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/test/_j2objc/test/" + genfilesFragment + "/java/com/google/test/test.h");

    String execPath = target.getConfiguration().getBinDirectory(RepositoryName.MAIN)
        .getExecPath() + "/";
    assertThat(PathFragment.safePathStrings(provider.get(ObjcProvider.INCLUDE))).containsExactly(
        execPath + "java/com/google/test/_j2objc/test/" + genfilesFragment,
        execPath + "java/com/google/test/_j2objc/test");
  }

  @Test
  public void testJ2ObjcProtoRuntimeLibraryAndHeadersExported() throws Exception {
    scratch.file("java/com/google/dummy/test/proto/test.java");
    scratch.file("java/com/google/dummy/test/proto/test.proto");
    scratch.file("java/com/google/dummy/test/proto/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        "    java_api_version = 2,",
        "    j2objc_api_version = 1)",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_proto'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = ['test'])");

    ConfiguredTarget j2objcLibraryTarget = getConfiguredTarget(
        "//java/com/google/dummy/test/proto:transpile");
    ObjcProvider provider = j2objcLibraryTarget.getProvider(ObjcProvider.class);
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY))).containsExactly(
        "third_party/java/j2objc/libjre_core_lib.a",
        "third_party/java/j2objc/libproto_runtime.a",
        "java/com/google/dummy/test/proto/libtest_j2objc.a",
        "java/com/google/dummy/test/proto/libtest_proto_j2objc.a");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.HEADER))).containsExactly(
        "third_party/java/j2objc/jre_core.h",
        "third_party/java/j2objc/runtime.h",
        "java/com/google/dummy/test/proto/test.j2objc.pb.h",
        "java/com/google/dummy/test/proto/_j2objc/test/java/com/google/dummy/test/proto/test.h");
  }

  @Test
  public void testJ2ObjcHeaderMapExportedInJavaLibrary() throws Exception {
    scratch.file("java/com/google/transpile/BUILD",
        "java_library(name = 'dummy',",
        "    srcs = ['dummy.java'])");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    J2ObjcMappingFileProvider provider = target.getProvider(J2ObjcMappingFileProvider.class);

    assertThat(Iterables.getOnlyElement(provider.getHeaderMappingFiles())
        .getRootRelativePath().toString()).isEqualTo(
            "java/com/google/transpile/dummy.mapping.j2objc");
  }

  @Test
  public void testDepsJ2ObjcHeaderMapExportedInJavaLibraryWithNoSourceFile() throws Exception {
    scratch.file("java/com/google/transpile/BUILD",
        "java_library(name = 'dummy',",
        "    exports = ['//java/com/google/dep:dep'])");
    scratch.file("java/com/google/dep/BUILD",
        "java_library(name = 'dep',",
        "    srcs = ['dummy.java'])");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    J2ObjcMappingFileProvider provider = target.getProvider(J2ObjcMappingFileProvider.class);

    assertThat(Iterables.getOnlyElement(provider.getHeaderMappingFiles())
        .getRootRelativePath().toString()).isEqualTo(
            "java/com/google/dep/dep.mapping.j2objc");
  }

  @Test
  public void testJ2ObjcProtoClassMappingFilesExportedInJavaLibrary() throws Exception {
    scratch.file("java/com/google/dummy/test/proto/test.java");
    scratch.file("java/com/google/dummy/test/proto/test.proto");
    scratch.file("java/com/google/dummy/test/proto/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        "    java_api_version = 2,",
        "    j2objc_api_version = 1)",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_proto'])");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget(
        "//java/com/google/dummy/test/proto:test");
    ConfiguredTarget proto =
        getConfiguredTarget(
            "//java/com/google/dummy/test/proto:test", getAppleCrosstoolConfiguration());
    J2ObjcMappingFileProvider provider = target.getProvider(J2ObjcMappingFileProvider.class);
    Artifact classMappingFile = getGenfilesArtifact("test.clsmap.properties", proto);

    assertThat(provider.getClassMappingFiles()).containsExactly(classMappingFile);
  }

  @Test
  public void testJavaProtoLibraryWithProtoLibrary() throws Exception {
    scratch.file("x/BUILD",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        "    java_api_version = 2,",
        "    j2objc_api_version = 1)",
        "",
        "java_proto_library(",
        "    name = 'test_java_proto',",
        "    deps = [':test_proto'])",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_java_proto'])");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//x:test");
    ConfiguredTarget test = getConfiguredTarget("//x:test_proto", getAppleCrosstoolConfiguration());
    J2ObjcMappingFileProvider provider = target.getProvider(J2ObjcMappingFileProvider.class);
    Artifact classMappingFile = getGenfilesArtifact("test.clsmap.properties", test);
    assertThat(provider.getClassMappingFiles()).containsExactly(classMappingFile);

    ObjcProvider objcProvider = target.getProvider(ObjcProvider.class);
    Artifact headerFile = getGenfilesArtifact("test.j2objc.pb.h", test);
    Artifact sourceFile = getGenfilesArtifact("test.j2objc.pb.m", test);
    assertThat(objcProvider.get(ObjcProvider.HEADER)).contains(headerFile);
    assertThat(objcProvider.get(ObjcProvider.SOURCE)).contains(sourceFile);
  }

  @Test
  public void testJ2ObjcInfoExportedInJavaImport() throws Exception {
    scratch.file("java/com/google/transpile/BUILD",
        "java_import(name = 'dummy',",
        "    jars = ['dummy.jar'],",
        "    srcjar = 'dummy.srcjar',",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    J2ObjcMappingFileProvider provider = target.getProvider(J2ObjcMappingFileProvider.class);

    assertThat(Iterables.getOnlyElement(provider.getHeaderMappingFiles())
        .getRootRelativePath().toString()).isEqualTo(
        "java/com/google/transpile/dummy.mapping.j2objc");
    assertThat(provider.getClassMappingFiles()).isEmpty();
  }

  protected void checkObjcArchiveAndLinkActions(
      String targetLabel,
      String archiveFileName,
      String objFileName,
      Iterable<String> compilationInputExecPaths)
      throws Exception {
    String labelName = Label.parseAbsolute(targetLabel).getName();
    CommandAction linkAction =
        (CommandAction)
            getGeneratingAction(
                getBinArtifact(
                    String.format("%s_bin", labelName), getConfiguredTarget(targetLabel)));

    checkObjcCompileActions(
        getFirstArtifactEndingWith(linkAction.getInputs(), archiveFileName),
        objFileName, compilationInputExecPaths);
  }

  @Test
  public void testMissingEntryClassesError() throws Exception {
    useConfiguration("--j2objc_dead_code_removal");
    checkError("java/com/google/dummy", "transpile", J2ObjcLibrary.NO_ENTRY_CLASS_ERROR_MSG,
        "j2objc_library(name = 'transpile', deps = ['//java/com/google/dummy/test:test'])");
  }

  @Test
  public void testNoJ2ObjcDeadCodeRemovalActionWithoutOptFlag() throws Exception {
    useConfiguration("--noj2objc_dead_code_removal");
    addSimpleJ2ObjcLibraryWithEntryClasses();
    addSimpleBinaryTarget("//java/com/google/app/test:transpile");

    Artifact expectedPrunedSource = getBinArtifact(
        "_j2objc_pruned/app/java/com/google/app/test/_j2objc/test/"
        + "java/com/google/app/test/test_pruned.m", getConfiguredTarget("//app:app"));
    assertThat(getGeneratingAction(expectedPrunedSource)).isNull();
  }

  @Test
  public void testExplicitJreDeps() throws Exception {
    ConfiguredTarget j2objcLibraryTarget = getConfiguredTarget(
        "//java/com/google/dummy/test:transpile");
    ObjcProvider provider = j2objcLibraryTarget.getProvider(ObjcProvider.class);
    // jre_io_lib and jre_emul_lib should be excluded.
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY))).containsExactly(
        "third_party/java/j2objc/libjre_core_lib.a",
        "java/com/google/dummy/test/libtest_j2objc.a");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.HEADER))).containsExactly(
        "third_party/java/j2objc/jre_core.h",
        "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h");
  }

  @Test
  public void testTranspilationActionTreeArtifactOutputsFromSourceJar() throws Exception {
    useConfiguration("--ios_cpu=i386", "--ios_minimum_os=1.0");
    scratch.file("java/com/google/transpile/dummy.java");
    scratch.file("java/com/google/transpile/dummyjar.srcjar");
    scratch.file("java/com/google/transpile/BUILD",
        "java_library(name = 'dummy',",
        "    srcs = ['dummy.java', 'dummyjar.srcjar'],",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    ObjcProvider provider = target.getProvider(ObjcProvider.class);
    Artifact srcJarSources = getFirstArtifactEndingWith(
        provider.get(ObjcProvider.SOURCE), "source_files");
    Artifact srcJarHeaders = getFirstArtifactEndingWith(
        provider.get(ObjcProvider.HEADER), "header_files");
    assertThat(srcJarSources.getRootRelativePathString())
        .isEqualTo("java/com/google/transpile/_j2objc/src_jar_files/dummy/source_files");
    assertThat(srcJarHeaders.getRootRelativePathString())
        .isEqualTo("java/com/google/transpile/_j2objc/src_jar_files/dummy/header_files");
    assertThat(srcJarSources.isTreeArtifact()).isTrue();
    assertThat(srcJarHeaders.isTreeArtifact()).isTrue();
  }

  @Test
  public void testGeneratedTreeArtifactFromGenJar() throws Exception {
    useConfiguration("--ios_cpu=i386", "--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/app/test:transpile");
    ObjcProvider provider = j2objcLibraryTarget.getProvider(ObjcProvider.class);
    Artifact headers =
        getFirstArtifactEndingWith(provider.get(ObjcProvider.HEADER), "header_files");
    Artifact sources =
        getFirstArtifactEndingWith(provider.get(ObjcProvider.SOURCE), "source_files");
    assertThat(headers.isTreeArtifact()).isTrue();
    assertThat(sources.isTreeArtifact()).isTrue();

    SpawnAction j2objcAction = (SpawnAction) getGeneratingAction(headers);
    assertThat(j2objcAction.getOutputs()).containsAllOf(headers, sources);

    Artifact paramFile = getFirstArtifactEndingWith(j2objcAction.getInputs(), ".param.j2objc");
    ParameterFileWriteAction paramFileAction =
        (ParameterFileWriteAction) getGeneratingAction(paramFile);
    assertContainsSublist(
        ImmutableList.copyOf(paramFileAction.getContents()),
        ImmutableList.of(
            "--output_gen_source_dir",
            sources.getExecPathString(),
            "--output_gen_header_dir",
            headers.getExecPathString()));
  }

  @Test
  public void testCrosstoolCompilationSupport() throws Exception {
    MockObjcSupport.createCrosstoolPackage(mockToolsConfig);
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all");
    addSimpleBinaryTarget("//java/com/google/dummy/test:transpile");
    assertThat(getConfiguredTarget("//app:app")).isNotNull();
  }

  @Test
  public void testProtoCrosstoolCompilationSupport() throws Exception {
    MockObjcSupport.createCrosstoolPackage(mockToolsConfig);
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all");
    scratch.file("x/test.java");
    scratch.file("x/test.proto");
    scratch.file("x/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        "    cc_api_version = 2,",
        "    java_api_version = 2,",
        "    j2objc_api_version = 1)",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_proto'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = ['test'])");
    addSimpleBinaryTarget("//x:transpile");
    assertThat(getConfiguredTarget("//app:app")).isNotNull();
  }

  @Test
  public void testJ2ObjcHeaderMappingAction() throws Exception {
    scratch.file("java/com/google/transpile/BUILD",
        "java_library(name = 'lib1',",
        "    srcs = ['libOne.java', 'jar.srcjar'],",
        "    deps = [':lib2']",
        ")",
        "",
        "java_library(name = 'lib2',",
        "    srcs = ['libTwo.java'],",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget(
        "//java/com/google/transpile:lib1");
    J2ObjcMappingFileProvider mappingFileProvider =
        target.getProvider(J2ObjcMappingFileProvider.class);
    assertThat(Artifact.toRootRelativePaths(mappingFileProvider.getHeaderMappingFiles()))
        .containsExactly(
            "java/com/google/transpile/lib1.mapping.j2objc",
            "java/com/google/transpile/lib2.mapping.j2objc");

    Artifact mappingFile = getFirstArtifactEndingWith(
        mappingFileProvider.getHeaderMappingFiles(), "lib1.mapping.j2objc");
    SpawnAction headerMappingAction = (SpawnAction) getGeneratingAction(mappingFile);
    String execPath = target.getConfiguration().getBinDirectory(RepositoryName.MAIN)
        .getExecPath() + "/";
    assertThat(Artifact.toRootRelativePaths(headerMappingAction.getInputs()))
        .containsAllOf(
            "java/com/google/transpile/libOne.java", "java/com/google/transpile/jar.srcjar");
    assertThat(headerMappingAction.getArguments())
        .containsExactly(
            "tools/j2objc/j2objc_header_map.py",
            "--source_files",
            "java/com/google/transpile/libOne.java",
            "--source_jars",
            "java/com/google/transpile/jar.srcjar",
            "--output_mapping_file",
            execPath + "java/com/google/transpile/lib1.mapping.j2objc")
        .inOrder();
  }

  protected void checkObjcCompileActions(
      Artifact archiveFile, String objFileName, Iterable<String> compilationInputExecPaths)
      throws Exception {
    CommandAction compileAction = getObjcCompileAction(archiveFile, objFileName);
    assertThat(Artifact.toRootRelativePaths(compileAction.getInputs())).containsAllIn(
        compilationInputExecPaths);
  }

  protected CommandAction getObjcCompileAction(Artifact archiveFile, String objFileName)
      throws Exception {
    CommandAction archiveAction = (CommandAction) getGeneratingAction(archiveFile);
    CommandAction compileAction =
        (CommandAction)
            getGeneratingAction(getFirstArtifactEndingWith(archiveAction.getInputs(), objFileName));
    return compileAction;
  }

  protected void addSimpleBinaryTarget(String j2objcLibraryTargetDep) throws Exception {
    scratch.file("app/app.m");
    scratch.file("app/Info.plist");
    scratch.file("app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "    name = 'lib',",
        "    deps = ['" + j2objcLibraryTargetDep + "'])",
        "",
        "objc_binary(",
        "    name = 'app',",
        "    srcs = ['app.m'],",
        "    deps = [':lib'],",
        ")");

  }

  protected void addSimpleJ2ObjcLibraryWithEntryClasses() throws Exception {
    scratch.file("java/com/google/app/test/test.java");
    scratch.file("java/com/google/app/test/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    entry_classes = ['com.google.app.test.test'],",
        "    deps = ['test'])");
  }

  protected void addSimpleJ2ObjcLibraryWithJavaPlugin() throws Exception {
    scratch.file("java/com/google/app/test/test.java");
    scratch.file("java/com/google/app/test/plugin.java");
    scratch.file(
        "java/com/google/app/test/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    plugins = [':plugin'])",
        "",
        "java_plugin(",
        "    name = 'plugin',",
        "    processor_class = 'com.google.process.stuff',",
        "    srcs = ['plugin.java'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = [':test'])");
  }

  protected Artifact j2objcArchive(String j2objcLibraryTarget, String javaTargetName)
      throws Exception {
    ConfiguredTarget target = getConfiguredTarget(j2objcLibraryTarget);
    ObjcProvider provider = target.getProvider(ObjcProvider.class);
    String archiveName = String.format("lib%s_j2objc.a", javaTargetName);
    return getFirstArtifactEndingWith(provider.get(ObjcProvider.LIBRARY), archiveName);
  }

  @Test
  public void testJ2ObjcInformationExportedFromObjcLibrary() throws Exception {
    scratch.file("app/lib.m");
    scratch.file(
        "app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = ['//java/com/google/dummy/test:transpile'])");

    ConfiguredTarget objcTarget = getConfiguredTarget("//app:lib");

    ObjcProvider provider = objcTarget.getProvider(ObjcProvider.class);
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY)))
        .containsExactly(
            "third_party/java/j2objc/libjre_core_lib.a",
            "java/com/google/dummy/test/libtest_j2objc.a",
            "app/liblib.a");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.HEADER)))
        .containsExactly(
            "third_party/java/j2objc/jre_core.h",
            "java/com/google/dummy/test/_j2objc/test/java/com/google/dummy/test/test.h");

    String execPath =
        objcTarget.getConfiguration().getBinDirectory(RepositoryName.MAIN).getExecPath() + "/";
    assertThat(PathFragment.safePathStrings(provider.get(ObjcProvider.INCLUDE)))
        .containsExactly(execPath + "java/com/google/dummy/test/_j2objc/test");
  }

  @Test
  public void testJ2ObjcInfoExportedInObjcLibraryFromRuntimeDeps() throws Exception {
    scratch.file("app/lib.m");
    scratch.file(
        "app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "",
        "java_library(name = 'dummyOne',",
        "    srcs = ['dummyOne.java'])",
        "java_library(name = 'dummyTwo',",
        "    srcs = ['dummyTwo.java'],",
        "    runtime_deps = [':dummyOne'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = [':dummyTwo'])",
        "",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = ['//app:transpile'])");

    ConfiguredTarget objcTarget = getConfiguredTarget("//app:lib");

    ObjcProvider provider = objcTarget.getProvider(ObjcProvider.class);
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY)))
        .containsExactly(
            "third_party/java/j2objc/libjre_core_lib.a",
            "app/libdummyOne_j2objc.a",
            "app/libdummyTwo_j2objc.a",
            "app/liblib.a");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.HEADER)))
        .containsExactly(
            "third_party/java/j2objc/jre_core.h",
            "app/_j2objc/dummyOne/app/dummyOne.h",
            "app/_j2objc/dummyTwo/app/dummyTwo.h");

    String execPath =
        objcTarget.getConfiguration().getBinDirectory(RepositoryName.MAIN).getExecPath() + "/";
    assertThat(PathFragment.safePathStrings(provider.get(ObjcProvider.INCLUDE)))
        .containsExactly(execPath + "app/_j2objc/dummyOne", execPath + "app/_j2objc/dummyTwo");
  }

  @Test
  public void testJ2ObjcAppearsInLinkArgs() throws Exception {
    scratch.file(
        "java/c/y/BUILD", "java_library(", "    name = 'ylib',", "    srcs = ['lib.java'],", ")");
    scratch.file(
        "x/BUILD",
        "j2objc_library(",
        "    name = 'j2',",
        "    deps = [ '//java/c/y:ylib' ],",
        "    jre_deps = [ '//third_party/java/j2objc:jre_io_lib' ])",
        "ios_application(",
        "    name = 'app',",
        "    binary = ':bin',",
        "    infoplist = 'info.plist',",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = [':j2'],",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    deps = [':j2'],",
        "    xctest = 1,",
        "    xctest_app = ':app',",
        ")");

    CommandAction linkAction = linkAction("//x:test");
    List<String> linkArgs = normalizeBashArgs(linkAction.getArguments());
    ConfiguredTarget target = getConfiguredTarget("//x:test");
    String binDir =
        target.getConfiguration().getBinDirectory(RepositoryName.MAIN).getExecPathString();
    Artifact fileList = getFirstArtifactEndingWith(linkAction.getInputs(), "test-linker.objlist");
    ParameterFileWriteAction filelistWriteAction =
        (ParameterFileWriteAction) getGeneratingAction(fileList);
    assertThat(linkArgs).contains(fileList.getExecPathString());
    assertThat(filelistWriteAction.getContents())
        .containsAllOf(
            binDir + "/java/c/y/libylib_j2objc.a",
            // All jre libraries mus appear after java libraries in the link order.
            binDir + "/third_party/java/j2objc/libjre_io_lib.a",
            binDir + "/third_party/java/j2objc/libjre_core_lib.a")
        .inOrder();
  }

  @Test
  public void testArchiveLinkActionWithTreeArtifactFromGenJar() throws Exception {
    useConfiguration("--ios_cpu=i386", "--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    Artifact archive = j2objcArchive("//java/com/google/app/test:transpile", "test");
    CommandAction archiveAction = (CommandAction) getGeneratingAction(archive);
    Artifact archiveObjList =
        getFirstArtifactEndingWith(archiveAction.getInputs(), "-archive.objlist");
    Artifact objectFilesFromGenJar =
        getFirstArtifactEndingWith(archiveAction.getInputs(), "source_files");
    Artifact normalObjectFile = getFirstArtifactEndingWith(archiveAction.getInputs(), "test.o");

    ParameterFileWriteAction paramFileAction =
        (ParameterFileWriteAction) getGeneratingAction(archiveObjList);

    // Test that the archive obj list param file contains the individual object files inside
    // the object file tree artifact.
    assertThat(paramFileAction.getContents(DUMMY_ARTIFACT_EXPANDER))
        .containsExactly(
            objectFilesFromGenJar.getExecPathString() + "/children1",
            objectFilesFromGenJar.getExecPathString() + "/children2",
            normalObjectFile.getExecPathString());
  }

  @Test
  public void testJ2ObjCCustomModuleMap() throws Exception {
    useConfiguration("--experimental_objc_enable_module_maps");
    scratch.file("java/com/google/transpile/dummy.java");
    scratch.file("java/com/google/transpile/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(name = 'dummy',",
        "    srcs = ['dummy.java'])");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");

    ObjcProvider provider = target.getProvider(ObjcProvider.class);
    Artifact moduleMap =
        getFirstArtifactEndingWith(
            provider.get(ObjcProvider.MODULE_MAP), "dummy.modulemaps/module.modulemap");

    Artifact umbrellaHeader =
        getFirstArtifactEndingWith(
            provider.get(ObjcProvider.UMBRELLA_HEADER), "dummy.modulemaps/umbrella.h");

    CppModuleMapAction moduleMapAction = (CppModuleMapAction) getGeneratingAction(moduleMap);
    UmbrellaHeaderAction umbrellaHeaderAction =
        (UmbrellaHeaderAction) getGeneratingAction(umbrellaHeader);

    ActionExecutionContext dummyActionExecutionContext =
        new ActionExecutionContext(
            null, null, null, null, ImmutableMap.<String, String>of(), DUMMY_ARTIFACT_EXPANDER);
    ByteArrayOutputStream moduleMapStream = new ByteArrayOutputStream();
    ByteArrayOutputStream umbrellaHeaderStream = new ByteArrayOutputStream();
    moduleMapAction.newDeterministicWriter(dummyActionExecutionContext)
        .writeOutputFile(moduleMapStream);
    umbrellaHeaderAction.newDeterministicWriter(dummyActionExecutionContext)
        .writeOutputFile(umbrellaHeaderStream);
    String moduleMapContent = moduleMapStream.toString();
    String umbrellaHeaderContent = umbrellaHeaderStream.toString();

    assertThat(moduleMapContent).contains("umbrella header \"umbrella.h\"");
    assertThat(umbrellaHeaderContent).contains("/dummy.h");
    assertThat(umbrellaHeaderContent).contains("#include");
  }

  @Test
  public void testModuleMapFromGenJarTreeArtifact() throws Exception {
    useConfiguration("--ios_cpu=i386", "--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/app/test:transpile");
    ObjcProvider provider = j2objcLibraryTarget.getProvider(ObjcProvider.class);
    Artifact moduleMap =
        getFirstArtifactEndingWith(
            provider.get(ObjcProvider.MODULE_MAP), "test.modulemaps/module.modulemap");
    Artifact umbrellaHeader =
        getFirstArtifactEndingWith(
            provider.get(ObjcProvider.UMBRELLA_HEADER), "test.modulemaps/umbrella.h");

    CppModuleMapAction moduleMapAction = (CppModuleMapAction) getGeneratingAction(moduleMap);
    UmbrellaHeaderAction umbrellaHeaderAction =
        (UmbrellaHeaderAction) getGeneratingAction(umbrellaHeader);
    Artifact headers =
        getFirstArtifactEndingWith(provider.get(ObjcProvider.HEADER), "header_files");

    // Test that the module map action contains the header tree artifact as both the public header
    // and part of the action inputs.
    assertThat(moduleMapAction.getPublicHeaders()).contains(headers);
    assertThat(moduleMapAction.getInputs()).contains(headers);

    ActionExecutionContext dummyActionExecutionContext =
        new ActionExecutionContext(
            null, null, null, null, ImmutableMap.<String, String>of(), DUMMY_ARTIFACT_EXPANDER);

    ByteArrayOutputStream moduleMapStream = new ByteArrayOutputStream();
    ByteArrayOutputStream umbrellaHeaderStream = new ByteArrayOutputStream();
    moduleMapAction
        .newDeterministicWriter(dummyActionExecutionContext)
        .writeOutputFile(moduleMapStream);
    umbrellaHeaderAction
        .newDeterministicWriter(dummyActionExecutionContext)
        .writeOutputFile(umbrellaHeaderStream);
    String moduleMapContent = moduleMapStream.toString();
    String umbrellaHeaderContent = umbrellaHeaderStream.toString();

    // Test that the module map content contains the individual headers inside the header tree
    // artifact.
    assertThat(moduleMapContent).contains("umbrella header \"umbrella.h\"");
    assertThat(umbrellaHeaderContent).contains(headers.getExecPathString() + "/children1");
    assertThat(umbrellaHeaderContent).contains(headers.getExecPathString() + "/children2");
  }

  @Test
  public void testJ2ObjCFullyLinkAction() throws Exception {
    AbstractAction linkAction = (AbstractAction) getGeneratingActionForLabel(
        "//java/com/google/dummy/test:transpile_fully_linked.a");
    String fullyLinkBinaryPath =
        Iterables.getOnlyElement(linkAction.getOutputs()).getExecPathString();
    assertThat(fullyLinkBinaryPath).contains("transpile_fully_linked.a");
  }
}
