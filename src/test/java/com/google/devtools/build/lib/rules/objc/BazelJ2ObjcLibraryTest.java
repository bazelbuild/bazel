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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionTemplate;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.UmbrellaHeaderAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.io.ByteArrayOutputStream;
import java.util.Optional;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit test for Java source file translation into ObjC in <code>java_library</code> and translated
 * file compilation and linking in {@link ObjcBinary} and {@link J2ObjcLibrary}.
 */
@RunWith(JUnit4.class)
public class BazelJ2ObjcLibraryTest extends J2ObjcLibraryTest {

  private static final Provider.Key starlarkJ2objcMappingFileProviderKey =
      new StarlarkProvider.Key(
          Label.parseCanonicalUnchecked("@_builtins//:common/objc/providers.bzl"),
          "J2ObjcMappingFileInfo");

  private StructImpl getJ2ObjcMappingFileInfoFromTarget(ConfiguredTarget configuredTarget)
      throws Exception {
    return (StructImpl) configuredTarget.get(starlarkJ2objcMappingFileProviderKey);
  }

  private ImmutableList<Artifact> getArtifacts(StructImpl j2ObjcMappingFileInfo, String attribute)
      throws Exception {
    Depset filesDepset = (Depset) j2ObjcMappingFileInfo.getValue(attribute);
    return filesDepset.toList(Artifact.class);
  }

  @Test
  public void testJ2ObjCInformationExportedFromJ2ObjcLibrary() throws Exception {
    ConfiguredTarget j2objcLibraryTarget = getConfiguredTarget(
        "//java/com/google/dummy/test:transpile");

    CcLinkingContext ccLinkingContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly("libjre_core_lib.a", "libtest_j2objc.lo");

    CcCompilationContext ccCompilationContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "test.h");

    String execPath = getRuleContext(j2objcLibraryTarget).getBinFragment() + "/";
    assertThat(
            Iterables.transform(
                ccCompilationContext.getIncludeDirs(), PathFragment::getSafePathString))
        .containsExactly(execPath + "java/com/google/dummy/test/_j2objc/test");
  }

  @Test
  public void testJ2ObjCAspectDisablesParseHeaders() throws Exception {
    MockObjcSupport.setupCcToolchainConfig(
        mockToolsConfig, MockObjcSupport.darwinX86_64().withFeatures(CppRuleClasses.PARSE_HEADERS));
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");

    ConfiguredTarget j2objcAspectTarget =
        getJ2ObjCAspectConfiguredTarget("//java/com/google/dummy/test:test");
    CcCompilationContext ccCompilationContext =
        j2objcAspectTarget.get(CcInfo.PROVIDER).getCcCompilationContext();

    assertThat(baseArtifactNames(ccCompilationContext.getHeaderTokens().toList()))
        .doesNotContain("test.h.processed");
  }

  @Test
  public void testJ2ObjCInformationExportedWithGeneratedJavaSources() throws Exception {
    scratch.file("java/com/google/test/in.txt");
    scratch.file(
        "java/com/google/test/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "genrule(",
        "    name = 'dummy_gen',",
        "    srcs = ['in.txt'],",
        "    outs = ['test.java'],",
        "    cmd = 'dummy'",
        ")",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = [':test.java']",
        ")",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = ['test'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//java/com/google/test:transpile");

    CcLinkingContext ccLinkingContext = target.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly("libjre_core_lib.a", "libtest_j2objc.lo");

    CcCompilationContext ccCompilationContext =
        target.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "test.h");

    String execPath = getRuleContext(target).getBinFragment() + "/";
    String genfilesFragment = getRuleContext(target).getGenfilesFragment().toString();
    assertThat(
            Iterables.transform(
                ccCompilationContext.getIncludeDirs(), PathFragment::getSafePathString))
        .containsExactly(
            execPath + "java/com/google/test/_j2objc/test/" + genfilesFragment,
            execPath + "java/com/google/test/_j2objc/test");
  }

  @Test
  public void testJ2ObjcProtoRuntimeLibraryAndHeadersExported() throws Exception {
    scratch.file("java/com/google/dummy/test/proto/test.java");
    scratch.file("java/com/google/dummy/test/proto/test.proto");
    scratch.file(
        "java/com/google/dummy/test/proto/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        ")",
        "java_proto_library(",
        "    name = 'test_java_proto',",
        "    deps = [':test_proto'],",
        ")",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_java_proto']",
        ")",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = ['test'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");

    ConfiguredTarget j2objcLibraryTarget = getConfiguredTarget(
        "//java/com/google/dummy/test/proto:transpile");

    CcLinkingContext ccLinkingContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly(
            "libjre_core_lib.a",
            "libproto_runtime.a",
            "libtest_j2objc.lo",
            "libtest_proto_j2objc.lo");

    CcCompilationContext ccCompilationContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "runtime.h", "test.j2objc.pb.h", "test.h");
  }

  @Test
  public void testJ2ObjcHeaderMapExportedInJavaLibrary() throws Exception {
    scratch.file(
        "java/com/google/transpile/BUILD",
        "java_library(",
        "    name = 'dummy',",
        "    srcs = ['dummy.java']",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> headerMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "header_mapping_files");

    assertThat(headerMappingFilesList.get(0).getRootRelativePath().toString())
        .isEqualTo("java/com/google/transpile/dummy.mapping.j2objc");
  }

  @Test
  public void testDepsJ2ObjcHeaderMapExportedInJavaLibraryWithNoSourceFile() throws Exception {
    scratch.file(
        "java/com/google/transpile/BUILD",
        "java_library(",
        "    name = 'dummy',",
        "    exports = ['//java/com/google/dep:dep'],",
        ")");
    scratch.file(
        "java/com/google/dep/BUILD",
        "java_library(",
        "    name = 'dep',",
        "    srcs = ['dummy.java'],",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> headerMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "header_mapping_files");

    assertThat(headerMappingFilesList.get(0).getRootRelativePath().toString())
        .isEqualTo("java/com/google/dep/dep.mapping.j2objc");
  }

  @Test
  public void testJ2ObjcProtoClassMappingFilesExportedInJavaLibrary() throws Exception {
    scratch.file("java/com/google/dummy/test/proto/test.java");
    scratch.file("java/com/google/dummy/test/proto/test.proto");
    scratch.file(
        "java/com/google/dummy/test/proto/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        ")",
        "java_proto_library(",
        "    name = 'test_java_proto',",
        "    deps = [':test_proto'],",
        ")",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_java_proto']",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget(
        "//java/com/google/dummy/test/proto:test");
    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> classMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "class_mapping_files");

    assertThat(classMappingFilesList.get(0).getExecPathString())
        .containsMatch(
            "/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin/java/com/google/dummy/test/proto/test.clsmap.properties");
  }

  @Test
  public void testJavaProtoLibraryWithProtoLibrary() throws Exception {
    scratch.file(
        "x/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        ")",
        "java_proto_library(",
        "    name = 'test_java_proto',",
        "    deps = [':test_proto'],",
        ")",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_java_proto']",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//x:test");
    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> classMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "class_mapping_files");
    assertThat(classMappingFilesList.get(0).getExecPathString())
        .containsMatch(
            "/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin/x/test.clsmap.properties");

    ObjcProvider objcProvider = target.get(ObjcProvider.STARLARK_CONSTRUCTOR);
    CcCompilationContext ccCompilationContext =
        target.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(ccCompilationContext.getDeclaredIncludeSrcs().toList().toString())
        .containsMatch("/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin]x/test.j2objc.pb.h");
    assertThat(objcProvider.get(ObjcProvider.SOURCE).toList().toString())
        .containsMatch("/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin]x/test.j2objc.pb.m,");
  }

  @Test
  public void testJavaProtoLibraryWithProtoLibrary_external() throws Exception {
    scratch.file("/bla/WORKSPACE");
    // Create the rule '@bla//foo:test_proto'.
    scratch.file(
        "/bla/foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        ")",
        "java_proto_library(",
        "    name = 'test_java_proto',",
        "    deps = [':test_proto'])",
        "");

    String existingWorkspace =
        new String(FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    scratch.overwriteFile(
        "WORKSPACE", "local_repository(name = 'bla', path = '/bla/')", existingWorkspace);
    invalidatePackages(); // A dash of magic to re-evaluate the WORKSPACE file.

    scratch.file(
        "x/BUILD",
        "",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = ['@bla//foo:test_java_proto'])");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//x:test");

    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> classMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "class_mapping_files");

    assertThat(classMappingFilesList.get(0).getExecPathString())
        .containsMatch(
            "/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin/external/bla/foo/test.clsmap.properties");

    ObjcProvider objcProvider = target.get(ObjcProvider.STARLARK_CONSTRUCTOR);
    CcCompilationContext ccCompilationContext =
        target.get(CcInfo.PROVIDER).getCcCompilationContext();

    assertThat(ccCompilationContext.getDeclaredIncludeSrcs().toList().toString())
        .containsMatch(
            "/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin]external/bla/foo/test.j2objc.pb.h");
    assertThat(objcProvider.get(ObjcProvider.SOURCE).toList().toString())
        .containsMatch(
            "/applebin_macos-darwin_x86_64-fastbuild-ST-[^/]*/bin]external/bla/foo/test.j2objc.pb.m");
    assertThat(ccCompilationContext.getIncludeDirs())
        .contains(
            getConfiguration(target)
                .getGenfilesFragment(RepositoryName.create("bla"))
                .getRelative("external/bla"));
  }

  @Test
  public void testJ2ObjcInfoExportedInJavaImport() throws Exception {
    scratch.file(
        "java/com/google/transpile/BUILD",
        "java_import(",
        "    name = 'dummy',",
        "    jars = ['dummy.jar'],",
        "    srcjar = 'dummy.srcjar',",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> headerMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "header_mapping_files");
    ImmutableList<Artifact> classMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "class_mapping_files");

    assertThat(headerMappingFilesList.get(0).getRootRelativePath().toString())
        .isEqualTo("java/com/google/transpile/dummy.mapping.j2objc");
    assertThat(classMappingFilesList).isEmpty();
  }

  protected Artifact getBinArtifact(String outputName, RuleConfiguredTarget target) {
    for (ActionAnalysisMetadata action : target.getActions()) {
      Optional<Artifact> artifact =
          action.getOutputs().stream().filter(artifactNamed(outputName)).findFirst();
      if (artifact.isPresent()) {
        return artifact.get();
      }
    }
    return null;
  }

  protected void checkObjcArchiveAndLinkActions(
      String archiveFileName, String objFileName, Iterable<String> compilationInputExecPaths)
      throws Exception {
    CommandAction linkAction =
        (CommandAction)
            getGeneratingAction(
                getBinArtifact(
                    "app/app_bin", (RuleConfiguredTarget) getConfiguredTarget("//app:app")));

    checkObjcCompileActions(
        getFirstArtifactEndingWith(linkAction.getInputs(), archiveFileName),
        objFileName, compilationInputExecPaths);
  }

  @Test
  public void testMissingEntryClassesError() throws Exception {
    useConfiguration("--j2objc_dead_code_removal");
    checkError(
        "java/com/google/dummy",
        "transpile",
        "Entry classes must be specified when flag --compilation_mode=opt is on in order to perform"
            + " J2ObjC dead code stripping.",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = ['//java/com/google/dummy/test:test'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");
  }

  @Test
  public void testNoJ2ObjcDeadCodeRemovalActionWithoutOptFlag() throws Exception {
    useConfiguration("--noj2objc_dead_code_removal");
    addSimpleJ2ObjcLibraryWithEntryClasses();
    addAppleBinaryStarlarkRule(scratch);
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

    CcLinkingContext ccLinkingContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly("libjre_core_lib.a", "libtest_j2objc.lo");

    CcCompilationContext ccCompilationContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "test.h");
  }

  @Test
  public void testTagInJreDeps() throws Exception {
    scratch.file(
        "app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "    name = 'no_tag_dep',",
        ")",
        "j2objc_library(",
        "    name = 'test',",
        "    jre_deps = ['no_tag_dep'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//app:test");

    assertContainsEvent(
        Pattern.compile(
            ".* objc_library rule \\'//app:no_tag_dep\\' is misplaced here \\(Only J2ObjC JRE"
                + " libraries are allowed\\)"));
  }

  @Test
  public void testTranspilationActionTreeArtifactOutputsFromSourceJar() throws Exception {
    useConfiguration("--ios_minimum_os=1.0");
    scratch.file("java/com/google/transpile/dummy.java");
    scratch.file("java/com/google/transpile/dummyjar.srcjar");
    scratch.file(
        "java/com/google/transpile/BUILD",
        "java_library(",
        "    name = 'dummy',",
        "    srcs = ['dummy.java', 'dummyjar.srcjar'],",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");
    ObjcProvider provider = target.get(ObjcProvider.STARLARK_CONSTRUCTOR);
    CcCompilationContext ccCompilationContext =
        target.get(CcInfo.PROVIDER).getCcCompilationContext();
    Artifact srcJarSources = getFirstArtifactEndingWith(
        provider.get(ObjcProvider.SOURCE), "source_files");
    Artifact srcJarHeaders =
        getFirstArtifactEndingWith(ccCompilationContext.getDeclaredIncludeSrcs(), "header_files");
    assertThat(srcJarSources.getRootRelativePathString())
        .isEqualTo("java/com/google/transpile/_j2objc/src_jar_files/dummy/source_files");
    assertThat(srcJarHeaders.getRootRelativePathString())
        .isEqualTo("java/com/google/transpile/_j2objc/src_jar_files/dummy/header_files");
    assertThat(srcJarSources.isTreeArtifact()).isTrue();
    assertThat(srcJarHeaders.isTreeArtifact()).isTrue();
  }

  @Test
  public void testGeneratedTreeArtifactFromGenJar() throws Exception {
    useConfiguration("--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/app/test:transpile");
    ObjcProvider provider = j2objcLibraryTarget.get(ObjcProvider.STARLARK_CONSTRUCTOR);
    CcCompilationContext ccCompilationContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    Artifact headers =
        getFirstArtifactEndingWith(ccCompilationContext.getDeclaredIncludeSrcs(), "header_files");
    Artifact sources =
        getFirstArtifactEndingWith(provider.get(ObjcProvider.SOURCE), "source_files");
    assertThat(headers.isTreeArtifact()).isTrue();
    assertThat(sources.isTreeArtifact()).isTrue();

    SpawnAction j2objcAction = (SpawnAction) getGeneratingAction(headers);
    assertThat(j2objcAction.getOutputs()).containsAtLeast(headers, sources);

    assertContainsSublist(
        ImmutableList.copyOf(paramFileArgsForAction(j2objcAction)),
        ImmutableList.of(
            "--output_gen_source_dir",
            sources.getExecPathString(),
            "--output_gen_header_dir",
            headers.getExecPathString()));
  }

  @Test
  public void testJ2ObjcHeaderMappingAction() throws Exception {
    scratch.file(
        "java/com/google/transpile/BUILD",
        "java_library(",
        "    name = 'lib1',",
        "    srcs = ['libOne.java', 'jar.srcjar'],",
        "    deps = [':lib2']",
        ")",
        "",
        "java_library(",
        "    name = 'lib2',",
        "    srcs = ['libTwo.java'],",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget(
        "//java/com/google/transpile:lib1");
    StructImpl j2ObjcMappingFileInfo = getJ2ObjcMappingFileInfoFromTarget(target);
    ImmutableList<Artifact> headerMappingFilesList =
        getArtifacts(j2ObjcMappingFileInfo, "header_mapping_files");
    assertThat(baseArtifactNames(headerMappingFilesList))
        .containsExactly("lib1.mapping.j2objc", "lib2.mapping.j2objc");

    Artifact mappingFile =
        getFirstArtifactEndingWith(headerMappingFilesList, "lib1.mapping.j2objc");
    SpawnAction headerMappingAction = (SpawnAction) getGeneratingAction(mappingFile);
    String execPath = getRuleContext(target).getBinFragment() + "/";
    assertThat(baseArtifactNames(headerMappingAction.getInputs()))
        .containsAtLeast("libOne.java", "jar.srcjar");
    assertThat(headerMappingAction.getArguments().get(0))
        .contains("tools/j2objc/j2objc_header_map_binary");
    assertThat(headerMappingAction.getArguments().get(1)).isEqualTo("--source_files");
    assertThat(headerMappingAction.getArguments().get(2))
        .isEqualTo("java/com/google/transpile/libOne.java");
    assertThat(headerMappingAction.getArguments().get(3)).isEqualTo("--source_jars");
    assertThat(headerMappingAction.getArguments().get(4))
        .isEqualTo("java/com/google/transpile/jar.srcjar");
    assertThat(headerMappingAction.getArguments().get(5)).isEqualTo("--output_mapping_file");
    assertThat(headerMappingAction.getArguments().get(6))
        .isEqualTo(execPath + "java/com/google/transpile/lib1.mapping.j2objc");
  }

  protected void checkObjcCompileActions(
      Artifact archiveFile, String objFileName, Iterable<String> compilationInputExecPaths)
      throws Exception {
    CommandAction compileAction = getObjcCompileAction(archiveFile, objFileName);
    assertThat(baseArtifactNames(compileAction.getPossibleInputsForTesting()))
        .containsAtLeastElementsIn(compilationInputExecPaths);
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
    scratch.file(
        "app/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "    name = 'lib',",
        "    deps = ['" + j2objcLibraryTargetDep + "'])",
        "",
        "apple_binary_starlark(",
        "    name = 'app',",
        "    platform_type = 'ios',",
        "    deps = [':main_lib'],",
        ")",
        "objc_library(",
        "    name = 'main_lib',",
        "    srcs = ['app.m'],",
        "    deps = [':lib'],",
        ")");
  }

  protected void addSimpleJ2ObjcLibraryWithEntryClasses() throws Exception {
    scratch.file("java/com/google/app/test/test.java");
    scratch.file(
        "java/com/google/app/test/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        ")",
        "j2objc_library(",
        "    name = 'transpile',",
        "    entry_classes = ['com.google.app.test.test'],",
        "    deps = ['test'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");
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
        "    plugins = [':plugin'],",
        ")",
        "java_plugin(",
        "    name = 'plugin',",
        "    processor_class = 'com.google.process.stuff',",
        "    srcs = ['plugin.java'],",
        ")",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = [':test'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");
  }

  protected Artifact j2objcArchive(String j2objcLibraryTarget, String javaTargetName)
      throws Exception {
    ConfiguredTarget target = getConfiguredTarget(j2objcLibraryTarget);
    CcLinkingContext ccLinkingContext = target.get(CcInfo.PROVIDER).getCcLinkingContext();
    String archiveName = String.format("lib%s_j2objc.lo", javaTargetName);
    return getFirstArtifactEndingWith(
        ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries(), archiveName);
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
        "    deps = ['//java/com/google/dummy/test:transpile'],",
        ")");

    ConfiguredTarget objcTarget = getConfiguredTarget("//app:lib");

    CcLinkingContext ccLinkingContext = objcTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly("libjre_core_lib.a", "libtest_j2objc.lo", "liblib.a");

    CcCompilationContext ccCompilationContext =
        objcTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "test.h");

    String execPath = getRuleContext(objcTarget).getBinFragment() + "/";
    assertThat(
            Iterables.transform(
                ccCompilationContext.getIncludeDirs(), PathFragment::getSafePathString))
        .containsExactly(execPath + "java/com/google/dummy/test/_j2objc/test");
  }

  @Test
  public void testJ2ObjcInfoExportedInObjcLibraryFromRuntimeDeps() throws Exception {
    scratch.file("app/lib.m");
    scratch.file(
        "app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "",
        "java_library(",
        "    name = 'dummyOne',",
        "    srcs = ['dummyOne.java'],",
        ")",
        "java_library(",
        "    name = 'dummyTwo',",
        "    srcs = ['dummyTwo.java'],",
        "    runtime_deps = [':dummyOne'],",
        ")",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = [':dummyTwo'],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = ['//app:transpile'],",
        ")");

    ConfiguredTarget objcTarget = getConfiguredTarget("//app:lib");

    CcLinkingContext ccLinkingContext = objcTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly(
            "libjre_core_lib.a", "libdummyOne_j2objc.lo", "libdummyTwo_j2objc.lo", "liblib.a");

    CcCompilationContext ccCompilationContext =
        objcTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "dummyOne.h", "dummyTwo.h");

    String execPath = getRuleContext(objcTarget).getBinFragment() + "/";
    assertThat(
            Iterables.transform(
                ccCompilationContext.getIncludeDirs(), PathFragment::getSafePathString))
        .containsExactly(execPath + "app/_j2objc/dummyOne", execPath + "app/_j2objc/dummyTwo");
  }

  @Nullable
  private ParameterFileWriteAction paramFileWriteActionForLinkAction(Action action) {
    for (Artifact input : action.getInputs().toList()) {
      if (!(input instanceof SpecialArtifact)) {
        if (input.getFilename().endsWith("linker.objlist")) {
          Action generatingAction = getGeneratingAction(input);
          if (generatingAction instanceof ParameterFileWriteAction) {
            return (ParameterFileWriteAction) generatingAction;
          }
        }
      }
    }
    return null;
  }

  @Test
  public void testJ2ObjcAppearsInLinkArgs() throws Exception {
    scratch.file(
        "java/c/y/BUILD", "java_library(", "    name = 'ylib',", "    srcs = ['lib.java'],", ")");
    addAppleBinaryStarlarkRule(scratch);
    scratch.file(
        "x/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "j2objc_library(",
        "    name = 'j2',",
        "    deps = [ '//java/c/y:ylib' ],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        "    jre_deps = [ '"
            + TestConstants.TOOLS_REPOSITORY
            + "//third_party/java/j2objc:jre_io_lib' ],",
        ")",
        "apple_binary_starlark(",
        "    name = 'test',",
        "    platform_type = 'ios',",
        "    deps = [':main_lib'],",
        ")",
        "objc_library(",
        "    name = 'main_lib',",
        "    srcs = ['test.m'],",
        "    deps = [':j2'],",
        ")");

    CommandAction linkAction = linkAction("//x:test");
    ConfiguredTarget target = getConfiguredTarget("//x:test");
    String binDir = removeConfigFragment(getRuleContext(target).getBinFragment().toString());
    ParameterFileWriteAction paramAction = paramFileWriteActionForLinkAction(linkAction);
    Iterable<String> paramFileArgs = paramAction.getCommandLine().arguments();
    assertThat(removeConfigFragment(ImmutableList.copyOf(linkAction.getArguments())))
        .contains("-force_load " + binDir + "/java/c/y/libylib_j2objc.lo");
    assertThat(removeConfigFragment(ImmutableList.copyOf(paramFileArgs)))
        .containsAtLeast(
            // All jre libraries mus appear after java libraries in the link order.
            binDir
                + "/"
                + TestConstants.TOOLS_REPOSITORY_PATH_PREFIX
                + "third_party/java/j2objc/libjre_io_lib.a",
            binDir
                + "/"
                + TestConstants.TOOLS_REPOSITORY_PATH_PREFIX
                + "third_party/java/j2objc/libjre_core_lib.a")
        .inOrder();
  }

  @Test
  public void testArchiveLinkActionWithTreeArtifactFromGenJar() throws Exception {
    useConfiguration("--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    Artifact archive = j2objcArchive("//java/com/google/app/test:transpile", "test");
    CppLinkAction archiveAction = (CppLinkAction) getGeneratingAction(archive);
    Artifact objectFilesFromGenJar =
        getFirstArtifactEndingWith(archiveAction.getInputs(), "source_files");
    Artifact normalObjectFile = getFirstArtifactEndingWith(archiveAction.getInputs(), "test.o");

    // Test that the archive commandline contains the individual object files inside
    // the object file tree artifact.
    assertThat(archiveAction.getLinkCommandLineForTesting().arguments(DUMMY_ARTIFACT_EXPANDER))
        .containsAtLeast(
            objectFilesFromGenJar.getExecPathString() + "/children1",
            objectFilesFromGenJar.getExecPathString() + "/children2",
            normalObjectFile.getExecPathString());
  }

  @Test
  public void testJ2ObjCCustomModuleMap() throws Exception {
    scratch.file("java/com/google/transpile/dummy.java");
    scratch.file(
        "java/com/google/transpile/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(",
        "    name = 'dummy',",
        "    srcs = ['dummy.java'],",
        ")");

    ConfiguredTarget target = getJ2ObjCAspectConfiguredTarget("//java/com/google/transpile:dummy");

    ObjcProvider provider = target.get(ObjcProvider.STARLARK_CONSTRUCTOR);
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
            /* executor= */ null,
            /* inputMetadataProvider= */ null,
            ActionInputPrefetcher.NONE,
            actionKeyContext,
            /* outputMetadataStore= */ null,
            /* rewindingEnabled= */ false,
            LostInputsCheck.NONE,
            /* fileOutErr= */ null,
            /* eventHandler= */ null,
            /* clientEnv= */ ImmutableMap.of(),
            /* topLevelFilesets= */ ImmutableMap.of(),
            DUMMY_ARTIFACT_EXPANDER,
            /* actionFileSystem= */ null,
            /* skyframeDepsResult= */ null,
            DiscoveredModulesPruner.DEFAULT,
            SyscallCache.NO_CACHE,
            ThreadStateReceiver.NULL_INSTANCE);
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
    useConfiguration("--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/app/test:transpile");
    ObjcProvider provider = j2objcLibraryTarget.get(ObjcProvider.STARLARK_CONSTRUCTOR);
    CcCompilationContext ccCompilationContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
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
        getFirstArtifactEndingWith(ccCompilationContext.getDeclaredIncludeSrcs(), "header_files");

    // Test that the module map action contains the header tree artifact as both the public header
    // and part of the action inputs.
    assertThat(moduleMapAction.getPublicHeaders()).contains(headers);
    assertThat(moduleMapAction.getInputs().toList()).contains(headers);

    ActionExecutionContext dummyActionExecutionContext =
        new ActionExecutionContext(
            /* executor= */ null,
            /* inputMetadataProvider= */ null,
            ActionInputPrefetcher.NONE,
            actionKeyContext,
            /* outputMetadataStore= */ null,
            /* rewindingEnabled= */ false,
            LostInputsCheck.NONE,
            /* fileOutErr= */ null,
            /* eventHandler= */ null,
            /* clientEnv= */ ImmutableMap.of(),
            /* topLevelFilesets= */ ImmutableMap.of(),
            DUMMY_ARTIFACT_EXPANDER,
            /* actionFileSystem= */ null,
            /* skyframeDepsResult= */ null,
            DiscoveredModulesPruner.DEFAULT,
            SyscallCache.NO_CACHE,
            ThreadStateReceiver.NULL_INSTANCE);

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
  public void testObjcCompileAction() throws Exception {
    Artifact archive = j2objcArchive("//java/com/google/dummy/test:transpile", "test");
    CommandAction compileAction = getObjcCompileAction(archive, "test.o");
    assertThat(baseArtifactNames(compileAction.getPossibleInputsForTesting()))
        .containsAtLeast("jre_core.h", "test.h", "test.m");
    assertThat(compileAction.getArguments())
        .containsAtLeast("-fno-objc-arc", "-fno-strict-overflow", "-fobjc-weak");
  }

  @Test
  public void testObjcCompileArcAction() throws Exception {
    useConfiguration("--j2objc_translation_flags=-use-arc");
    Artifact archive = j2objcArchive("//java/com/google/dummy/test:transpile", "test");
    CommandAction compileAction = getObjcCompileAction(archive, "test.o");
    assertThat(baseArtifactNames(compileAction.getPossibleInputsForTesting()))
        .containsAtLeast("jre_core.h", "test.h", "test.m");
    assertThat(compileAction.getArguments())
        .containsAtLeast("-fobjc-arc", "-fobjc-arc-exceptions", "-fno-strict-overflow");
  }

  @Test
  public void testJ2ObjcSourcesCompilationAndLinking() throws Exception {
    addAppleBinaryStarlarkRule(scratch);
    addSimpleBinaryTarget("//java/com/google/dummy/test:transpile");

    checkObjcArchiveAndLinkActions(
        "libtest_j2objc.lo",
        "test.o",
        ImmutableList.of("jre_core.h", "test.h", "test.m"));
  }

  @Test
  public void testNestedJ2ObjcLibraryDeps() throws Exception {
    scratch.file("java/com/google/dummy/dummy.java");
    scratch.file(
        "java/com/google/dummy/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(",
        "    name = 'dummy',",
        "    srcs = ['dummy.java'],",
        ")",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        "    deps = [",
        "        ':dummy',",
        "        '//java/com/google/dummy/test:transpile',",
        "    ])");
    addAppleBinaryStarlarkRule(scratch);
    addSimpleBinaryTarget("//java/com/google/dummy:transpile");

    checkObjcArchiveAndLinkActions(
        "libtest_j2objc.lo",
        "test.o",
        ImmutableList.of("jre_core.h", "test.h", "test.m"));

    checkObjcArchiveAndLinkActions(
        "libdummy_j2objc.lo",
        "dummy.o",
        ImmutableList.of("jre_core.h", "dummy.h", "dummy.m"));
  }

  // Tests that a j2objc library can acquire java library information from a Starlark rule target.
  @Test
  public void testJ2ObjcLibraryDepThroughStarlarkRule() throws Exception {
    scratch.file("examples/inner.java");
    scratch.file("examples/outer.java");
    scratch.file(
        "examples/fake_rule.bzl",
        "def _fake_rule_impl(ctx):",
        "  myProvider = ctx.attr.deps[0][JavaInfo]",
        "  return myProvider",
        "",
        "fake_rule = rule(",
        "  implementation = _fake_rule_impl,",
        "  attrs = {'deps': attr.label_list()},",
        "  provides = [JavaInfo],",
        ")");
    scratch.file(
        "examples/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "load('//examples:fake_rule.bzl', 'fake_rule')",
        "java_library(",
        "    name = 'inner',",
        "    srcs = ['inner.java'],",
        ")",
        "fake_rule(",
        "    name = 'propagator',",
        "    deps = [':inner'],",
        ")",
        "java_library(",
        "    name = 'outer',",
        "    srcs = ['outer.java'],",
        "    deps = [':propagator'],",
        ")",
        "j2objc_library(",
        "    name = 'transpile',",
        "    deps = [",
        "        ':outer',",
        "    ],",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = [':transpile'],",
        ")");

    ConfiguredTarget objcTarget = getConfiguredTarget("//examples:lib");

    // The only way that //examples:lib can see inner's archive is through the Starlark rule.
    CcLinkingContext ccLinkingContext = objcTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .contains("libinner_j2objc.lo");
  }

  @Test
  public void testJ2ObjcTranspiledHeaderInCompilationAction() throws Exception {
    scratch.file("app/lib.m");
    scratch.file(
        "app/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = ['//java/com/google/dummy/test:transpile'],",
        ")");

    checkObjcCompileActions(
        getBinArtifact("liblib.a", getConfiguredTarget("//app:lib")),
        "lib.o",
        ImmutableList.of("test.h"));
  }

  @Test
  public void testProtoToolchainForJ2ObjcFlag() throws Exception {
    useConfiguration(
        "--proto_toolchain_for_java=//tools/proto/toolchains:java",
        "--proto_toolchain_for_j2objc=//tools/j2objc:alt_j2objc_proto_toolchain");

    scratch.file("tools/j2objc/proto_plugin_binary");
    scratch.file("tools/j2objc/alt_proto_runtime.h");
    scratch.file("tools/j2objc/alt_proto_runtime.m");
    scratch.file("tools/j2objc/proto_to_exclude.proto");

    scratch.overwriteFile(
        "tools/j2objc/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "package(default_visibility=['//visibility:public'])",
        "exports_files(['j2objc_deploy.jar'])",
        "py_binary(",
        "    name = 'j2objc_wrapper_binary',",
        "    srcs = ['j2objc_wrapper_binary.py'],",
        ")",
        "proto_library(",
        "    name = 'excluded_protos',",
        "    srcs = ['proto_to_exclude.proto'],",
        ")",
        "py_binary(",
        "    name = 'j2objc_header_map_binary',",
        "    srcs = ['j2objc_header_map_binary.py'],",
        ")",
        "proto_lang_toolchain(",
        "    name = 'alt_j2objc_proto_toolchain',",
        "    command_line = '--PLUGIN_j2objc_out=file_dir_mapping,generate_class_mappings:$(OUT)',",
        "    plugin_format_flag = '--plugin=protoc-gen-PLUGIN_j2objc=%s', ",
        "    plugin = ':alt_proto_plugin',",
        "    runtime = ':alt_proto_runtime',",
        "    blacklisted_protos = [':excluded_protos'],",
        ")",
        "proto_library(",
        "   name = 'excluded_proto_library',",
        "   srcs = ['proto_to_exclude.proto'],",
        ")",
        "objc_library(",
        "    name = 'alt_proto_runtime',",
        "    hdrs = ['alt_proto_runtime.h'],",
        "    srcs = ['alt_proto_runtime.m'],",
        ")",
        "filegroup(",
        "    name = 'alt_proto_plugin',",
        "    srcs = ['proto_plugin_binary']",
        ")");

    scratch.file("java/com/google/dummy/test/proto/test.java");
    scratch.file("java/com/google/dummy/test/proto/test.proto");
    scratch.file(
        "java/com/google/dummy/test/proto/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "package(default_visibility=['//visibility:public'])",
        "proto_library(",
        "    name = 'test_proto',",
        "    srcs = ['test.proto'],",
        "    deps = ['//tools/j2objc:excluded_proto_library'],",
        ")",
        "java_proto_library(",
        "    name = 'test_java_proto',",
        "    deps = [':test_proto'],",
        ")",
        "java_library(",
        "    name = 'test',",
        "    srcs = ['test.java'],",
        "    deps = [':test_java_proto'])",
        "",
        "j2objc_library(",
        "    name = 'transpile',",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        "    deps = ['test'])");

    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/dummy/test/proto:transpile");

    CcLinkingContext ccLinkingContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcLinkingContext();
    assertThat(baseArtifactNames(ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .containsExactly(
            "libjre_core_lib.a",
            "libalt_proto_runtime.a",
            "libtest_j2objc.lo",
            "libtest_proto_j2objc.lo");

    CcCompilationContext ccCompilationContext =
        j2objcLibraryTarget.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(baseArtifactNames(ccCompilationContext.getDeclaredIncludeSrcs()))
        .containsExactly("jre_core.h", "alt_proto_runtime.h", "test.j2objc.pb.h", "test.h");
  }

  @Test
  public void testJ2ObjcDeadCodeRemovalActionWithOptFlag() throws Exception {
    useConfiguration("--j2objc_dead_code_removal");
    addSimpleJ2ObjcLibraryWithEntryClasses();
    addAppleBinaryStarlarkRule(scratch);
    addSimpleBinaryTarget("//java/com/google/app/test:transpile");

    RuleConfiguredTarget appTarget = (RuleConfiguredTarget) getConfiguredTarget("//app:app");
    Artifact prunedArchive =
        getBinArtifact(
            "app/_j2objc_pruned/app/java/com/google/app/test/libtest_j2objc_pruned.lo", appTarget);
    Action action = getGeneratingAction(prunedArchive);
    ConfiguredTarget javaTarget = getConfiguredTarget("//java/com/google/app/test:test");
    Artifact inputArchive = getBinArtifact("libtest_j2objc.lo", javaTarget);
    Artifact headerMappingFile = getBinArtifact("test.mapping.j2objc", javaTarget);
    Artifact dependencyMappingFile = getBinArtifact("test.dependency_mapping.j2objc", javaTarget);
    Artifact archiveSourceMappingFile =
        getBinArtifact("test.archive_source_mapping.j2objc", javaTarget);
    String execPath = removeConfigFragment(getRuleContext(javaTarget).getBinFragment() + "/");

    assertContainsSublist(
        removeConfigFragment(ImmutableList.copyOf(paramFileArgsForAction(action))),
        ImmutableList.of(
            "--input_archive",
            removeConfigFragment(inputArchive.getExecPathString()),
            "--output_archive",
            removeConfigFragment(prunedArchive.getExecPathString()),
            "--dummy_archive",
            execPath
                + TestConstants.TOOLS_REPOSITORY_PATH_PREFIX
                + "tools/objc/dummy/libdummy_lib.a",
            "--dependency_mapping_files",
            removeConfigFragment(dependencyMappingFile.getExecPathString()),
            "--header_mapping_files",
            removeConfigFragment(headerMappingFile.getExecPathString()),
            "--archive_source_mapping_files",
            removeConfigFragment(archiveSourceMappingFile.getExecPathString()),
            "--entry_classes",
            "com.google.app.test.test"));

    SpawnAction deadCodeRemovalAction = (SpawnAction) getGeneratingAction(prunedArchive);
    assertThat(deadCodeRemovalAction.getArguments().get(0))
        .contains("tools/objc/j2objc_dead_code_pruner_binary");
    assertThat(deadCodeRemovalAction.getOutputs()).containsExactly(prunedArchive);
  }

  /** Returns the actions created by the action template corresponding to given artifact. */
  protected ImmutableList<CppCompileAction> getActionsForInputsOfGeneratingActionTemplate(
      Artifact artifact, TreeFileArtifact treeFileArtifact) throws ActionExecutionException {
    CppCompileActionTemplate template =
        (CppCompileActionTemplate) getActionGraph().getGeneratingAction(artifact);
    return template.generateActionsForInputArtifacts(
        ImmutableSet.of(treeFileArtifact), ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);
  }

  @Test
  public void testCompileActionTemplateFromGenJar() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_i386", "--ios_minimum_os=1.0");
    addSimpleJ2ObjcLibraryWithJavaPlugin();
    Artifact archive = j2objcArchive("//java/com/google/app/test:transpile", "test");
    CommandAction archiveAction = (CommandAction) getGeneratingAction(archive);
    Artifact objectFilesFromGenJar =
        getFirstArtifactEndingWith(archiveAction.getInputs(), "source_files");

    assertThat(objectFilesFromGenJar.isTreeArtifact()).isTrue();
    assertThat(objectFilesFromGenJar.getRootRelativePathString())
        .isEqualTo("java/com/google/app/test/_objs/test/non_arc/source_files");

    ActionAnalysisMetadata actionTemplate =
        getActionGraph().getGeneratingAction(objectFilesFromGenJar);
    Artifact sourceFilesFromGenJar =
        getFirstArtifactEndingWith(actionTemplate.getInputs(), "source_files");
    assertThat(sourceFilesFromGenJar.getRootRelativePathString())
        .isEqualTo("java/com/google/app/test/_j2objc/src_jar_files/test/source_files");
    // We can't easily access the header artifact through the middleman artifact, so use its
    // expected path directly.
    String headerFilesFromGenJar =
        "java/com/google/app/test/_j2objc/src_jar_files/test/header_files";

    // The files contained inside the tree artifacts are not known until execution time.
    // Therefore we need to fake some files inside them to test the action template in this
    // analysis-time test.
    TreeFileArtifact oneSourceFileFromGenJar =
        TreeFileArtifact.createTreeOutput((SpecialArtifact) sourceFilesFromGenJar, "children1.m");
    TreeFileArtifact oneObjFileFromGenJar =
        TreeFileArtifact.createTreeOutput((SpecialArtifact) objectFilesFromGenJar, "children1.o");
    Iterable<CppCompileAction> compileActions =
        getActionsForInputsOfGeneratingActionTemplate(
            objectFilesFromGenJar, oneSourceFileFromGenJar);
    CommandAction compileAction = Iterables.getOnlyElement(compileActions);
    ConfiguredTarget j2objcLibraryTarget =
        getConfiguredTarget("//java/com/google/dummy/test:transpile");
    String genfilesFragment = getRuleContext(j2objcLibraryTarget).getGenfilesFragment().toString();
    String binFragment = getRuleContext(j2objcLibraryTarget).getBinFragment().toString();

    String commandLine = Joiner.on(" ").join(compileAction.getArguments());
    ImmutableList<String> expectedArgs =
        new ImmutableList.Builder<String>()
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
            .add("-O0")
            .add("-DDEBUG=1")
            .add("-iquote")
            .add(".")
            .add("-iquote")
            .add(genfilesFragment)
            .add("-I")
            .add(binFragment + "/java/com/google/app/test/_j2objc/test")
            .add("-I")
            .add(headerFilesFromGenJar)
            .add("-fno-strict-overflow")
            .add("-fno-objc-arc")
            .add("-c")
            .add(oneSourceFileFromGenJar.getExecPathString())
            .add("-o")
            .add(oneObjFileFromGenJar.getExecPathString())
            .build();
    for (String expectedArg : expectedArgs) {
      assertThat(commandLine).contains(expectedArg);
    }
  }

  @Test
  public void j2objcLibrary_noPropperTag_lockdownError() throws Exception {
    useConfiguration("--incompatible_j2objc_library_migration");
    scratch.file("test/BUILD", "j2objc_library(name = 'test')");
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//test");

    assertContainsEvent(
        "j2objc_library is locked. Please do not use this rule since it will be deleted in the"
            + " future.");
  }

  @Test
  public void j2objcLibrary_withPropperTag_noError() throws Exception {
    useConfiguration("--incompatible_j2objc_library_migration");
    scratch.file(
        "test/BUILD",
        "j2objc_library(",
        "    name = 'test',",
        "    tags = ['__J2OBJC_LIBRARY_MIGRATION_DO_NOT_USE_WILL_BREAK__'],",
        ")");

    getConfiguredTarget("//test");

    assertNoEvents();
  }
}
