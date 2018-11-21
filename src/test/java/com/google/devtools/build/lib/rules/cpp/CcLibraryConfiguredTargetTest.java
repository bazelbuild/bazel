// Copyright 2015 The Bazel Authors. All rights reserved.
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
// Copyright 2006 Google Inc. All rights reserved.

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.extra.CppLinkInfo;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.protobuf.TextFormat;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * "White-box" unit test of cc_library rule.
 */
@RunWith(JUnit4.class)
public class CcLibraryConfiguredTargetTest extends BuildViewTestCase {
  private static final PathFragment STL_CPPMAP = PathFragment.create("stl.cppmap");
  private static final PathFragment CROSSTOOL_CPPMAP = PathFragment.create("crosstool.cppmap");

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    return builder.addRuleDefinition(new TestRuleClassProvider.MakeVariableTesterRule()).build();
  }

  @Before
  public final void createFiles() throws Exception {
    scratch.file(
        "hello/BUILD",
        "cc_library(",
        "  name = 'hello',",
        "  srcs = ['hello.cc'],",
        ")",
        "cc_library(",
        "  name = 'hello_static',",
        "  srcs = ['hello.cc'],",
        "  linkstatic = 1,",
        ")",
        "cc_library(",
        "  name = 'hello_alwayslink',",
        "  srcs = ['hello.cc'],",
        "  alwayslink = 1,",
        ")",
        "cc_binary(",
        "  name = 'hello_bin',",
        "  srcs = ['hello_main.cc'],",
        ")");
    scratch.file(
        "hello/hello.cc",
        "#include <stdio.h>",
        "int hello_world() { printf(\"Hello, world!\\n\"); }");
    scratch.file(
        "hello/hello_main.cc",
        "#include <stdio.h>",
        "int main() { printf(\"Hello, world!\\n\"); }");
  }

  private CppCompileAction getCppCompileAction(String label) throws Exception {
    return getCppCompileAction(getConfiguredTarget(label));
  }

  private CppCompileAction getCppCompileAction(ConfiguredTarget target) throws Exception {
    List<CppCompileAction> compilationSteps =
        actionsTestUtil().findTransitivePrerequisitesOf(
            ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), ".a"),
            CppCompileAction.class);
    return compilationSteps.get(0);
  }

  private CppModuleMapAction getCppModuleMapAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    CppModuleMap cppModuleMap =
        target.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap();
    return (CppModuleMapAction) getGeneratingAction(cppModuleMap.getArtifact());
  }

  private void assertNoCppModuleMapAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    assertThat(target.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap()).isNull();
  }

  public void checkWrongExtensionInArtifactNamePattern(
      String categoryName, ImmutableList<String> correctExtensions) throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION,
            MockCcSupport.TARGETS_WINDOWS_CONFIGURATION,
            "supports_interface_shared_objects: true",
            "artifact_name_pattern {"
                + "   category_name: '"
                + categoryName
                + "'"
                + "   prefix: ''"
                + "   extension: '.wrong_ext'"
                + "}");

    try {
      useConfiguration();
      fail("Should fail");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              String.format(
                  "Unrecognized file extension '.wrong_ext', allowed "
                     + "extensions are %s, please check artifact_name_pattern configuration for %s "
                     + "in your CROSSTOOL.",
                  StringUtil.joinEnglishList(correctExtensions, "or", "'"), categoryName));
    }
  }

  @Test
  public void testDefinesAndMakeVariables() throws Exception {
    ConfiguredTarget l = scratchConfiguredTarget("a", "l",
        "cc_library(name='l', srcs=['l.cc'], defines=['V=$(FOO)'], toolchains=[':v'])",
        "make_variable_tester(name='v', variables={'FOO': 'BAR'})");
    assertThat(l.get(CcInfo.PROVIDER).getCcCompilationContext().getDefines()).contains("V=BAR");
  }

  @Test
  public void testMisconfiguredCrosstoolRaisesErrorWhenLinking() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.NO_LEGACY_FEATURES_FEATURE,
            MockCcSupport.EMPTY_COMPILE_ACTION_CONFIG,
            MockCcSupport.PIC_FEATURE);
    useConfiguration();

    checkError(
        "test",
        "test",
        "Expected action_config for 'c++-link-static-library' to be configured",
        "cc_library(name = 'test', srcs = ['test.cc'])");
  }

  @Test
  public void testMisconfiguredCrosstoolRaisesErrorWhenCompiling() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.NO_LEGACY_FEATURES_FEATURE,
            MockCcSupport.EMPTY_STATIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.PIC_FEATURE);
    useConfiguration();

    checkError(
        "test",
        "test",
        "Expected action_config for 'c++-compile' to be configured",
        "cc_library(name = 'test', srcs = ['test.cc'])");
  }

  @Test
  public void testFilesToBuild() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    String cpu = getTargetConfiguration().getCpu();
    Artifact archive = getBinArtifact("libhello.a", hello);
    Artifact implSharedObject = getBinArtifact("libhello.so", hello);
    Artifact implInterfaceSharedObject = getBinArtifact("libhello.ifso", hello);
    Artifact implSharedObjectLink =
        getSharedArtifact("_solib_" + cpu + "/libhello_Slibhello.so", hello);
    Artifact implInterfaceSharedObjectLink =
        getSharedArtifact("_solib_" + cpu + "/libhello_Slibhello.ifso", hello);
    assertThat(getFilesToBuild(hello)).containsExactly(archive, implSharedObject,
        implInterfaceSharedObject);
    assertThat(LinkerInputs.toLibraryArtifacts(
        hello.getProvider(CcNativeLibraryProvider.class).getTransitiveCcNativeLibraries()))
        .containsExactly(implInterfaceSharedObjectLink);
    assertThat(
            hello
                .get(CcInfo.PROVIDER)
                .getCcLinkingInfo()
                .getDynamicModeParamsForExecutable()
                .getDynamicLibrariesForRuntime())
        .containsExactly(implSharedObjectLink);
  }

  @Test
  public void testFilesToBuildWithoutDSO() throws Exception {
    CrosstoolConfig.CrosstoolRelease.Builder release = CrosstoolConfig.CrosstoolRelease.newBuilder()
        .mergeFrom(CrosstoolConfigurationHelper.simpleCompleteToolchainProto());
    release.getToolchainBuilder(0)
        .setTargetCpu("k8")
        .setCompiler("compiler")
        .clearLinkingModeFlags();

    scratch.file("crosstool/BUILD",
        "cc_toolchain_suite(",
        "    name = 'crosstool',",
        "    toolchains = {'k8|compiler': ':cc-compiler-k8'})",
        "filegroup(name = 'empty')",
        "cc_toolchain(",
        "    name = 'cc-compiler-k8',",
        "    output_licenses = ['unencumbered'],",
        "    cpu = 'k8',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    coverage_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    static_runtime_libs = [':empty'],",
        "    dynamic_runtime_libs = [':empty'],",
        "    all_files = ':empty',",
        "    licenses = ['unencumbered'])");
    scratch.file("crosstool/CROSSTOOL", TextFormat.printToString(release));

    // This is like the preceding test, but with a toolchain that can't build '.so' files
    useConfiguration("--crosstool_top=//crosstool:crosstool", "--compiler=compiler",
        "--cpu=k8", "--host_cpu=k8");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact archive = getBinArtifact("libhello.a", hello);
    assertThat(getFilesToBuild(hello)).containsExactly(archive);
  }

  @Test
  public void testFilesToBuildWithInterfaceSharedObjects() throws Exception {
    useConfiguration("--interface_shared_objects");
    useConfiguration("--cpu=k8");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    String cpu = getTargetConfiguration().getCpu();
    Artifact archive = getBinArtifact("libhello.a", hello);
    Artifact sharedObject = getBinArtifact("libhello.ifso", hello);
    Artifact implSharedObject = getBinArtifact("libhello.so", hello);
    Artifact sharedObjectLink =
        getSharedArtifact("_solib_" + cpu + "/libhello_Slibhello.ifso", hello);
    Artifact implSharedObjectLink =
        getSharedArtifact("_solib_" + cpu + "/libhello_Slibhello.so", hello);
    assertThat(getFilesToBuild(hello)).containsExactly(archive, sharedObject, implSharedObject);
    assertThat(LinkerInputs.toLibraryArtifacts(
        hello.getProvider(CcNativeLibraryProvider.class).getTransitiveCcNativeLibraries()))
        .containsExactly(sharedObjectLink);
    assertThat(
            hello
                .get(CcInfo.PROVIDER)
                .getCcLinkingInfo()
                .getDynamicModeParamsForExecutable()
                .getDynamicLibrariesForRuntime())
        .containsExactly(implSharedObjectLink);
  }

  @Test
  public void testEmptyLinkopts() throws Exception {
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertThat(
            hello
                .get(CcInfo.PROVIDER)
                .getCcLinkingInfo()
                .getDynamicModeParamsForExecutable()
                .getLinkopts()
                .isEmpty())
        .isTrue();
  }

  @Test
  public void testSoName() throws Exception {
    // Without interface shared libraries.
    useConfiguration("--nointerface_shared_objects");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact sharedObject = getOnlyElement(FileType.filter(getFilesToBuild(hello),
        CppFileTypes.SHARED_LIBRARY));
    CppLinkAction action = (CppLinkAction) getGeneratingAction(sharedObject);
    for (String option : action.getLinkCommandLine().getLinkopts()) {
      assertThat(option).doesNotContain("-Wl,-soname");
    }

    // With interface shared libraries.
    useConfiguration("--interface_shared_objects");
    useConfiguration("--cpu=k8");
    hello = getConfiguredTarget("//hello:hello");
    sharedObject =
        FileType.filter(getFilesToBuild(hello), CppFileTypes.SHARED_LIBRARY).iterator().next();
    action = (CppLinkAction) getGeneratingAction(sharedObject);
    assertThat(action.getLinkCommandLine().getLinkopts())
        .contains("-Wl,-soname=libhello_Slibhello.so");
  }

  @Test
  public void testCppLinkActionExtraActionInfoWithoutSharedLibraries() throws Exception {
    useConfiguration("--nointerface_shared_objects");

    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact sharedObject = getOnlyElement(FileType.filter(getFilesToBuild(hello),
        CppFileTypes.SHARED_LIBRARY));
    CppLinkAction action = (CppLinkAction) getGeneratingAction(sharedObject);

    ExtraActionInfo.Builder builder = action.getExtraActionInfo(actionKeyContext);
    ExtraActionInfo info = builder.build();
    assertThat(info.getMnemonic()).isEqualTo("CppLink");

    CppLinkInfo cppLinkInfo = info.getExtension(CppLinkInfo.cppLinkInfo);
    assertThat(cppLinkInfo).isNotNull();

    Iterable<String> inputs =
        Artifact.asExecPaths(action.getLinkCommandLine().getLinkerInputArtifacts());
    assertThat(cppLinkInfo.getInputFileList()).containsExactlyElementsIn(inputs);
    assertThat(cppLinkInfo.getOutputFile())
        .isEqualTo(action.getPrimaryOutput().getExecPathString());
    assertThat(cppLinkInfo.hasInterfaceOutputFile()).isFalse();
    assertThat(cppLinkInfo.getLinkTargetType())
        .isEqualTo(action.getLinkCommandLine().getLinkTargetType().name());
    assertThat(cppLinkInfo.getLinkStaticness())
        .isEqualTo(action.getLinkCommandLine().getLinkingMode().name());
    Iterable<String> linkstamps = Artifact.asExecPaths(action.getLinkstampObjects());
    assertThat(cppLinkInfo.getLinkStampList()).containsExactlyElementsIn(linkstamps);
    Iterable<String> buildInfoHeaderArtifacts =
        Artifact.asExecPaths(action.getBuildInfoHeaderArtifacts());
    assertThat(cppLinkInfo.getBuildInfoHeaderArtifactList())
        .containsExactlyElementsIn(buildInfoHeaderArtifacts);
    assertThat(cppLinkInfo.getLinkOptList()).containsExactlyElementsIn(action.getArguments());
  }

  @Test
  public void testCppLinkActionExtraActionInfoWithSharedLibraries() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact sharedObject =
        FileType.filter(getFilesToBuild(hello), CppFileTypes.SHARED_LIBRARY).iterator().next();
    CppLinkAction action = (CppLinkAction) getGeneratingAction(sharedObject);

    ExtraActionInfo.Builder builder = action.getExtraActionInfo(actionKeyContext);
    ExtraActionInfo info = builder.build();
    assertThat(info.getMnemonic()).isEqualTo("CppLink");

    CppLinkInfo cppLinkInfo = info.getExtension(CppLinkInfo.cppLinkInfo);
    assertThat(cppLinkInfo).isNotNull();

    Iterable<String> inputs =
        Artifact.asExecPaths(action.getLinkCommandLine().getLinkerInputArtifacts());
    assertThat(cppLinkInfo.getInputFileList()).containsExactlyElementsIn(inputs);
    assertThat(cppLinkInfo.getOutputFile())
        .isEqualTo(action.getPrimaryOutput().getExecPathString());
    assertThat(cppLinkInfo.getLinkTargetType())
        .isEqualTo(action.getLinkCommandLine().getLinkTargetType().name());
    assertThat(cppLinkInfo.getLinkStaticness())
        .isEqualTo(action.getLinkCommandLine().getLinkingMode().name());
    Iterable<String> linkstamps = Artifact.asExecPaths(action.getLinkstampObjects());
    assertThat(cppLinkInfo.getLinkStampList()).containsExactlyElementsIn(linkstamps);
    Iterable<String> buildInfoHeaderArtifacts =
        Artifact.asExecPaths(action.getBuildInfoHeaderArtifacts());
    assertThat(cppLinkInfo.getBuildInfoHeaderArtifactList())
        .containsExactlyElementsIn(buildInfoHeaderArtifacts);
    assertThat(cppLinkInfo.getLinkOptList()).containsExactlyElementsIn(action.getArguments());
  }

  @Test
  public void testLinkActionCanConsumeArtifactExtensions() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.STATIC_LINK_TWEAKED_CONFIGURATION);
    useConfiguration("--features=" + Link.LinkTargetType.STATIC_LIBRARY.getActionName());
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact archive =
        FileType.filter(getFilesToBuild(hello), FileType.of(".lib")).iterator().next();

    CppLinkAction action = (CppLinkAction) getGeneratingAction(archive);

    assertThat(action.getArguments()).contains(archive.getExecPathString());
  }

  @Test
  public void testObjectFileNamesCanBeSpecifiedInToolchain() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "artifact_name_pattern {"
                + "   category_name: 'object_file'"
                + "   prefix: ''"
                + "   extension: '.obj'"
                + "}");

    useConfiguration();
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertThat(artifactByPath(getFilesToBuild(hello), ".a", ".obj")).isNotNull();
  }

  @Test
  public void testWindowsFileNamePatternsCanBeSpecifiedInToolchain() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION,
            MockCcSupport.TARGETS_WINDOWS_CONFIGURATION,
            "needsPic: false",
            "supports_interface_shared_objects: true",
            "artifact_name_pattern {"
                + "   category_name: 'object_file'"
                + "   prefix: ''"
                + "   extension: '.obj'"
                + "}",
            "artifact_name_pattern {"
                + "   category_name: 'static_library'"
                + "   prefix: ''"
                + "   extension: '.lib'"
                + "}",
            "artifact_name_pattern {"
                + "   category_name: 'alwayslink_static_library'"
                + "   prefix: ''"
                + "   extension: '.lo.lib'"
                + "}",
            "artifact_name_pattern {"
                + "   category_name: 'executable'"
                + "   prefix: ''"
                + "   extension: '.exe'"
                + "}",
            "artifact_name_pattern {"
                + "   category_name: 'dynamic_library'"
                + "   prefix: ''"
                + "   extension: '.dll'"
                + "}",
            "artifact_name_pattern {"
                + "   category_name: 'interface_library'"
                + "   prefix: ''"
                + "   extension: '.if.lib'"
                + "}");

    useConfiguration();

    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact helloObj =
        getBinArtifact("_objs/hello/hello.obj", getConfiguredTarget("//hello:hello"));
    CppCompileAction helloObjAction = (CppCompileAction) getGeneratingAction(helloObj);
    assertThat(helloObjAction).isNotNull();

    Artifact helloLib =
        FileType.filter(getFilesToBuild(hello), CppFileTypes.ARCHIVE).iterator().next();
    assertThat(helloLib.getExecPathString()).endsWith("hello.lib");

    ConfiguredTarget helloAlwaysLink = getConfiguredTarget("//hello:hello_alwayslink");
    Artifact helloLibAlwaysLink =
        FileType.filter(getFilesToBuild(helloAlwaysLink), CppFileTypes.ALWAYS_LINK_LIBRARY)
            .iterator()
            .next();
    assertThat(helloLibAlwaysLink.getExecPathString()).endsWith("hello_alwayslink.lo.lib");

    ConfiguredTarget helloBin = getConfiguredTarget("//hello:hello_bin");
    Artifact helloBinExe = getFilesToBuild(helloBin).iterator().next();
    assertThat(helloBinExe.getExecPathString()).endsWith("hello_bin.exe");

    assertThat(
            artifactsToStrings(getOutputGroup(hello, CcLibrary.DYNAMIC_LIBRARY_OUTPUT_GROUP_NAME)))
        .containsExactly("bin hello/hello.dll", "bin hello/hello.if.lib");
  }

  @Test
  public void testWrongObjectFileArtifactNamePattern() throws Exception {
    checkWrongExtensionInArtifactNamePattern(
        "object_file",
        ArtifactCategory.OBJECT_FILE.getAllowedExtensions());
  }

  @Test
  public void testWrongStaticLibraryArtifactNamePattern() throws Exception {
    checkWrongExtensionInArtifactNamePattern(
        "static_library",
        ArtifactCategory.STATIC_LIBRARY.getAllowedExtensions());
  }

  @Test
  public void testWrongAlwayslinkStaticLibraryArtifactNamePattern() throws Exception {
    checkWrongExtensionInArtifactNamePattern(
        "alwayslink_static_library",
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY.getAllowedExtensions());
  }

  @Test
  public void testWrongExecutableArtifactNamePattern() throws Exception {
    checkWrongExtensionInArtifactNamePattern(
        "executable",
        ArtifactCategory.EXECUTABLE.getAllowedExtensions());
  }

  @Test
  public void testWrongDynamicLibraryArtifactNamePattern() throws Exception {
    checkWrongExtensionInArtifactNamePattern(
        "dynamic_library",
        ArtifactCategory.DYNAMIC_LIBRARY.getAllowedExtensions());
  }

  @Test
  public void testWrongInterfaceLibraryArtifactNamePattern() throws Exception {
    checkWrongExtensionInArtifactNamePattern(
        "interface_library",
        ArtifactCategory.INTERFACE_LIBRARY.getAllowedExtensions());
  }

  @Test
  public void testArtifactSelectionBaseNameTemplating() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.STATIC_LINK_AS_DOT_A_CONFIGURATION);
    useConfiguration("--features=" + Link.LinkTargetType.STATIC_LIBRARY.getActionName());
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact archive =
        FileType.filter(getFilesToBuild(hello), CppFileTypes.ARCHIVE).iterator().next();
    assertThat(archive.getExecPathString()).endsWith("libhello.a");
  }

  @Test
  public void testArtifactsToAlwaysBuild() throws Exception {
    useConfiguration("--cpu=k8");
    // ArtifactsToAlwaysBuild should apply both for static libraries.
    ConfiguredTarget helloStatic = getConfiguredTarget("//hello:hello_static");
    assertThat(artifactsToStrings(getOutputGroup(helloStatic, OutputGroupInfo.HIDDEN_TOP_LEVEL)))
        .containsExactly("bin hello/_objs/hello_static/hello.pic.o");
    Artifact implSharedObject = getBinArtifact("libhello_static.so", helloStatic);
    assertThat(getFilesToBuild(helloStatic)).doesNotContain(implSharedObject);

    // And for shared libraries.
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertThat(artifactsToStrings(getOutputGroup(helloStatic, OutputGroupInfo.HIDDEN_TOP_LEVEL)))
        .containsExactly("bin hello/_objs/hello_static/hello.pic.o");
    implSharedObject = getBinArtifact("libhello.so", hello);
    assertThat(getFilesToBuild(hello)).contains(implSharedObject);
  }

  @Test
  public void testTransitiveArtifactsToAlwaysBuildStatic() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget x = scratchConfiguredTarget(
        "foo", "x",
        "cc_library(name = 'x', srcs = ['x.cc'], deps = [':y'], linkstatic = 1)",
        "cc_library(name = 'y', srcs = ['y.cc'], deps = [':z'])",
        "cc_library(name = 'z', srcs = ['z.cc'])");
    assertThat(artifactsToStrings(getOutputGroup(x, OutputGroupInfo.HIDDEN_TOP_LEVEL)))
        .containsExactly(
            "bin foo/_objs/x/x.pic.o", "bin foo/_objs/y/y.pic.o", "bin foo/_objs/z/z.pic.o");
  }

  @Test
  public void testBuildHeaderModulesAsPrerequisites() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration("--cpu=k8");
    ConfiguredTarget x =

        scratchConfiguredTarget(
            "foo",
            "x",
            "package(features = ['header_modules'])",
            "cc_library(name = 'x', srcs = ['x.cc'], deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseArtifactNames(
        getOutputGroup(x, OutputGroupInfo.COMPILATION_PREREQUISITES)))
        .containsAllOf("y.h", "y.cppmap", "crosstool.cppmap", "x.cppmap", "y.pic.pcm", "x.cc");
  }

  @Test
  public void testCodeCoverage() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration("--cpu=k8", "--collect_code_coverage");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "package(features = ['header_modules'])",
            "cc_library(name = 'x', srcs = ['x.cc'])");
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                x.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR).getInstrumentationMetadataFiles()))
        .containsExactly("x.pic.gcno");
  }

  @Test
  public void testDisablingHeaderModulesWhenDependingOnModuleBuildTransitively() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration();
    scratch.file("module/BUILD",
        "package(features = ['header_modules'])",
        "cc_library(",
        "    name = 'module',",
        "    srcs = ['a.cc', 'a.h'],",
        ")");
    scratch.file("nomodule/BUILD",
        "package(features = ['-header_modules'])",
        "cc_library(",
        "    name = 'nomodule',",
        "    srcs = ['a.cc', 'a.h'],",
        "    deps = ['//module']",
        ")");
    CppCompileAction moduleAction = getCppCompileAction("//module:module");
    assertThat(moduleAction.getCompilerOptions()).contains("module_name://module:module");
    CppCompileAction noModuleAction = getCppCompileAction("//nomodule:nomodule");
    assertThat(noModuleAction.getCompilerOptions()).doesNotContain("module_name://module:module");
  }

  /**
   * Returns the non-system module maps in {@code input}.
   */
  private Iterable<Artifact> getNonSystemModuleMaps(Iterable<Artifact> input) {
    return Iterables.filter(input, new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact input) {
        PathFragment path = input.getExecPath();
        return CppFileTypes.CPP_MODULE_MAP.matches(path)
            && !path.endsWith(STL_CPPMAP)
            && !path.endsWith(CROSSTOOL_CPPMAP);
      }
    });
  }

  /**
   * Returns the header module artifacts in {@code input}.
   */
  private Iterable<Artifact> getHeaderModules(Iterable<Artifact> input) {
    return Iterables.filter(input, new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact input) {
        return CppFileTypes.CPP_MODULE.matches(input.getExecPath());
      }
    });
  }

  /**
   * Returns the flags in {@code input} that reference a header module.
   */
  private Iterable<String> getHeaderModuleFlags(Iterable<String> input) {
    List<String> names = new ArrayList<>();
    for (String flag : input) {
      if (CppFileTypes.CPP_MODULE.matches(flag)) {
        names.add(PathFragment.create(flag).getBaseName());
      }
    }
    return names;
  }

  @Test
  public void testCompileHeaderModules() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            ""
                + "feature { name: 'header_modules' implies: 'use_header_modules' }"
                + "feature { name: 'module_maps' }"
                + "feature { name: 'use_header_modules' }");
    useConfiguration("--cpu=k8");
    scratch.file("module/BUILD",
        "package(features = ['header_modules'])",
        "cc_library(",
        "    name = 'a',",
        "    srcs = ['a.h', 'a.cc'],",
        "    deps = ['b']",
        ")",
        "cc_library(",
        "    name = 'b',",
        "    srcs = ['b.h'],",
        "    textual_hdrs = ['t.h'],",
        ")");
    ConfiguredTarget moduleB = getConfiguredTarget("//module:b");
    Artifact bModuleArtifact = getBinArtifact("_objs/b/b.pic.pcm", moduleB);
    CppCompileAction bModuleAction = (CppCompileAction) getGeneratingAction(bModuleArtifact);
    assertThat(bModuleAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/b.h"), getSourceArtifact("module/t.h"));
    assertThat(bModuleAction.getInputs()).contains(getGenfilesArtifact("b.cppmap", moduleB));

    ConfiguredTarget moduleA = getConfiguredTarget("//module:a");
    Artifact aObjectArtifact = getBinArtifact("_objs/a/a.pic.o", moduleA);
    CppCompileAction aObjectAction = (CppCompileAction) getGeneratingAction(aObjectArtifact);
    assertThat(aObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/a.cc"));
    assertThat(aObjectAction.getCcCompilationContext().getTransitiveModules(true))
        .contains(getBinArtifact("_objs/b/b.pic.pcm", moduleB));
    assertThat(aObjectAction.getInputs()).contains(getGenfilesArtifact("b.cppmap", moduleB));
    assertNoEvents();
  }

  private void setupPackagesForSourcesWithSameBaseNameTests() throws Exception {
    scratch.file(
        "foo/BUILD",
        "cc_library(",
        "    name = 'lib',",
        "    srcs = ['a.cc', 'subpkg1/b.cc', 'subpkg1/a.c', '//bar:srcs', 'subpkg2/A.c'],",
        ")");
    scratch.file("bar/BUILD", "filegroup(name = 'srcs', srcs = ['a.cpp'])");
  }

  @Test
  public void testContainingSourcesWithSameBaseName() throws Exception {
    AnalysisMock.get().ccSupport().setup(mockToolsConfig);
    useConfiguration("--cpu=k8");
    setupPackagesForSourcesWithSameBaseNameTests();
    getConfiguredTarget("//foo:lib");

    Artifact a0 = getBinArtifact("_objs/lib/0/a.pic.o", getConfiguredTarget("//foo:lib"));
    Artifact a1 = getBinArtifact("_objs/lib/1/a.pic.o", getConfiguredTarget("//foo:lib"));
    Artifact a2 = getBinArtifact("_objs/lib/2/a.pic.o", getConfiguredTarget("//foo:lib"));
    Artifact a3 = getBinArtifact("_objs/lib/3/A.pic.o", getConfiguredTarget("//foo:lib"));
    Artifact b = getBinArtifact("_objs/lib/b.pic.o", getConfiguredTarget("//foo:lib"));

    assertThat(getGeneratingAction(a0)).isNotNull();
    assertThat(getGeneratingAction(a1)).isNotNull();
    assertThat(getGeneratingAction(a2)).isNotNull();
    assertThat(getGeneratingAction(a3)).isNotNull();
    assertThat(getGeneratingAction(b)).isNotNull();

    assertThat(getGeneratingAction(a0).getInputs()).contains(getSourceArtifact("foo/a.cc"));
    assertThat(getGeneratingAction(a1).getInputs()).contains(getSourceArtifact("foo/subpkg1/a.c"));
    assertThat(getGeneratingAction(a2).getInputs()).contains(getSourceArtifact("bar/a.cpp"));
    assertThat(getGeneratingAction(a3).getInputs()).contains(getSourceArtifact("foo/subpkg2/A.c"));
    assertThat(getGeneratingAction(b).getInputs()).contains(getSourceArtifact("foo/subpkg1/b.cc"));
  }

  private void setupPackagesForModuleTests(boolean useHeaderModules) throws Exception {
    scratch.file("module/BUILD",
        "package(features = ['header_modules'])",
        "cc_library(",
        "    name = 'b',",
        "    srcs = ['b.h'],",
        "    deps = ['//nomodule:a'],",
        ")",
        "cc_library(",
        "    name = 'g',",
        "    srcs = ['g.h', 'g.cc'],",
        "    deps = ['//nomodule:c'],",
        ")",
        "cc_library(",
        "    name = 'j',",
        "    srcs = ['j.h', 'j.cc'],",
        "    deps = ['//nomodule:c', '//nomodule:i'],",
        ")");
    scratch.file("nomodule/BUILD",
        "package(features = ['-header_modules'"
            + (useHeaderModules ? ", 'use_header_modules'" : "") + "])",
        "cc_library(",
        "    name = 'y',",
        "    srcs = ['y.h'],",
        ")",
        "cc_library(",
        "    name = 'z',",
        "    srcs = ['z.h'],",
        "    deps = [':y'],",
        ")",
        "cc_library(",
        "    name = 'a',",
        "    srcs = ['a.h'],",
        "    deps = [':z'],",
        ")",
        "cc_library(",
        "    name = 'c',",
        "    srcs = ['c.h', 'c.cc'],",
        "    deps = ['//module:b'],",
        ")",
        "cc_library(",
        "    name = 'd',",
        "    srcs = ['d.h', 'd.cc'],",
        "    deps = [':c'],",
        ")",
        "cc_library(",
        "    name = 'e',",
        "    srcs = ['e.h'],",
        "    deps = [':a'],",
        ")",
        "cc_library(",
        "    name = 'f',",
        "    srcs = ['f.h', 'f.cc'],",
        "    deps = [':e'],",
        ")",
        "cc_library(",
        "    name = 'h',",
        "    srcs = ['h.h', 'h.cc'],",
        "    deps = ['//module:g'],",
        ")",
        "cc_library(",
        "    name = 'i',",
        "    srcs = ['i.h', 'i.cc'],",
        "    deps = [':h'],",
        ")");
    }

  @Test
  public void testCompileHeaderModulesTransitively() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration("--cpu=k8");
    setupPackagesForModuleTests(/*useHeaderModules=*/false);

    // The //nomodule:f target only depends on non-module targets, thus it should be module-free.
    ConfiguredTarget nomoduleF = getConfiguredTarget("//nomodule:f");
    ConfiguredTarget nomoduleE = getConfiguredTarget("//nomodule:e");
    assertThat(getGeneratingAction(getBinArtifact("_objs/f/f.pic.pcm", nomoduleF))).isNull();
    Artifact fObjectArtifact = getBinArtifact("_objs/f/f.pic.o", nomoduleF);
    CppCompileAction fObjectAction = (CppCompileAction) getGeneratingAction(fObjectArtifact);
    // Only the module map of f itself itself and the direct dependencies are needed.
    assertThat(getNonSystemModuleMaps(fObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("f.cppmap", nomoduleF), getGenfilesArtifact("e.cppmap", nomoduleE));
    assertThat(getHeaderModules(fObjectAction.getInputs())).isEmpty();
    assertThat(fObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("nomodule/f.cc"));
    assertThat(getHeaderModuleFlags(fObjectAction.getCompilerOptions())).isEmpty();

    // The //nomodule:c target will get the header module for //module:b, which is a direct
    // dependency.
    ConfiguredTarget nomoduleC = getConfiguredTarget("//nomodule:c");
    assertThat(getGeneratingAction(getBinArtifact("_objs/c/c.pic.pcm", nomoduleC))).isNull();
    Artifact cObjectArtifact = getBinArtifact("_objs/c/c.pic.o", nomoduleC);
    CppCompileAction cObjectAction = (CppCompileAction) getGeneratingAction(cObjectArtifact);
    assertThat(getNonSystemModuleMaps(cObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("b.cppmap", "//module:b"),
            getGenfilesArtifact("c.cppmap", nomoduleC));
    assertThat(getHeaderModules(cObjectAction.getInputs())).isEmpty();
    // All headers of transitive dependencies that are built as modules are needed as entry points
    // for include scanning.
    assertThat(cObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("nomodule/c.cc"));
    assertThat(cObjectAction.getMainIncludeScannerSource()).isEqualTo(
        getSourceArtifact("nomodule/c.cc"));
    assertThat(getHeaderModuleFlags(cObjectAction.getCompilerOptions())).isEmpty();

    // The //nomodule:d target depends on //module:b via one indirection (//nomodule:c).
    getConfiguredTarget("//nomodule:d");
    assertThat(
            getGeneratingAction(
                getBinArtifact("_objs/d/d.pic.pcm", getConfiguredTarget("//nomodule:d"))))
        .isNull();
    Artifact dObjectArtifact =
        getBinArtifact("_objs/d/d.pic.o", getConfiguredTarget("//nomodule:d"));
    CppCompileAction dObjectAction = (CppCompileAction) getGeneratingAction(dObjectArtifact);
    // Module map 'c.cppmap' is needed because it is a direct dependency.
    assertThat(getNonSystemModuleMaps(dObjectAction.getInputs())).containsExactly(
        getGenfilesArtifact("c.cppmap", "//nomodule:c"),
        getGenfilesArtifact("d.cppmap", "//nomodule:d"));
    assertThat(getHeaderModules(dObjectAction.getInputs())).isEmpty();
    assertThat(dObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("nomodule/d.cc"));
    assertThat(getHeaderModuleFlags(dObjectAction.getCompilerOptions())).isEmpty();

    // The //module:j target depends on //module:g via //nomodule:h and on //module:b via
    // both //module:g and //nomodule:c.
    ConfiguredTarget moduleJ = getConfiguredTarget("//module:j");
    Artifact jObjectArtifact = getBinArtifact("_objs/j/j.pic.o", moduleJ);
    CppCompileAction jObjectAction = (CppCompileAction) getGeneratingAction(jObjectArtifact);
    assertThat(getHeaderModules(jObjectAction.getCcCompilationContext().getTransitiveModules(true)))
        .containsExactly(
            getBinArtifact("_objs/b/b.pic.pcm", getConfiguredTarget("//module:b")),
            getBinArtifact("_objs/g/g.pic.pcm", getConfiguredTarget("//module:g")));
    assertThat(jObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/j.cc"));
    assertThat(jObjectAction.getMainIncludeScannerSource()).isEqualTo(
        getSourceArtifact("module/j.cc"));
    assertThat(getHeaderModules(jObjectAction.getCcCompilationContext().getTransitiveModules(true)))
        .containsExactly(
            getBinArtifact("_objs/b/b.pic.pcm", getConfiguredTarget("//module:b")),
            getBinArtifact("_objs/g/g.pic.pcm", getConfiguredTarget("//module:g")));
  }

  @Test
  public void testCompileUsingHeaderModulesTransitively() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration("--cpu=k8");
    setupPackagesForModuleTests( /*useHeaderModules=*/true);

    ConfiguredTarget nomoduleF = getConfiguredTarget("//nomodule:f");
    Artifact fObjectArtifact =
        getBinArtifact("_objs/f/f.pic.o", getConfiguredTarget("//nomodule:f"));
    CppCompileAction fObjectAction = (CppCompileAction) getGeneratingAction(fObjectArtifact);
    // Only the module map of f itself itself and the direct dependencies are needed.
    assertThat(getNonSystemModuleMaps(fObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("f.cppmap", nomoduleF),
            getGenfilesArtifact("e.cppmap", "//nomodule:e"));

    getConfiguredTarget("//nomodule:c");
    Artifact cObjectArtifact =
        getBinArtifact("_objs/c/c.pic.o", getConfiguredTarget("//nomodule:c"));
    CppCompileAction cObjectAction = (CppCompileAction) getGeneratingAction(cObjectArtifact);
    assertThat(getNonSystemModuleMaps(cObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("b.cppmap", "//module:b"),
            getGenfilesArtifact("c.cppmap", "//nomodule:c"));
    assertThat(getHeaderModules(cObjectAction.getCcCompilationContext().getTransitiveModules(true)))
        .containsExactly(getBinArtifact("_objs/b/b.pic.pcm", getConfiguredTarget("//module:b")));

    getConfiguredTarget("//nomodule:d");
    Artifact dObjectArtifact =
        getBinArtifact("_objs/d/d.pic.o", getConfiguredTarget("//nomodule:d"));
    CppCompileAction dObjectAction = (CppCompileAction) getGeneratingAction(dObjectArtifact);
    assertThat(getNonSystemModuleMaps(dObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("c.cppmap", "//nomodule:c"),
            getGenfilesArtifact("d.cppmap", "//nomodule:d"));
    assertThat(getHeaderModules(dObjectAction.getCcCompilationContext().getTransitiveModules(true)))
        .containsExactly(getBinArtifact("_objs/b/b.pic.pcm", getConfiguredTarget("//module:b")));
  }

  private void writeSimpleCcLibrary() throws Exception {
    scratch.file("module/BUILD",
        "cc_library(",
        "    name = 'map',",
        "    srcs = ['a.cc', 'a.h'],",
        ")");
  }

  @Test
  public void testPicNotAvailableError() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.EMPTY_STATIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.EMPTY_COMPILE_ACTION_CONFIG,
            MockCcSupport.NO_LEGACY_FEATURES_FEATURE);
    useConfiguration("--cpu=k8");
    writeSimpleCcLibrary();
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//module:map");
    assertContainsEvent("PIC compilation is requested but the toolchain does not support it");
  }

  @Test
  public void testToolchainWithoutPicForNoPicCompilation() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "needsPic: false",
            MockCcSupport.EMPTY_COMPILE_ACTION_CONFIG,
            MockCcSupport.EMPTY_EXECUTABLE_ACTION_CONFIG,
            MockCcSupport.EMPTY_DYNAMIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.EMPTY_TRANSITIVE_DYNAMIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.EMPTY_STATIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.EMPTY_STRIP_ACTION_CONFIG,
            MockCcSupport.NO_LEGACY_FEATURES_FEATURE);
    useConfiguration();
    scratchConfiguredTarget("a", "a",
        "cc_binary(name='a', srcs=['a.cc'], deps=[':b'])",
        "cc_library(name='b', srcs=['b.cc'])");
  }

  @Test
  public void testNoCppModuleMap() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.EMPTY_COMPILE_ACTION_CONFIG,
            MockCcSupport.EMPTY_EXECUTABLE_ACTION_CONFIG,
            MockCcSupport.EMPTY_STATIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.EMPTY_DYNAMIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.EMPTY_TRANSITIVE_DYNAMIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.NO_LEGACY_FEATURES_FEATURE,
            MockCcSupport.PIC_FEATURE);
    useConfiguration();
    writeSimpleCcLibrary();
    assertNoCppModuleMapAction("//module:map");
  }

  @Test
  public void testCppModuleMap() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "feature { name: 'module_maps' }");
    useConfiguration();
    writeSimpleCcLibrary();
    CppModuleMapAction action = getCppModuleMapAction("//module:map");
    assertThat(ActionsTestUtil.baseArtifactNames(action.getDependencyArtifacts()))
        .contains("crosstool.cppmap");
    assertThat(artifactsToStrings(action.getPrivateHeaders()))
        .containsExactly("src module/a.h");
    assertThat(action.getPublicHeaders()).isEmpty();
  }

  /**
   * Historically, blaze hasn't added the pre-compiled libraries from srcs to the files to build.
   * This test ensures that we do not accidentally break that - we may do so intentionally.
   */
  @Test
  public void testFilesToBuildWithPrecompiledStaticLibrary() throws Exception {
    ConfiguredTarget hello = scratchConfiguredTarget("precompiled", "library",
        "cc_library(name = 'library', ",
        "           srcs = ['missing.a'])");
    assertThat(artifactsToStrings(getFilesToBuild(hello)))
        .doesNotContain("src precompiled/missing.a");
  }

  @Test
  public void testAllowDuplicateNonCompiledSources() throws Exception {
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "x",
            "x",
            "filegroup(name = 'xso', srcs = ['x.so'])",
            "cc_library(name = 'x', srcs = ['x.so', ':xso'])");
    assertThat(x).isNotNull();
  }

  @Test
  public void testDoNotCompileSourceFilesInHeaders() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PARSE_HEADERS_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers");
    ConfiguredTarget x =
        scratchConfiguredTarget("x", "x", "cc_library(name = 'x', hdrs = ['x.cc'])");
    assertThat(getGeneratingAction(getBinArtifact("_objs/x/.pic.o", x))).isNull();
  }

  @Test
  public void testProcessHeadersInDependencies() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PARSE_HEADERS_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "cc_library(name = 'x', deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupInfo.HIDDEN_TOP_LEVEL)))
        .isEqualTo("y.h.processed");
  }

  @Test
  public void testProcessHeadersInDependenciesOfBinaries() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PARSE_HEADERS_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "cc_binary(name = 'x', deps = [':y', ':z'])",
            "cc_library(name = 'y', hdrs = ['y.h'])",
            "cc_library(name = 'z', srcs = ['z.cc'])");
    String hiddenTopLevel =
        ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupInfo.HIDDEN_TOP_LEVEL));
    assertThat(hiddenTopLevel).contains("y.h.processed");
    assertThat(hiddenTopLevel).doesNotContain("z.pic.o");
  }

  @Test
  public void testDoNotProcessHeadersInDependencies() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PARSE_HEADERS_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "cc_library(name = 'x', deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupInfo.HIDDEN_TOP_LEVEL)))
        .isEmpty();
  }

  @Test
  public void testProcessHeadersInCompileOnlyMode() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PARSE_HEADERS_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget y =
        scratchConfiguredTarget(
            "foo",
            "y",
            "cc_library(name = 'x', deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(y, OutputGroupInfo.FILES_TO_COMPILE)))
        .isEqualTo("y.h.processed");
  }

  @Test
  public void testIncludePathOrder() throws Exception {
    scratch.file("foo/BUILD",
        "cc_library(",
        "    name = 'bar',",
        "    includes = ['bar'],",
        ")",
        "cc_library(",
        "    name = 'foo',",
        "    srcs = ['foo.cc'],",
        "    includes = ['foo'],",
        "    deps = [':bar'],",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//foo");
    CppCompileAction action = getCppCompileAction(target);
    String genfilesDir = getConfiguration(target).getGenfilesFragment().toString();
    String binDir = getConfiguration(target).getBinFragment().toString();
    // Local include paths come first.
    assertContainsSublist(
        action.getCompilerOptions(),
        ImmutableList.of(
            "-isystem",
            "foo/foo",
            "-isystem",
            genfilesDir + "/foo/foo",
            "-isystem",
            binDir + "/foo/foo",
            "-isystem",
            "foo/bar",
            "-isystem",
            genfilesDir + "/foo/bar",
            "-isystem",
            binDir + "/foo/bar"));
  }

  @Test
  public void testDefinesOrder() throws Exception {
    scratch.file("foo/BUILD",
        "cc_library(",
        "    name = 'bar',",
        "    defines = ['BAR'],",
        ")",
        "cc_library(",
        "    name = 'foo',",
        "    srcs = ['foo.cc'],",
        "    defines = ['FOO'],",
        "    deps = [':bar'],",
        ")");
    CppCompileAction action = getCppCompileAction("//foo");
    // Inherited defines come first.
    assertContainsSublist(action.getCompilerOptions(), ImmutableList.of("-DBAR", "-DFOO"));
  }

  // Regression test - setting "-shared" caused an exception when computing the link command.
  @Test
  public void testLinkOptsNotPassedToStaticLink() throws Exception {
    scratchConfiguredTarget("foo", "foo",
        "cc_library(",
        "    name = 'foo',",
        "    srcs = ['foo.cc'],",
        "    linkopts = ['-shared'],",
        ")");
  }

  private static final String COMPILATION_MODE_FEATURES = ""
      + "feature {"
      + "  name: 'dbg'"
      + "  flag_set {"
      + "    action: 'c++-compile'"
      + "    flag_group { flag: '-dbg' }"
      + "  }"
      + "}"
      + "feature {"
      + "  name: 'fastbuild'"
      + "  flag_set {"
      + "    action: 'c++-compile'"
      + "    flag_group { flag: '-fastbuild' }"
      + "  }"
      + "}"
      + "feature {"
      + "  name: 'opt'"
      + "  flag_set {"
      + "    action: 'c++-compile'"
      + "    flag_group { flag: '-opt' }"
      + "  }"
      + "}";

  private List<String> getCompilationModeFlags(String... flags) throws Exception {
    AnalysisMock.get().ccSupport().setupCrosstool(mockToolsConfig, COMPILATION_MODE_FEATURES);
    useConfiguration(flags);
    scratch.overwriteFile("mode/BUILD", "cc_library(name = 'a', srcs = ['a.cc'])");
    getConfiguredTarget("//mode:a");
    Artifact objectArtifact = getBinArtifact("_objs/a/a.pic.o", getConfiguredTarget("//mode:a"));
    CppCompileAction action = (CppCompileAction) getGeneratingAction(objectArtifact);
    return action.getCompilerOptions();
  }

  @Test
  public void testCompilationModeFeatures() throws Exception {
    List<String> flags;
    flags = getCompilationModeFlags("--cpu=k8");
    assertThat(flags).contains("-fastbuild");
    assertThat(flags).containsNoneOf("-opt", "-dbg");

    flags = getCompilationModeFlags("--cpu=k8", "--compilation_mode=fastbuild");
    assertThat(flags).contains("-fastbuild");
    assertThat(flags).containsNoneOf("-opt", "-dbg");

    flags = getCompilationModeFlags("--cpu=k8", "--compilation_mode=opt");
    assertThat(flags).contains("-opt");
    assertThat(flags).containsNoneOf("-fastbuild", "-dbg");

    flags = getCompilationModeFlags("--cpu=k8", "--compilation_mode=dbg");
    assertThat(flags).contains("-dbg");
    assertThat(flags).containsNoneOf("-fastbuild", "-opt");
  }

  private List<String> getHostAndTargetFlags(boolean useHost) throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HOST_AND_NONHOST_CONFIGURATION);
    scratch.overwriteFile("mode/BUILD", "cc_library(name = 'a', srcs = ['a.cc'])");
    useConfiguration("--cpu=k8");
    ConfiguredTarget target;
    String objectPath;
    if (useHost) {
      target = getHostConfiguredTarget("//mode:a");
      objectPath = "_objs/a/a.o";
    } else {
      target = getConfiguredTarget("//mode:a");
      objectPath = "_objs/a/a.pic.o";
    }
    Artifact objectArtifact = getBinArtifact(objectPath, target);
    CppCompileAction action = (CppCompileAction) getGeneratingAction(objectArtifact);
    assertThat(action).isNotNull();
    return action.getCompilerOptions();
  }

  @Test
  public void testHostAndNonHostFeatures() throws Exception {
    List<String> flags;

    flags = getHostAndTargetFlags(true);
    assertThat(flags).contains("-host");
    assertThat(flags).doesNotContain("-nonhost");

    flags = getHostAndTargetFlags(false);
    assertThat(flags).contains("-nonhost");
    assertThat(flags).doesNotContain("-host");
  }

  @Test
  public void testIncludePathsOutsideExecutionRoot() throws Exception {
    scratchRule(
        "root",
        "a",
        "cc_library(name='a', srcs=['a.cc'], copts=['-Id/../../somewhere'])");
    CppCompileAction compileAction = getCppCompileAction("//root:a");
    try {
      compileAction.verifyActionIncludePaths(compileAction.getSystemIncludeDirs());
    } catch (ActionExecutionException exception) {
      assertThat(exception)
          .hasMessageThat()
          .isEqualTo(
              "The include path '../somewhere' references a path outside of the execution root.");
    }
  }

  @Test
  public void testAbsoluteIncludePathsOutsideExecutionRoot() throws Exception {
    scratchRule(
        "root",
        "a",
        "cc_library(name='a', srcs=['a.cc'], copts=['-I/somewhere'])");
    CppCompileAction compileAction = getCppCompileAction("//root:a");
    try {
      compileAction.verifyActionIncludePaths(compileAction.getSystemIncludeDirs());
    } catch (ActionExecutionException exception) {
      assertThat(exception)
          .hasMessageThat()
          .isEqualTo(
              "The include path '/somewhere' references a path outside of the execution root.");
    }
  }

  @Test
  public void testSystemIncludePathsOutsideExecutionRoot() throws Exception {
    scratchRule(
        "root",
        "a",
        "cc_library(name='a', srcs=['a.cc'], copts=['-isystem../system'])");
    CppCompileAction compileAction = getCppCompileAction("//root:a");
    try {
      compileAction.verifyActionIncludePaths(compileAction.getSystemIncludeDirs());
    } catch (ActionExecutionException exception) {
      assertThat(exception)
          .hasMessageThat()
          .isEqualTo(
              "The include path '../system' references a path outside of the execution root.");
    }
  }

  @Test
  public void testAbsoluteSystemIncludePathsOutsideExecutionRoot() throws Exception {
    scratchRule(
        "root",
        "a",
        "cc_library(name='a', srcs=['a.cc'], copts=['-isystem/system'])");
    CppCompileAction compileAction = getCppCompileAction("//root:a");
    try {
      compileAction.verifyActionIncludePaths(compileAction.getSystemIncludeDirs());
    } catch (ActionExecutionException exception) {
      assertThat(exception)
          .hasMessageThat()
          .isEqualTo("The include path '/system' references a path outside of the execution root.");
    }
  }

  /**
   * Tests that configurable "srcs" doesn't crash because of orphaned implicit .so outputs.
   * (see {@link CcLibrary#appearsToHaveObjectFiles}).
   */
  @Test
  public void testConfigurableSrcs() throws Exception {
    scratch.file("foo/BUILD",
        "cc_library(",
        "    name = 'foo',",
        "    srcs = select({'//conditions:default': []}),",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//foo:foo");
    Artifact soOutput = getBinArtifact("libfoo.so", target);
    assertThat(getGeneratingAction(soOutput)).isInstanceOf(FailAction.class);
  }

  @Test
  public void alwaysAddStaticAndDynamicLibraryToFilesToBuildWhenBuilding() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "b", "cc_library(name = 'b', srcs = ['source.cc'])");

    assertThat(artifactsToStrings(getFilesToBuild(target)))
        .containsExactly("bin a/libb.a", "bin a/libb.ifso", "bin a/libb.so");
  }

  @Test
  public void addOnlyStaticLibraryToFilesToBuildWhenWrappingIffImplicitOutput() throws Exception {
    // This shared library has the same name as the archive generated by this rule, so it should
    // override said archive. However, said archive should still be put in files to build.
    ConfiguredTargetAndData target =
        scratchConfiguredTargetAndData("a", "b", "cc_library(name = 'b', srcs = ['libb.so'])");

    if (target.getTarget().getAssociatedRule().getImplicitOutputsFunction()
        != ImplicitOutputsFunction.NONE) {
      assertThat(artifactsToStrings(getFilesToBuild(target.getConfiguredTarget())))
          .containsExactly("bin a/libb.a");
    } else {
      assertThat(artifactsToStrings(getFilesToBuild(target.getConfiguredTarget()))).isEmpty();
    }
  }

  @Test
  public void addStaticLibraryToStaticSharedLinkParamsWhenBuilding() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "foo", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    Iterable<Artifact> libraries =
        LinkerInputs.toNonSolibArtifacts(
            target
                .get(CcInfo.PROVIDER)
                .getCcLinkingInfo()
                .getStaticModeParamsForDynamicLibrary()
                .getLibraries());
    assertThat(artifactsToStrings(libraries)).contains("bin a/libfoo.a");
  }

  @Test
  public void dontAddStaticLibraryToStaticSharedLinkParamsWhenWrappingSameLibraryIdentifier()
      throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "foo", "cc_library(name = 'foo', srcs = ['libfoo.so'])");

    Iterable<Artifact> libraries =
        LinkerInputs.toNonSolibArtifacts(
            target
                .get(CcInfo.PROVIDER)
                .getCcLinkingInfo()
                .getStaticModeParamsForDynamicLibrary()
                .getLibraries());
    assertThat(artifactsToStrings(libraries)).doesNotContain("bin a/libfoo.a");
    assertThat(artifactsToStrings(libraries)).contains("src a/libfoo.so");
  }

  @Test
  public void onlyAddOneWrappedLibraryWithSameLibraryIdentifierToLinkParams() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a", "foo", "cc_library(name = 'foo', srcs = ['libfoo.lo', 'libfoo.so'])");

    Iterable<Artifact> libraries =
        LinkerInputs.toNonSolibArtifacts(
            target
                .get(CcInfo.PROVIDER)
                .getCcLinkingInfo()
                .getStaticModeParamsForDynamicLibrary()
                .getLibraries());
    assertThat(artifactsToStrings(libraries)).doesNotContain("src a/libfoo.so");
    assertThat(artifactsToStrings(libraries)).contains("src a/libfoo.lo");
  }

  @Test
  public void testCcLinkParamsHasDynamicLibrariesForRuntime() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig, MockCcSupport.COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION);
    useConfiguration("--cpu=k8", "--features=copy_dynamic_libraries_to_binary");
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "foo", "cc_library(name = 'foo', srcs = ['foo.cc'])");
    Iterable<Artifact> libraries =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingInfo()
            .getDynamicModeParamsForDynamicLibrary()
            .getDynamicLibrariesForRuntime();
    assertThat(artifactsToStrings(libraries)).doesNotContain("bin a/libfoo.ifso");
    assertThat(artifactsToStrings(libraries)).contains("bin a/libfoo.so");
  }

  @Test
  public void testCcLinkParamsHasDynamicLibrariesForRuntimeWithoutCopyFeature() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "foo", "cc_library(name = 'foo', srcs = ['foo.cc'])");
    Iterable<Artifact> libraries =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingInfo()
            .getDynamicModeParamsForDynamicLibrary()
            .getDynamicLibrariesForRuntime();
    assertThat(artifactsToStrings(libraries)).doesNotContain("bin _solib_k8/liba_Slibfoo.ifso");
    assertThat(artifactsToStrings(libraries)).contains("bin _solib_k8/liba_Slibfoo.so");
  }

  @Test
  public void testCcLinkParamsDoNotHaveDynamicLibrariesForRuntime() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a", "foo", "cc_library(name = 'foo', srcs = ['foo.cc'], linkstatic=1)");
    Iterable<Artifact> libraries =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingInfo()
            .getDynamicModeParamsForDynamicLibrary()
            .getDynamicLibrariesForRuntime();
    assertThat(artifactsToStrings(libraries)).isEmpty();
  }

  @Test
  public void forbidBuildingAndWrappingSameLibraryIdentifier() throws Exception {
    useConfiguration("--cpu=k8");
    checkError(
        "a",
        "foo",
        "in cc_library rule //a:foo: Can't put libfoo.lo into the srcs of a cc_library with the "
            + "same name (foo) which also contains other code or objects to link; it shares a name "
            + "with libfoo.a, libfoo.ifso, libfoo.so (output compiled and linked from the "
            + "non-library sources of this rule), which could cause confusion",
        "cc_library(name = 'foo', srcs = ['foo.cc', 'libfoo.lo'])");
  }


  @Test
  public void testProcessedHeadersWithPicSharedLibsAndNoPicBinaries() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PARSE_HEADERS_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "-c", "opt");
    // Should not crash
    scratchConfiguredTarget("a", "a", "cc_library(name='a', hdrs=['a.h'])");
  }
}
