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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.extra.CppLinkInfo;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.FileType;
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

  @Before
  public final void createFiles() throws Exception {
    scratch.file("hello/BUILD",
                "cc_library(name = 'hello',",
                "           srcs = ['hello.cc'])",
                "cc_library(name = 'hello_static',",
                "           srcs = ['hello.cc'],",
                "           linkstatic = 1)");
    scratch.file("hello/hello.cc",
                "#include <stdio.h>",
                "int hello_world() { printf(\"Hello, world!\\n\"); }");
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
    CppModuleMap cppModuleMap = target.getProvider(CppCompilationContext.class).getCppModuleMap();
    return (CppModuleMapAction) getGeneratingAction(cppModuleMap.getArtifact());
  }

  private void assertNoCppModuleMapAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    assertThat(target.getProvider(CppCompilationContext.class).getCppModuleMap()).isNull();
  }

  @Test
  public void testMisconfiguredCrosstoolRaisesErrorWhenLinking() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.NO_LEGACY_FEATURES_FEATURE,
            MockCcSupport.INCOMPLETE_COMPILE_ACTION_CONFIG);
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
            MockCcSupport.INCOMPLETE_STATIC_LIBRARY_ACTION_CONFIG);
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
    assertThat(hello.getProvider(CcExecutionDynamicLibrariesProvider.class)
            .getExecutionDynamicLibraryArtifacts()).containsExactly(implSharedObjectLink);
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
    assertThat(hello.getProvider(CcExecutionDynamicLibrariesProvider.class)
            .getExecutionDynamicLibraryArtifacts()).containsExactly(implSharedObjectLink);
  }

  @Test
  public void testEmptyLinkopts() throws Exception {
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertThat(
            hello
                .get(CcLinkParamsProvider.CC_LINK_PARAMS)
                .getCcLinkParams(false, false)
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

    ExtraActionInfo.Builder builder = action.getExtraActionInfo();
    ExtraActionInfo info = builder.build();
    assertThat(info.getMnemonic()).isEqualTo("CppLink");

    CppLinkInfo cppLinkInfo = info.getExtension(CppLinkInfo.cppLinkInfo);
    assertThat(cppLinkInfo).isNotNull();

    Iterable<String> inputs = Artifact.asExecPaths(
        LinkerInputs.toLibraryArtifacts(action.getLinkCommandLine().getLinkerInputs()));
    assertThat(cppLinkInfo.getInputFileList()).containsExactlyElementsIn(inputs);
    assertThat(cppLinkInfo.getOutputFile())
        .isEqualTo(action.getPrimaryOutput().getExecPathString());
    assertThat(cppLinkInfo.hasInterfaceOutputFile()).isFalse();
    assertThat(cppLinkInfo.getLinkTargetType())
        .isEqualTo(action.getLinkCommandLine().getLinkTargetType().name());
    assertThat(cppLinkInfo.getLinkStaticness())
        .isEqualTo(action.getLinkCommandLine().getLinkStaticness().name());
    Iterable<String> linkstamps = Artifact.asExecPaths(
        action.getLinkCommandLine().getLinkstamps().values());
    assertThat(cppLinkInfo.getLinkStampList()).containsExactlyElementsIn(linkstamps);
    Iterable<String> buildInfoHeaderArtifacts = Artifact.asExecPaths(
        action.getLinkCommandLine().getBuildInfoHeaderArtifacts());
    assertThat(cppLinkInfo.getBuildInfoHeaderArtifactList())
        .containsExactlyElementsIn(buildInfoHeaderArtifacts);
    assertThat(cppLinkInfo.getLinkOptList())
        .containsExactlyElementsIn(action.getLinkCommandLine().getRawLinkArgv());
  }

  @Test
  public void testCppLinkActionExtraActionInfoWithSharedLibraries() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact sharedObject =
        FileType.filter(getFilesToBuild(hello), CppFileTypes.SHARED_LIBRARY).iterator().next();
    CppLinkAction action = (CppLinkAction) getGeneratingAction(sharedObject);

    ExtraActionInfo.Builder builder = action.getExtraActionInfo();
    ExtraActionInfo info = builder.build();
    assertThat(info.getMnemonic()).isEqualTo("CppLink");

    CppLinkInfo cppLinkInfo = info.getExtension(CppLinkInfo.cppLinkInfo);
    assertThat(cppLinkInfo).isNotNull();

    Iterable<String> inputs = Artifact.asExecPaths(
        LinkerInputs.toLibraryArtifacts(action.getLinkCommandLine().getLinkerInputs()));
    assertThat(cppLinkInfo.getInputFileList()).containsExactlyElementsIn(inputs);
    assertThat(cppLinkInfo.getOutputFile())
        .isEqualTo(action.getPrimaryOutput().getExecPathString());
    assertThat(cppLinkInfo.getLinkTargetType())
        .isEqualTo(action.getLinkCommandLine().getLinkTargetType().name());
    assertThat(cppLinkInfo.getLinkStaticness())
        .isEqualTo(action.getLinkCommandLine().getLinkStaticness().name());
    Iterable<String> linkstamps = Artifact.asExecPaths(
        action.getLinkCommandLine().getLinkstamps().values());
    assertThat(cppLinkInfo.getLinkStampList()).containsExactlyElementsIn(linkstamps);
    Iterable<String> buildInfoHeaderArtifacts = Artifact.asExecPaths(
        action.getLinkCommandLine().getBuildInfoHeaderArtifacts());
    assertThat(cppLinkInfo.getBuildInfoHeaderArtifactList())
        .containsExactlyElementsIn(buildInfoHeaderArtifacts);
    assertThat(cppLinkInfo.getLinkOptList())
        .containsExactlyElementsIn(action.getLinkCommandLine().getRawLinkArgv());
  }

  @Test
  public void testLinkActionCanConsumeArtifactExtensions() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.STATIC_LINK_TWEAKED_CONFIGURATION);
    useConfiguration("--features=" + Link.LinkTargetType.STATIC_LIBRARY.getActionName());
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact archive =
        FileType.filter(getFilesToBuild(hello), FileType.of(".tweaked.a")).iterator().next();

    CppLinkAction action = (CppLinkAction) getGeneratingAction(archive);

    assertThat(action.getArgv()).contains(archive.getExecPathString());
  }

  @Test
  public void testObjectFileNamesCanBeSpecifiedInToolchain() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig,
            "artifact_name_pattern {"
                + "   category_name: 'object_file'"
                + "   pattern: '%{output_name}.test.o'"
                + "}");

    useConfiguration();
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertThat(artifactByPath(getFilesToBuild(hello), ".a", ".test.o")).isNotNull();
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
  public void testArtifactSelectionErrorOnBadTemplateVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.STATIC_LINK_BAD_TEMPLATE_CONFIGURATION);
    useConfiguration("--features=" + Link.LinkTargetType.STATIC_LIBRARY.getActionName());
    try {
      getConfiguredTarget("//hello:hello");
      fail("Should fail");
    } catch (AssertionError e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Invalid toolchain configuration: Cannot find variable named 'bad_variable'");
    }
  }

  @Test
  public void testArtifactsToAlwaysBuild() throws Exception {
    useConfiguration("--cpu=k8");
    // ArtifactsToAlwaysBuild should apply both for static libraries.
    ConfiguredTarget helloStatic = getConfiguredTarget("//hello:hello_static");
    assertThat(
        artifactsToStrings(getOutputGroup(helloStatic, OutputGroupProvider.HIDDEN_TOP_LEVEL)))
        .containsExactly("bin hello/_objs/hello_static/hello/hello.pic.o");
    Artifact implSharedObject = getBinArtifact("libhello_static.so", helloStatic);
    assertThat(getFilesToBuild(helloStatic)).doesNotContain(implSharedObject);

    // And for shared libraries.
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertThat(
        artifactsToStrings(getOutputGroup(helloStatic, OutputGroupProvider.HIDDEN_TOP_LEVEL)))
        .containsExactly("bin hello/_objs/hello_static/hello/hello.pic.o");
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
    assertThat(artifactsToStrings(getOutputGroup(x, OutputGroupProvider.HIDDEN_TOP_LEVEL)))
        .containsExactly(
            "bin foo/_objs/x/foo/x.pic.o",
            "bin foo/_objs/y/foo/y.pic.o",
            "bin foo/_objs/z/foo/z.pic.o");
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
    assertThat(
            ActionsTestUtil.baseNamesOf(
                getOutputGroup(x, OutputGroupProvider.COMPILATION_PREREQUISITES)))
        .isEqualTo("y.h y.cppmap stl.cppmap crosstool.cppmap x.cppmap y.pic.pcm x.cc");
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
                x.getProvider(InstrumentedFilesProvider.class).getInstrumentationMetadataFiles()))
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
    getConfiguredTarget("//module:b");
    Artifact bModuleArtifact = getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b");
    CppCompileAction bModuleAction = (CppCompileAction) getGeneratingAction(bModuleArtifact);
    assertThat(bModuleAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/b.h"), getSourceArtifact("module/t.h"));
    assertThat(bModuleAction.getInputs()).contains(
        getGenfilesArtifactWithNoOwner("module/b.cppmap"));

    getConfiguredTarget("//module:a");
    Artifact aObjectArtifact = getBinArtifact("_objs/a/module/a.pic.o", "//module:a");
    CppCompileAction aObjectAction = (CppCompileAction) getGeneratingAction(aObjectArtifact);
    assertThat(aObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/a.cc"));
    assertThat(aObjectAction.getContext().getTransitiveModules(true)).contains(
        getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"));
    assertThat(aObjectAction.getInputs()).contains(
        getGenfilesArtifactWithNoOwner("module/b.cppmap"));
    assertNoEvents();
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
    getConfiguredTarget("//nomodule:f");
    assertThat(getGeneratingAction(getBinArtifact("_objs/f/nomodule/f.pic.pcm", "//nomodule:f")))
        .isNull();
    Artifact fObjectArtifact = getBinArtifact("_objs/f/nomodule/f.pic.o", "//nomodule:f");
    CppCompileAction fObjectAction = (CppCompileAction) getGeneratingAction(fObjectArtifact);
    // Only the module map of f itself itself and the direct dependencies are needed.
    assertThat(getNonSystemModuleMaps(fObjectAction.getInputs())).containsExactly(
        getGenfilesArtifact("f.cppmap", "//nomodule:f"),
        getGenfilesArtifact("e.cppmap", "//nomodule:e"));
    assertThat(getHeaderModules(fObjectAction.getInputs())).isEmpty();
    assertThat(fObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("nomodule/f.cc"));
    assertThat(getHeaderModuleFlags(fObjectAction.getCompilerOptions())).isEmpty();

    // The //nomodule:c target will get the header module for //module:b, which is a direct
    // dependency.
    getConfiguredTarget("//nomodule:c");
    assertThat(getGeneratingAction(getBinArtifact("_objs/c/nomodule/c.pic.pcm", "//nomodule:c")))
        .isNull();
    Artifact cObjectArtifact = getBinArtifact("_objs/c/nomodule/c.pic.o", "//nomodule:c");
    CppCompileAction cObjectAction = (CppCompileAction) getGeneratingAction(cObjectArtifact);
    assertThat(getNonSystemModuleMaps(cObjectAction.getInputs())).containsExactly(
        getGenfilesArtifact("b.cppmap", "//module:b"),
        getGenfilesArtifact("c.cppmap", "//nomodule:e"));
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
    assertThat(getGeneratingAction(getBinArtifact("_objs/d/nomodule/d.pic.pcm", "//nomodule:d")))
        .isNull();
    Artifact dObjectArtifact = getBinArtifact("_objs/d/nomodule/d.pic.o", "//nomodule:d");
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
    getConfiguredTarget("//module:j");
    Artifact jObjectArtifact = getBinArtifact("_objs/j/module/j.pic.o", "//module:j");
    CppCompileAction jObjectAction = (CppCompileAction) getGeneratingAction(jObjectArtifact);
    assertThat(getHeaderModules(jObjectAction.getContext().getTransitiveModules(true)))
        .containsExactly(
            getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"),
            getBinArtifact("_objs/g/module/g.pic.pcm", "//module:g"));
    assertThat(jObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/j.cc"));
    assertThat(jObjectAction.getMainIncludeScannerSource()).isEqualTo(
        getSourceArtifact("module/j.cc"));
    assertThat(getHeaderModules(jObjectAction.getContext().getTransitiveModules(true)))
        .containsExactly(
            getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"),
            getBinArtifact("_objs/g/module/g.pic.pcm", "//module:g"));
  }

  @Test
  public void testCompileUsingHeaderModulesTransitively() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration("--cpu=k8");
    setupPackagesForModuleTests( /*useHeaderModules=*/true);

    getConfiguredTarget("//nomodule:f");
    Artifact fObjectArtifact = getBinArtifact("_objs/f/nomodule/f.pic.o", "//nomodule:f");
    CppCompileAction fObjectAction = (CppCompileAction) getGeneratingAction(fObjectArtifact);
    // Only the module map of f itself itself and the direct dependencies are needed.
    assertThat(getNonSystemModuleMaps(fObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("f.cppmap", "//nomodule:f"),
            getGenfilesArtifact("e.cppmap", "//nomodule:e"));

    getConfiguredTarget("//nomodule:c");
    Artifact cObjectArtifact = getBinArtifact("_objs/c/nomodule/c.pic.o", "//nomodule:c");
    CppCompileAction cObjectAction = (CppCompileAction) getGeneratingAction(cObjectArtifact);
    assertThat(getNonSystemModuleMaps(cObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("b.cppmap", "//module:b"),
            getGenfilesArtifact("c.cppmap", "//nomodule:e"));
    assertThat(getHeaderModules(cObjectAction.getContext().getTransitiveModules(true)))
        .containsExactly(getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"));
    
    getConfiguredTarget("//nomodule:d");
    Artifact dObjectArtifact = getBinArtifact("_objs/d/nomodule/d.pic.o", "//nomodule:d");
    CppCompileAction dObjectAction = (CppCompileAction) getGeneratingAction(dObjectArtifact);
    assertThat(getNonSystemModuleMaps(dObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("c.cppmap", "//nomodule:c"),
            getGenfilesArtifact("d.cppmap", "//nomodule:d"));
    assertThat(getHeaderModules(dObjectAction.getContext().getTransitiveModules(true)))
        .containsExactly(getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"));
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
            MockCcSupport.INCOMPLETE_STATIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_COMPILE_ACTION_CONFIG,
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
            MockCcSupport.INCOMPLETE_COMPILE_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_EXECUTABLE_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_DYNAMIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_STATIC_LIBRARY_ACTION_CONFIG,
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
            MockCcSupport.INCOMPLETE_COMPILE_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_EXECUTABLE_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_STATIC_LIBRARY_ACTION_CONFIG,
            MockCcSupport.INCOMPLETE_DYNAMIC_LIBRARY_ACTION_CONFIG,
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
    assertThat(ActionsTestUtil.baseArtifactNames(action.getDependencyArtifacts())).containsExactly(
        "stl.cppmap",
        "crosstool.cppmap");
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
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_PROCESSING_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers");
    ConfiguredTarget x =
        scratchConfiguredTarget("x", "x", "cc_library(name = 'x', hdrs = ['x.cc'])");
    assertThat(getGeneratingAction(getBinArtifact("_objs/x/x/x.pic.o", x))).isNull();
  }

  @Test
  public void testProcessHeadersInDependencies() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_PROCESSING_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "cc_library(name = 'x', deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupProvider.HIDDEN_TOP_LEVEL)))
        .isEqualTo("y.h.processed");
  }

  @Test
  public void testProcessHeadersInDependenciesOfBinaries() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_PROCESSING_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "cc_binary(name = 'x', deps = [':y', ':z'])",
            "cc_library(name = 'y', hdrs = ['y.h'])",
            "cc_library(name = 'z', srcs = ['z.cc'])");
    String hiddenTopLevel =
        ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupProvider.HIDDEN_TOP_LEVEL));
    assertThat(hiddenTopLevel).contains("y.h.processed");
    assertThat(hiddenTopLevel).doesNotContain("z.pic.o");
  }

  @Test
  public void testDoNotProcessHeadersInDependencies() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_PROCESSING_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "cc_library(name = 'x', deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupProvider.HIDDEN_TOP_LEVEL)))
        .isEmpty();
  }

  @Test
  public void testProcessHeadersInCompileOnlyMode() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_PROCESSING_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget y =
        scratchConfiguredTarget(
            "foo",
            "y",
            "cc_library(name = 'x', deps = [':y'])",
            "cc_library(name = 'y', hdrs = ['y.h'])");
    assertThat(ActionsTestUtil.baseNamesOf(getOutputGroup(y, OutputGroupProvider.FILES_TO_COMPILE)))
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
    String genfilesDir = target.getConfiguration().getGenfilesFragment().toString();
    // Local include paths come first.
    assertContainsSublist(action.getCompilerOptions(), ImmutableList.of(
        "-isystem", "foo/foo", "-isystem", genfilesDir + "/foo/foo",
        "-isystem", "foo/bar", "-isystem", genfilesDir + "/foo/bar",
        "-isystem", TestConstants.GCC_INCLUDE_PATH));
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
    Artifact objectArtifact = getBinArtifact("_objs/a/mode/a.pic.o", "//mode:a");
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
      objectPath = "_objs/a/mode/a.o";
    } else {
      target = getConfiguredTarget("//mode:a");
      objectPath = "_objs/a/mode/a.pic.o";
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
    checkError(
        "root",
        "a",
        "The include path 'd/../../somewhere' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-Id/../../somewhere'])");
  }

  @Test
  public void testAbsoluteIncludePathsOutsideExecutionRoot() throws Exception {
    checkError(
        "root",
        "a",
        "The include path '/somewhere' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-I/somewhere'])");
  }

  @Test
  public void testSystemIncludePathsOutsideExecutionRoot() throws Exception {
    checkError(
        "root",
        "a",
        "The include path '../system' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-isystem../system'])");
  }

  @Test
  public void testAbsoluteSystemIncludePathsOutsideExecutionRoot() throws Exception {
    checkError(
        "root",
        "a",
        "The include path '/system' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-isystem/system'])");
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
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "b", "cc_library(name = 'b', srcs = ['libb.so'])");

    if (target.getTarget().getAssociatedRule().getImplicitOutputsFunction()
        != ImplicitOutputsFunction.NONE) {
      assertThat(artifactsToStrings(getFilesToBuild(target))).containsExactly("bin a/libb.a");
    } else {
      assertThat(artifactsToStrings(getFilesToBuild(target))).isEmpty();
    }
  }

  @Test
  public void addStaticLibraryToStaticSharedLinkParamsWhenBuilding() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "foo", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    Iterable<Artifact> libraries =
        LinkerInputs.toNonSolibArtifacts(
            target
                .get(CcLinkParamsProvider.CC_LINK_PARAMS)
                .getCcLinkParams(true, true)
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
                .get(CcLinkParamsProvider.CC_LINK_PARAMS)
                .getCcLinkParams(true, true)
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
                .get(CcLinkParamsProvider.CC_LINK_PARAMS)
                .getCcLinkParams(true, true)
                .getLibraries());
    assertThat(artifactsToStrings(libraries)).doesNotContain("src a/libfoo.so");
    assertThat(artifactsToStrings(libraries)).contains("src a/libfoo.lo");
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
        .setupCrosstool(mockToolsConfig,
            MockCcSupport.HEADER_PROCESSING_FEATURE_CONFIGURATION);
    useConfiguration("--features=parse_headers", "-c", "opt");
    // Should not crash
    scratchConfiguredTarget("a", "a", "cc_library(name='a', hdrs=['a.h'])");
  }

  @Test
  public void testStlWithAlias() throws Exception {
    scratch.file("a/BUILD",
        "cc_library(name='a')",
        "alias(name='stl', actual=':realstl')",
        "cc_library(name='realstl')");

    useConfiguration("--experimental_stl=//a:stl");
    getConfiguredTarget("//a:a");
  }
}
