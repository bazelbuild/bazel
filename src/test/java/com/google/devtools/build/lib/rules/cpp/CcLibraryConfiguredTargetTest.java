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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
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
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.List;

/**
 * "White-box" unit test of cc_library rule.
 */
@RunWith(JUnit4.class)
public class CcLibraryConfiguredTargetTest extends BuildViewTestCase {
  private static final PathFragment STL_CPPMAP = new PathFragment("stl.cppmap");
  private static final PathFragment CROSSTOOL_CPPMAP = new PathFragment("crosstool.cppmap");


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
    assertNull(target.getProvider(CppCompilationContext.class).getCppModuleMap());
  }


  @Test
  public void testFilesToBuild() throws Exception {
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact archive = getBinArtifact("libhello.a", hello);
    Artifact implSharedObject = getBinArtifact("libhello.so", hello);
    Artifact implInterfaceSharedObject = getBinArtifact("libhello.ifso", hello);
    Artifact implSharedObjectLink = getSharedArtifact("_solib_k8/libhello_Slibhello.so", hello);
    Artifact implInterfaceSharedObjectLink =
        getSharedArtifact("_solib_k8/libhello_Slibhello.ifso", hello);
    assertThat(getFilesToBuild(hello)).containsExactly(archive, implSharedObject,
        implInterfaceSharedObject);
    assertThat(LinkerInputs.toLibraryArtifacts(
        hello.getProvider(CcNativeLibraryProvider.class).getTransitiveCcNativeLibraries()))
        .containsExactly(implInterfaceSharedObjectLink);
    assertThat(hello.getProvider(CcExecutionDynamicLibrariesProvider.class)
            .getExecutionDynamicLibraryArtifacts()).containsExactly(implSharedObjectLink);
  }

  @Test
  public void testFilesToBuildWithInterfaceSharedObjects() throws Exception {
    useConfiguration("--interface_shared_objects");
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact archive = getBinArtifact("libhello.a", hello);
    Artifact sharedObject = getBinArtifact("libhello.ifso", hello);
    Artifact implSharedObject = getBinArtifact("libhello.so", hello);
    Artifact sharedObjectLink = getSharedArtifact("_solib_k8/libhello_Slibhello.ifso", hello);
    Artifact implSharedObjectLink = getSharedArtifact("_solib_k8/libhello_Slibhello.so", hello);
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
    assertTrue(hello.getProvider(CcLinkParamsProvider.class)
        .getCcLinkParams(false, false).getLinkopts().isEmpty());
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
    assertEquals("CppLink", info.getMnemonic());

    CppLinkInfo cppLinkInfo = info.getExtension(CppLinkInfo.cppLinkInfo);
    assertNotNull(cppLinkInfo);

    Iterable<String> inputs = Artifact.asExecPaths(
        LinkerInputs.toLibraryArtifacts(action.getLinkCommandLine().getLinkerInputs()));
    assertThat(cppLinkInfo.getInputFileList()).containsExactlyElementsIn(inputs);
    assertEquals(action.getPrimaryOutput().getExecPathString(), cppLinkInfo.getOutputFile());
    assertFalse(cppLinkInfo.hasInterfaceOutputFile());
    assertEquals(action.getLinkCommandLine().getLinkTargetType().name(),
        cppLinkInfo.getLinkTargetType());
    assertEquals(action.getLinkCommandLine().getLinkStaticness().name(),
        cppLinkInfo.getLinkStaticness());
    Iterable<String> linkstamps = Artifact.asExecPaths(
        action.getLinkCommandLine().getLinkstamps().values());
    assertThat(cppLinkInfo.getLinkStampList()).containsExactlyElementsIn(linkstamps);
    Iterable<String> buildInfoHeaderArtifacts = Artifact.asExecPaths(
        action.getLinkCommandLine().getBuildInfoHeaderArtifacts());
    assertThat(cppLinkInfo.getBuildInfoHeaderArtifactList())
        .containsExactlyElementsIn(buildInfoHeaderArtifacts);
    assertThat(cppLinkInfo.getLinkOptList())
        .containsExactlyElementsIn(action.getLinkCommandLine().getLinkopts());
  }

  @Test
  public void testCppLinkActionExtraActionInfoWithSharedLibraries() throws Exception {
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    Artifact sharedObject  =
        FileType.filter(getFilesToBuild(hello), CppFileTypes.SHARED_LIBRARY).iterator().next();
    CppLinkAction action = (CppLinkAction) getGeneratingAction(sharedObject);

    ExtraActionInfo.Builder builder = action.getExtraActionInfo();
    ExtraActionInfo info = builder.build();
    assertEquals("CppLink", info.getMnemonic());

    CppLinkInfo cppLinkInfo = info.getExtension(CppLinkInfo.cppLinkInfo);
    assertNotNull(cppLinkInfo);

    Iterable<String> inputs = Artifact.asExecPaths(
        LinkerInputs.toLibraryArtifacts(action.getLinkCommandLine().getLinkerInputs()));
    assertThat(cppLinkInfo.getInputFileList()).containsExactlyElementsIn(inputs);
    assertEquals(action.getPrimaryOutput().getExecPathString(), cppLinkInfo.getOutputFile());
    Artifact interfaceOutput = action.getLinkCommandLine().getInterfaceOutput();
    assertEquals(interfaceOutput.getExecPathString(), cppLinkInfo.getInterfaceOutputFile());
    assertEquals(action.getLinkCommandLine().getLinkTargetType().name(),
        cppLinkInfo.getLinkTargetType());
    assertEquals(action.getLinkCommandLine().getLinkStaticness().name(),
        cppLinkInfo.getLinkStaticness());
    Iterable<String> linkstamps = Artifact.asExecPaths(
        action.getLinkCommandLine().getLinkstamps().values());
    assertThat(cppLinkInfo.getLinkStampList()).containsExactlyElementsIn(linkstamps);
    Iterable<String> buildInfoHeaderArtifacts = Artifact.asExecPaths(
        action.getLinkCommandLine().getBuildInfoHeaderArtifacts());
    assertThat(cppLinkInfo.getBuildInfoHeaderArtifactList())
        .containsExactlyElementsIn(buildInfoHeaderArtifacts);
    assertThat(cppLinkInfo.getLinkOptList())
        .containsExactlyElementsIn(action.getLinkCommandLine().getLinkopts());
  }

  @Test
  public void testArtifactsToAlwaysBuild() throws Exception {
    // ArtifactsToAlwaysBuild should apply both for static libraries.
    ConfiguredTarget helloStatic = getConfiguredTarget("//hello:hello_static");
    assertEquals(ImmutableSet.of("bin hello/_objs/hello_static/hello/hello.pic.o"),
        artifactsToStrings(getOutputGroup(helloStatic, OutputGroupProvider.HIDDEN_TOP_LEVEL)));
    Artifact implSharedObject = getBinArtifact("libhello_static.so", helloStatic);
    assertThat(getFilesToBuild(helloStatic)).doesNotContain(implSharedObject);

    // And for shared libraries.
    ConfiguredTarget hello = getConfiguredTarget("//hello:hello");
    assertEquals(ImmutableSet.of("bin hello/_objs/hello_static/hello/hello.pic.o"),
        artifactsToStrings(getOutputGroup(helloStatic, OutputGroupProvider.HIDDEN_TOP_LEVEL)));
    implSharedObject = getBinArtifact("libhello.so", hello);
    assertThat(getFilesToBuild(hello)).contains(implSharedObject);
  }

  @Test
  public void testTransitiveArtifactsToAlwaysBuildStatic() throws Exception {
    ConfiguredTarget x = scratchConfiguredTarget(
        "foo", "x",
        "cc_library(name = 'x', srcs = ['x.cc'], deps = [':y'], linkstatic = 1)",
        "cc_library(name = 'y', srcs = ['y.cc'], deps = [':z'])",
        "cc_library(name = 'z', srcs = ['z.cc'])");
    assertEquals(
        ImmutableSet.of(
          "bin foo/_objs/x/foo/x.pic.o",
          "bin foo/_objs/y/foo/y.pic.o",
          "bin foo/_objs/z/foo/z.pic.o"),
        artifactsToStrings(getOutputGroup(x, OutputGroupProvider.HIDDEN_TOP_LEVEL)));
  }

  @Test
  public void testBuildHeaderModulesAsPrerequisites() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration();
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
        .isEqualTo("y.h y.pic.pcm y.cppmap stl.cppmap crosstool.cppmap x.cppmap x.cc");
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
        names.add(new PathFragment(flag).getBaseName());
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
            "" + "feature { name: 'header_modules' }" + "feature { name: 'module_maps' }");
    useConfiguration();
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
        getSourceArtifact("module/a.cc"),
        getSourceArtifact("module/b.h"),
        getSourceArtifact("module/t.h"));
    assertThat(aObjectAction.getInputs()).contains(
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
    useConfiguration();
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
    assertThat(getHeaderModules(cObjectAction.getInputs())).containsExactly(
        getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"));
    // All headers of transitive dependencies that are built as modules are needed as entry points
    // for include scanning.
    assertThat(cObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("nomodule/c.cc"),
        getSourceArtifact("module/b.h"));
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
    assertThat(getHeaderModules(dObjectAction.getInputs())).containsExactly(
        getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"));
    // All headers of transitive dependencies that are built as modules are needed as entry points
    // for include scanning.
    assertThat(dObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/b.h"),
        getSourceArtifact("nomodule/d.cc"));
    assertThat(getHeaderModuleFlags(dObjectAction.getCompilerOptions())).isEmpty();

    // The //module:j target depends on //module:g via //nomodule:h and on //module:b via
    // both //module:g and //nomodule:c.
    getConfiguredTarget("//module:j");
    Artifact jObjectArtifact = getBinArtifact("_objs/j/module/j.pic.o", "//module:j");
    CppCompileAction jObjectAction = (CppCompileAction) getGeneratingAction(jObjectArtifact);
    assertThat(getHeaderModules(jObjectAction.getInputs())).containsExactly(
        getBinArtifact("_objs/b/module/b.pic.pcm", "//module:b"),
        getBinArtifact("_objs/g/module/g.pic.pcm", "//module:g"));
    assertThat(jObjectAction.getIncludeScannerSources()).containsExactly(
        getSourceArtifact("module/j.cc"),
        getSourceArtifact("module/b.h"),
        getSourceArtifact("module/g.h"));
    assertThat(jObjectAction.getMainIncludeScannerSource()).isEqualTo(
        getSourceArtifact("module/j.cc"));
    assertThat(getHeaderModuleFlags(jObjectAction.getCompilerOptions()))
        .containsExactly("b.pic.pcm", "g.pic.pcm");
  }
  
  @Test
  public void testCompileUsingHeaderModulesTransitivelyWithTranstiveModuleMaps() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION
                + "feature { name: 'transitive_module_maps' }");
    useConfiguration("--features=transitive_module_maps");
    setupPackagesForModuleTests(/*useHeaderModules=*/true);
    
    getConfiguredTarget("//nomodule:f");
    Artifact fObjectArtifact = getBinArtifact("_objs/f/nomodule/f.pic.o", "//nomodule:f");
    CppCompileAction fObjectAction = (CppCompileAction) getGeneratingAction(fObjectArtifact);
    // Only the module map of f itself itself and the direct dependencies are needed.
    assertThat(getNonSystemModuleMaps(fObjectAction.getInputs())).containsExactly(
        getGenfilesArtifact("y.cppmap", "//nomodule:y"),
        getGenfilesArtifact("z.cppmap", "//nomodule:z"),
        getGenfilesArtifact("a.cppmap", "//nomodule:a"),
        getGenfilesArtifact("f.cppmap", "//nomodule:f"),
        getGenfilesArtifact("e.cppmap", "//nomodule:e"));

    getConfiguredTarget("//nomodule:c");
    Artifact cObjectArtifact = getBinArtifact("_objs/c/nomodule/c.pic.o", "//nomodule:c");
    CppCompileAction cObjectAction = (CppCompileAction) getGeneratingAction(cObjectArtifact);
    assertThat(getNonSystemModuleMaps(cObjectAction.getInputs())).containsExactly(
        getGenfilesArtifact("y.cppmap", "//nomodule:y"),
        getGenfilesArtifact("z.cppmap", "//nomodule:z"),
        getGenfilesArtifact("a.cppmap", "//nomodule:a"),
        getGenfilesArtifact("b.cppmap", "//module:b"),
        getGenfilesArtifact("c.cppmap", "//nomodule:e"));
    assertThat(getHeaderModuleFlags(cObjectAction.getCompilerOptions()))
        .containsExactly("b.pic.pcm");
     
    getConfiguredTarget("//nomodule:d");
    Artifact dObjectArtifact = getBinArtifact("_objs/d/nomodule/d.pic.o", "//nomodule:d");
    CppCompileAction dObjectAction = (CppCompileAction) getGeneratingAction(dObjectArtifact);
    assertThat(getNonSystemModuleMaps(dObjectAction.getInputs())).containsExactly(
        getGenfilesArtifact("y.cppmap", "//nomodule:y"),
        getGenfilesArtifact("z.cppmap", "//nomodule:z"),
        getGenfilesArtifact("a.cppmap", "//nomodule:a"),
        getGenfilesArtifact("b.cppmap", "//module:b"),
        getGenfilesArtifact("c.cppmap", "//nomodule:c"),
        getGenfilesArtifact("d.cppmap", "//nomodule:d"));
    assertThat(getHeaderModuleFlags(dObjectAction.getCompilerOptions()))
        .containsExactly("b.pic.pcm");
  }

  @Test
  public void testCompileUsingHeaderModulesTransitively() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION);
    useConfiguration();
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
    assertThat(getHeaderModuleFlags(cObjectAction.getCompilerOptions()))
        .containsExactly("b.pic.pcm");

    getConfiguredTarget("//nomodule:d");
    Artifact dObjectArtifact = getBinArtifact("_objs/d/nomodule/d.pic.o", "//nomodule:d");
    CppCompileAction dObjectAction = (CppCompileAction) getGeneratingAction(dObjectArtifact);
    assertThat(getNonSystemModuleMaps(dObjectAction.getInputs()))
        .containsExactly(
            getGenfilesArtifact("c.cppmap", "//nomodule:c"),
            getGenfilesArtifact("d.cppmap", "//nomodule:d"));
    assertThat(getHeaderModuleFlags(dObjectAction.getCompilerOptions()))
        .containsExactly("b.pic.pcm");
  }

  @Test
  public void testTopLevelHeaderModules() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.HEADER_MODULES_FEATURE_CONFIGURATION
                + "feature { name: 'header_module_includes_dependencies' }");
    useConfiguration();
    setupPackagesForModuleTests(/*useHeaderModules=*/false);
    getConfiguredTarget("//module:j");
    Artifact jObjectArtifact = getBinArtifact("_objs/j/module/j.pic.o", "//module:j");
    CppCompileAction jObjectAction = (CppCompileAction) getGeneratingAction(jObjectArtifact);
    assertThat(getHeaderModuleFlags(jObjectAction.getCompilerOptions()))
        .containsExactly("g.pic.pcm");
  }

  private void writeSimpleCcLibrary() throws Exception {
    scratch.file("module/BUILD",
        "cc_library(",
        "    name = 'map',",
        "    srcs = ['a.cc', 'a.h'],",
        ")");
  }

  @Test
  public void testNoCppModuleMap() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "feature { name: 'no_legacy_features' }");
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
    assertEquals(ImmutableSet.of("src module/a.h"),
        artifactsToStrings(action.getPrivateHeaders()));
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
    flags = getCompilationModeFlags();
    assertThat(flags).contains("-fastbuild");
    assertThat(flags).containsNoneOf("-opt", "-dbg");
    
    flags = getCompilationModeFlags("-c", "fastbuild");
    assertThat(flags).contains("-fastbuild");
    assertThat(flags).containsNoneOf("-opt", "-dbg");
    
    flags = getCompilationModeFlags("-c", "opt");
    assertThat(flags).contains("-opt");
    assertThat(flags).containsNoneOf("-fastbuild", "-dbg");
    
    flags = getCompilationModeFlags("-c", "dbg");
    assertThat(flags).contains("-dbg");
    assertThat(flags).containsNoneOf("-fastbuild", "-opt");
  }

  @Test
  public void testIncludePathsOutsideExecutionRoot() throws Exception {
    checkError("root", "a",
        "The include path 'd/../../somewhere' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-Id/../../somewhere'])");
  }
  
  @Test
  public void testAbsoluteIncludePathsOutsideExecutionRoot() throws Exception {
    checkError("root", "a",
        "The include path '/somewhere' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-I/somewhere'])");
  }

  @Test
  public void testSystemIncludePathsOutsideExecutionRoot() throws Exception {  
    checkError("root", "a",
        "The include path '../system' references a path outside of the execution root.",
        "cc_library(name='a', srcs=['a.cc'], copts=['-isystem../system'])");
  }

  @Test
  public void testAbsoluteSystemIncludePathsOutsideExecutionRoot() throws Exception {  
    checkError("root", "a",
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
}
