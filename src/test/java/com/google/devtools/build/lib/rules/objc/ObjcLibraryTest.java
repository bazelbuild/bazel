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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;
import static com.google.devtools.build.lib.rules.objc.CompilationSupport.ABSOLUTE_INCLUDES_PATH_FORMAT;
import static com.google.devtools.build.lib.rules.objc.CompilationSupport.FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.CC_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_library. */
@RunWith(JUnit4.class)
public class ObjcLibraryTest extends ObjcRuleTestCase {

  static final RuleType RULE_TYPE = new OnlyNeedsSourcesRuleType("objc_library");
  private static final String WRAPPED_CLANG = "wrapped_clang";

  /**
   * Middleman artifact arising from //tools/osx/crosstool:link, containing tools that should be
   * inputs to link actions.
   */
  private static final String CROSSTOOL_LINK_MIDDLEMAN = "tools_Sosx_Scrosstool_Clink";

  /** Creates an {@code objc_library} target writer. */
  @Override
  protected ScratchAttributeWriter createLibraryTargetWriter(String labelString) {
    return ScratchAttributeWriter.fromLabelString(this, "objc_library", labelString);
  }

  @Test
  public void testFilesToBuild() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:One")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .write();

    Iterable<Artifact> files = getFilesToBuild(target);
    assertThat(Artifact.toRootRelativePaths(files)).containsExactly("objc/libOne.a");
  }

  @Test
  public void testCompilesSources() throws Exception {
    createLibraryTargetWriter("//objc/lib1")
        .setAndCreateFiles("srcs", "a.m")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    createLibraryTargetWriter("//objc/lib2")
        .setAndCreateFiles("srcs", "a.m")
        .setAndCreateFiles("hdrs", "hdr.h")
        .setList("deps", "//objc/lib1")
        .write();

    createLibraryTargetWriter("//objc:x")
        .setAndCreateFiles("srcs", "a.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .setList("deps", "//objc/lib2:lib2")
        .write();

    CppCompileAction compileA = (CppCompileAction) compileAction("//objc:x", "a.o");

    assertThat(Artifact.toRootRelativePaths(compileA.getPossibleInputsForTesting()))
        .containsAllOf("objc/a.m", "objc/hdr.h", "objc/private.h");
    assertThat(Artifact.toRootRelativePaths(compileA.getOutputs()))
        .containsExactly("objc/_objs/x/objc/a.o", "objc/_objs/x/objc/a.d");
  }

  @Test
  public void testObjcPlusPlusCompile() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386",
        "--ios_minimum_os=9.10.11");
    createLibraryTargetWriter("//objc:lib")
        .setList("srcs", "a.mm")
        .write();
    CommandAction compileAction = compileAction("//objc:lib", "a.o");
    assertThat(compileAction.getArguments())
        .containsAllOf("-stdlib=libc++", "-std=gnu++11", "-mios-simulator-version-min=9.10.11");
  }

  @Test
  public void testObjcPlusPlusCompileDarwin() throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL,
        "--cpu=darwin_x86_64",
        "--macos_minimum_os=9.10.11",
        // TODO(b/36126423): Darwin should imply macos, so the
        // following line should not be necessary.
        "--apple_platform_type=macos");
    createLibraryTargetWriter("//objc:lib")
        .setList("srcs", "a.mm")
        .write();
    CommandAction compileAction = compileAction("//objc:lib", "a.o");
    assertThat(compileAction.getArguments())
        .containsAllOf("-stdlib=libc++", "-std=gnu++11", "-mmacosx-version-min=9.10.11");
  }

  @Test
  public void testCompilationModeDbg() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386",
        "--compilation_mode=dbg");
    scratch.file("objc/a.m");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch,
            "objc",
            "lib",
            "srcs",
            "['a.m']"));

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).contains("--DBG_ONLY_FLAG");
    assertThat(compileActionA.getArguments()).doesNotContain("--FASTBUILD_ONLY_FLAG");
    assertThat(compileActionA.getArguments()).doesNotContain("--OPT_ONLY_FLAG");
  }

  @Test
  public void testCompilationModeFastbuild() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386",
        "--compilation_mode=fastbuild");
    scratch.file("objc/a.m");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch,
            "objc",
            "lib",
            "srcs",
            "['a.m']"));

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("--DBG_ONLY_FLAG");
    assertThat(compileActionA.getArguments()).contains("--FASTBUILD_ONLY_FLAG");
    assertThat(compileActionA.getArguments()).doesNotContain("--OPT_ONLY_FLAG");
  }

  @Test
  public void testCompilationModeOpt() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386",
        "--compilation_mode=opt");
    scratch.file("objc/a.m");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch,
            "objc",
            "lib",
            "srcs",
            "['a.m']"));

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("--DBG_ONLY_FLAG");
    assertThat(compileActionA.getArguments()).doesNotContain("--FASTBUILD_ONLY_FLAG");
    assertThat(compileActionA.getArguments()).contains("--OPT_ONLY_FLAG");
  }

  @Test
  public void testCreate_runfilesWithSourcesOnly() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:One")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .write();
    RunfilesProvider provider = target.getProvider(RunfilesProvider.class);
    assertThat(baseArtifactNames(provider.getDefaultRunfiles().getArtifacts())).isEmpty();
    assertThat(Artifact.toRootRelativePaths(provider.getDataRunfiles().getArtifacts()))
        .containsExactly("objc/libOne.a");
  }

  @Test
  public void testCreate_noErrorForEmptySourcesButHasDependency() throws Exception {
    createLibraryTargetWriter("//baselib:baselib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .write();
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("hdrs", "a.h")
        .setList("deps", "//baselib:baselib")
        .write();
    ObjcProvider provider = providerForTarget("//lib:lib");
    assertThat(provider.get(LIBRARY))
        .containsExactlyElementsIn(archiveAction("//baselib:baselib").getOutputs());
  }

  @Test
  public void testCreate_errorForEmptyFilegroupSources() throws Exception {
    checkError(
        "x",
        "x",
        "does not produce any objc_library srcs files (expected " + SRCS_TYPE + ")",
        "filegroup(name = 'fg', srcs = [])",
        "objc_library(name = 'x', srcs = ['fg'])");
  }

  @Test
  public void testCreate_srcsContainingHeaders() throws Exception {
    scratch.file("x/a.m", "dummy source file");
    scratch.file("x/a.h", "dummy header file");
    scratch.file("x/BUILD", "objc_library(name = 'Target', srcs = ['a.m', 'a.h'])");
    assertThat(view.hasErrors(getConfiguredTarget("//x:Target"))).isFalse();
  }

  @Test
  public void testCreate_warningForOverlappingSrcsAndHdrs() throws Exception {
    scratch.file("/x/a.h", "dummy header file");
    checkWarning(
        "x",
        "x",
        String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, "x/a.h"),
        "objc_library(name = 'x', srcs = ['a.h'], hdrs = ['a.h'])");
  }

  @Test
  public void testCreate_headerAndCompiledSourceWithSameName() throws Exception {
    scratch.file("objc/BUILD", "objc_library(name = 'Target', srcs = ['a.m'], hdrs = ['a.h'])");
    assertThat(view.hasErrors(getConfiguredTarget("//objc:Target"))).isFalse();
  }

  @Test
  public void testCreate_errorForCcInNonArcSources() throws Exception {
    scratch.file("x/cc.cc");
    checkError(
        "x",
        "x",
        "'//x:cc.cc' does not produce any objc_library non_arc_srcs files (expected "
            + NON_ARC_SRCS_TYPE
            + ")",
        "objc_library(name = 'x', non_arc_srcs = ['cc.cc'])");
  }

  @Test
  public void testFileInSrcsAndNonArcSources() throws Exception {
    checkError(
        "x",
        "x",
        String.format(CompilationSupport.FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT, "x/foo.m"),
        "objc_library(name = 'x', srcs = ['foo.m'], non_arc_srcs = ['foo.m'])");
  }

  @Test
  public void testCreate_headerContainingDotMAndDotCFiles() throws Exception {
    scratch.file("x/a.m", "dummy source file");
    scratch.file("x/a.h", "dummy header file");
    scratch.file("x/b.m", "dummy source file");
    scratch.file("x/a.c", "dummy source file");
    scratch.file(
        "x/BUILD", "objc_library(name = 'Target', srcs = ['a.m'], hdrs = ['a.h', 'b.m', 'a.c'])");
    assertThat(view.hasErrors(getConfiguredTarget("//x:Target"))).isFalse();
  }

  @Test
  public void testProvidesObjcHeadersWithDotMFiles() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:lib")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .setAndCreateFiles("hdrs", "a.h", "b.h", "f.m")
            .write();
    ConfiguredTarget depender =
        createLibraryTargetWriter("//objc2:lib")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .setAndCreateFiles("hdrs", "d.h", "e.m")
            .setList("deps", "//objc:lib")
            .write();
    assertThat(getArifactPaths(target, HEADER))
        .containsExactly("objc/a.h", "objc/b.h", "objc/f.m");
    assertThat(getArifactPaths(depender, HEADER))
        .containsExactly("objc/a.h", "objc/b.h", "objc/f.m", "objc2/d.h", "objc2/e.m");
  }

  @Test
  public void testNonPropagatedDepsProvider() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:lib")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .setAndCreateFiles("hdrs", "a.h", "b.h")
            .write();
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("non_propagated_deps", "//objc:lib")
        .write();
    ConfiguredTarget transitiveDepender =
        createLibraryTargetWriter("//objc3:lib")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .setAndCreateFiles("hdrs", "e.h", "f.h")
            .setList("non_propagated_deps", "//objc2:lib")
            .write();

    assertThat(getArifactPaths(target, HEADER))
        .containsExactly("objc/a.h", "objc/b.h");
    assertThat(getArifactPaths(transitiveDepender, HEADER))
        .containsExactly("objc2/c.h", "objc2/d.h", "objc3/e.h", "objc3/f.h");
  }

  @Test
  public void testMultiPlatformLibrary() throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64,armv7,arm64", "--ios_cpu=armv7");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h")
        .write();

    assertThat(view.hasErrors(getConfiguredTarget("//objc:lib"))).isFalse();
  }

  @Test
  public void testNonPropagatedDepsDiamond() throws Exception {
    // Non-propagated.
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h")
        .write();
    // Conflicts with non-propagated.
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h")
        .write();

    createLibraryTargetWriter("//objc3:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "b.h")
        .setList("non_propagated_deps", "//objc:lib")
        .write();

    createLibraryTargetWriter("//objc4:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setList("deps", "//objc2:lib", "//objc3:lib")
        .write();

    CommandAction action = compileAction("//objc4:lib", "a.o");
    assertThat(Artifact.toRootRelativePaths(action.getPossibleInputsForTesting()))
        .containsAllOf("objc2/a.h", "objc3/b.h", "objc4/c.h", "objc4/a.m", "objc4/private.h");
  }

  @Test
  public void testCompilationActions_simulator() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386");

    scratch.file("objc/a.m");
    scratch.file("objc/non_arc.m");
    scratch.file("objc/private.h");
    scratch.file("objc/c.h");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch,
            "objc",
            "lib",
            "srcs",
            "['a.m', 'private.h']",
            "hdrs",
            "['c.h']",
            "non_arc_srcs",
            "['non_arc.m']"));

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    CommandAction compileActionNonArc = compileAction("//objc:lib", "non_arc.o");

    assertRequiresDarwin(compileActionA);
    assertThat(compileActionA.getArguments())
        .contains("tools/osx/crosstool/iossim/" + WRAPPED_CLANG);
    assertThat(compileActionA.getArguments())
        .containsAllOf("-isysroot", AppleToolchain.sdkDir()).inOrder();
    assertThat(Collections.frequency(compileActionA.getArguments(),
        "-F" + AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")).isEqualTo(1);
    assertThat(
            Collections.frequency(
                compileActionA.getArguments(), "-F" + frameworkDir(ApplePlatform.IOS_SIMULATOR)))
        .isEqualTo(1);
    assertThat(compileActionA.getArguments())
        .containsAllIn(AppleToolchain.DEFAULT_WARNINGS.values());
    assertThat(compileActionA.getArguments())
        .containsAllIn(CompilationSupport.DEFAULT_COMPILER_FLAGS);
    assertThat(compileActionA.getArguments())
        .containsAllIn(CompilationSupport.SIMULATOR_COMPILE_FLAGS);
    assertThat(compileActionA.getArguments()).contains("-fobjc-arc");
    assertThat(compileActionA.getArguments()).containsAllOf("-c", "objc/a.m");
    assertThat(compileActionNonArc.getArguments()).contains("-fno-objc-arc");
    assertThat(compileActionA.getArguments()).containsAllIn(FASTBUILD_COPTS);
    assertThat(compileActionA.getArguments())
        .contains("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION);
    assertThat(compileActionA.getArguments()).contains("-arch i386");
  }

  @Test
  public void testCompilationActions_device() throws Exception {
    useConfiguration(
        "--cpu=ios_armv7",
        "--ios_cpu=armv7");

    scratch.file("objc/a.m");
    scratch.file("objc/non_arc.m");
    scratch.file("objc/private.h");
    scratch.file("objc/c.h");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch,
            "objc",
            "lib",
            "srcs",
            "['a.m', 'private.h']",
            "hdrs",
            "['c.h']",
            "non_arc_srcs",
            "['non_arc.m']"));

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    CommandAction compileActionNonArc = compileAction("//objc:lib", "non_arc.o");

    assertRequiresDarwin(compileActionA);
    assertThat(compileActionA.getArguments()).contains("tools/osx/crosstool/ios/" + WRAPPED_CLANG);
    assertThat(compileActionA.getArguments())
        .containsAllOf("-isysroot", AppleToolchain.sdkDir()).inOrder();
    assertThat(Collections.frequency(compileActionA.getArguments(),
        "-F" + AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")).isEqualTo(1);
    assertThat(
            Collections.frequency(
                compileActionA.getArguments(), "-F" + frameworkDir(ApplePlatform.IOS_DEVICE)))
        .isEqualTo(1);
    assertThat(compileActionA.getArguments())
        .containsAllIn(AppleToolchain.DEFAULT_WARNINGS.values());
    assertThat(compileActionA.getArguments())
        .containsAllIn(CompilationSupport.DEFAULT_COMPILER_FLAGS);
    assertThat(compileActionA.getArguments())
        .containsNoneIn(CompilationSupport.SIMULATOR_COMPILE_FLAGS);

    assertThat(compileActionA.getArguments()).contains("-fobjc-arc");
    assertThat(compileActionA.getArguments()).containsAllOf("-c", "objc/a.m");

    assertThat(compileActionNonArc.getArguments()).contains("-fno-objc-arc");
    assertThat(compileActionA.getArguments()).containsAllIn(FASTBUILD_COPTS);
    assertThat(compileActionA.getArguments())
        .contains("-miphoneos-version-min=" + DEFAULT_IOS_SDK_VERSION);
    assertThat(compileActionA.getArguments()).contains("-arch armv7");
  }

  @Test
  public void testArchivesPrecompiledObjectFiles() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    scratch.file("objc/a.m");
    scratch.file("objc/b.o");
    scratch.file("objc/BUILD", RULE_TYPE.target(scratch, "objc", "x", "srcs", "['a.m', 'b.o']"));
    assertThat(Artifact.toRootRelativePaths(archiveAction("//objc:x").getInputs()))
        .contains("objc/b.o");
  }

  @Test
  public void testCompileWithFrameworkImportsIncludesFlagsAndInputArtifacts() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    addBinWithTransitiveDepOnFrameworkImport();
    CommandAction compileAction = compileAction("//lib:lib", "a.o");

    assertThat(compileAction.getArguments()).doesNotContain("-framework");
    assertThat(Joiner.on("").join(compileAction.getArguments())).contains("-Ffx");
    assertThat(compileAction.getInputs())
        .containsAllOf(
            getSourceArtifact("fx/fx1.framework/a"),
            getSourceArtifact("fx/fx1.framework/b"),
            getSourceArtifact("fx/fx2.framework/c"),
            getSourceArtifact("fx/fx2.framework/d"));
  }

  @Test
  public void testPrecompiledHeaders() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    scratch.file("objc/a.m");
    scratch.file("objc/c.pch");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch, "objc", "x", "srcs", "['a.m']", "non_arc_srcs", "['b.m']", "pch", "'c.pch'"));
    CppCompileAction compileAction = (CppCompileAction) compileAction("//objc:x", "a.o");
    assertThat(Joiner.on(" ").join(compileAction.getArguments()))
        .contains("-include objc/c.pch");
    assertThat(Artifact.toRootRelativePaths(compileAction.getPossibleInputsForTesting()))
        .contains("objc/c.pch");
  }

  @Test
  public void testCompilationActionsWithCopts() throws Exception {
    useConfiguration("--cpu=ios_i386", "--ios_cpu=i386");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setList("copts", "-Ifoo", "--monkeys=$(TARGET_CPU)")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    assertThat(compileActionA.getArguments()).containsAllOf("-Ifoo", "--monkeys=ios_i386");
  }

  @Test
  public void testObjcCopts() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--objccopt=-foo");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .write();
    List<String> args = compileAction("//lib:lib", "a.o").getArguments();
    assertThat(args).contains("-foo");
  }

  @Test
  public void testObjcCopts_argumentOrdering() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--objccopt=-foo");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("copts", "-bar")
        .write();
    List<String> args = compileAction("//lib:lib", "a.o").getArguments();
    assertThat(args).containsAllOf("-fobjc-arc", "-foo", "-bar").inOrder();
  }

  @Test
  public void testCompilationActionsWithModuleMapsEnabled() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_enable_module_maps");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    assertThat(compileActionA.getArguments())
        .containsAllIn(moduleMapArtifactArguments("//objc", "lib"));
    assertThat(compileActionA.getArguments()).contains("-fmodule-maps");
    assertThat(Artifact.toRootRelativePaths(compileActionA.getInputs()))
        .doesNotContain("objc/lib.modulemaps/module.modulemap");
  }

  @Test
  public void testCompilationActionsWithEmbeddedBitcode() throws Exception {
    useConfiguration(
        "--ios_multi_cpus=arm64",
        "--apple_bitcode=embedded");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode");
  }

  @Test
  public void testCompilationActionsWithEmbeddedBitcodeMarkers() throws Exception {
    useConfiguration(
        "--ios_multi_cpus=arm64",
        "--apple_bitcode=embedded_markers");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode-marker");
  }

  @Test
  public void testCompilationActionsWithNoBitcode() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--ios_multi_cpus=arm64",
        "--apple_bitcode=none");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode-marker");
  }

  /**
   * Tests that bitcode is disabled for simulator builds even if enabled by flag.
   */
  @Test
  public void testCompilationActionsWithBitcode_simulator() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--ios_multi_cpus=x86_64",
        "--apple_bitcode=embedded");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode-marker");
  }

  @Test
  public void testModuleMapActionFiltersHeaders() throws Exception {
    RULE_TYPE.scratchTarget(
        scratch,
        "srcs",
        "['a.m', 'b.m', 'private.h', 'private.inc']",
        "hdrs",
        "['a.h', 'x.inc', 'foo.m', 'bar.mm']");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:x");
    Artifact moduleMap = getGenfilesArtifact("x.modulemaps/module.modulemap", configuredTarget);

    CppModuleMapAction genMap = (CppModuleMapAction) getGeneratingAction(moduleMap);

    assertThat(Artifact.toRootRelativePaths(genMap.getPrivateHeaders())).isEmpty();
    assertThat(Artifact.toRootRelativePaths(genMap.getPublicHeaders())).containsExactly("x/a.h");
  }

  @Test
  public void testArchiveAction_simulator() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction archiveAction = archiveAction("//objc:lib");
    assertThat(archiveAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                "tools/osx/crosstool/iossim/libtool",
                "-static",
                "-filelist",
                getBinArtifact("lib-archive.objlist", "//objc:lib").getExecPathString(),
                "-arch_only",
                "i386",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                Iterables.getOnlyElement(archiveAction.getOutputs()).getExecPathString()));
    assertThat(baseArtifactNames(archiveAction.getInputs()))
        .containsAllOf("a.o", "b.o", "lib-archive.objlist", CROSSTOOL_LINK_MIDDLEMAN);
    assertThat(baseArtifactNames(archiveAction.getOutputs())).containsExactly("liblib.a");
    assertRequiresDarwin(archiveAction);
  }

  @Test
  public void testArchiveAction_device() throws Exception {
    useConfiguration(
        "--cpu=ios_armv7",
        "--ios_cpu=armv7");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction archiveAction = archiveAction("//objc:lib");

    assertThat(archiveAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                "tools/osx/crosstool/ios/libtool",
                "-static",
                "-filelist",
                getBinArtifact("lib-archive.objlist", "//objc:lib").getExecPathString(),
                "-arch_only",
                "armv7",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                Iterables.getOnlyElement(archiveAction.getOutputs()).getExecPathString()));
    assertThat(baseArtifactNames(archiveAction.getInputs()))
        .containsAllOf("a.o", "b.o", "lib-archive.objlist");
    assertThat(baseArtifactNames(archiveAction.getOutputs())).containsExactly("liblib.a");
    assertRequiresDarwin(archiveAction);
  }

  @Test
  public void testFullyLinkArchiveAction_simulator() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386");
    createLibraryTargetWriter("//objc:lib_dep")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h", "b.h")
        .write();
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("deps", "//objc:lib_dep")
        .write();
    CommandAction linkAction =
        (CommandAction) getGeneratingActionForLabel("//objc2:lib_fully_linked.a");
    assertRequiresDarwin(linkAction);
    assertThat(linkAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                "tools/osx/crosstool/iossim/libtool",
                "-static",
                "-arch_only",
                "i386",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                Iterables.getOnlyElement(linkAction.getOutputs()).getExecPathString(),
                getBinArtifact("liblib.a", "//objc2:lib").getExecPathString(),
                getBinArtifact("liblib_dep.a", "//objc:lib_dep").getExecPathString()));
    // TODO(hlopko): make containsExactly once crosstools are updated so
    // link_dynamic_library.sh is not needed anymore
    assertThat(baseArtifactNames(linkAction.getInputs())).containsAllOf(
        "liblib_dep.a",
        "liblib.a",
        CROSSTOOL_LINK_MIDDLEMAN);
  }

  @Test
  public void testFullyLinkArchiveAction_device() throws Exception {
    useConfiguration(
        "--cpu=ios_armv7",
        "--ios_cpu=armv7");
    createLibraryTargetWriter("//objc:lib_dep")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h", "b.h")
        .write();
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("deps", "//objc:lib_dep")
        .write();
    CommandAction linkAction =
        (CommandAction) getGeneratingActionForLabel("//objc2:lib_fully_linked.a");
    assertRequiresDarwin(linkAction);
    assertThat(linkAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                "tools/osx/crosstool/ios/libtool",
                "-static",
                "-arch_only",
                "armv7",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                Iterables.getOnlyElement(linkAction.getOutputs()).getExecPathString(),
                getBinArtifact("liblib.a", "//objc2:lib").getExecPathString(),
                getBinArtifact("liblib_dep.a", "//objc:lib_dep").getExecPathString()));
    // TODO(hlopko): make containsExactly once crosstools are updated so
    // link_dynamic_library.sh is not needed anymore
    assertThat(baseArtifactNames(linkAction.getInputs())).containsAllOf(
        "liblib_dep.a",
        "liblib.a",
        CROSSTOOL_LINK_MIDDLEMAN);
  }

  @Test
  public void checkDoesNotStoreObjcLibsAsCC() throws Exception {
    createLibraryTargetWriter("//objc:lib_dep")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h", "b.h")
        .write();
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("deps", "//objc:lib_dep")
        .write();
    ObjcProvider objcProvider = providerForTarget("//objc2:lib");
    assertThat(objcProvider.get(CC_LIBRARY)).isEmpty();
  }

  @Test
  public void testIncludesDirsGetPassedToCompileAction() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("includes", "../third_party/foo", "opensource/bar")
        .write();
    CommandAction compileAction = compileAction("//lib:lib", "a.o");

    for (String path :
        rootedIncludePaths(
            getAppleCrosstoolConfiguration(), "third_party/foo", "lib/opensource/bar")) {
      assertThat(Joiner.on("").join(compileAction.getArguments())).contains("-I" + path);
    }
  }

  @Test
  public void testPropagatesDefinesToDependersTransitively() throws Exception {
    useConfiguration(
        "--cpu=ios_x86_64",
        "--ios_cpu=x86_64");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m")
        .setAndCreateFiles("non_arc_srcs", "b.m")
        .setList("defines", "A=foo", "B", "MONKEYS=$(TARGET_CPU)")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m")
        .setAndCreateFiles("non_arc_srcs", "b.m")
        .setList("deps", "//lib1:lib1")
        .setList("defines", "C=bar", "D")
        .write();
    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "c.m")
        .setList("deps", "//lib2:lib2")
        .write();

    assertThat(compileAction("//lib1:lib1", "a.o").getArguments())
        .containsAllOf("-DA=foo", "-DB", "-DMONKEYS=ios_x86_64")
        .inOrder();
    assertThat(compileAction("//lib1:lib1", "b.o").getArguments())
        .containsAllOf("-DA=foo", "-DB", "-DMONKEYS=ios_x86_64")
        .inOrder();
    assertThat(compileAction("//lib2:lib2", "a.o").getArguments())
        .containsAllOf("-DA=foo", "-DB", "-DMONKEYS=ios_x86_64", "-DC=bar", "-DD")
        .inOrder();
    assertThat(compileAction("//lib2:lib2", "b.o").getArguments())
        .containsAllOf("-DA=foo", "-DB", "-DMONKEYS=ios_x86_64", "-DC=bar", "-DD")
        .inOrder();
    // TODO: Add tests for //bin:bin once experimental_objc_binary is implemented
  }

  @Test
  public void testDuplicateDefines() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m")
        .setList("defines", "foo=bar", "foo=bar")
        .write();
    int timesDefinesAppear = 0;
    for (String arg : compileAction("//lib:lib", "a.o").getArguments()) {
      if (arg.equals("-Dfoo=bar")) {
        timesDefinesAppear++;
      }
    }
    assertWithMessage("Duplicate define \"foo=bar\" should occur only once in command line")
        .that(timesDefinesAppear)
        .isEqualTo(1);
  }

  @Test
  public void checkDefinesFromCcLibraryDep() throws Exception {
    checkDefinesFromCcLibraryDep(RULE_TYPE);
  }

  @Test
  public void testCppSourceCompilesWithCppFlags() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    createLibraryTargetWriter("//objc:x")
        .setAndCreateFiles("srcs", "a.mm", "b.cc", "c.mm", "d.cxx", "e.c", "f.m", "g.C")
        .write();
    assertThat(compileAction("//objc:x", "a.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//objc:x", "b.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//objc:x", "c.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//objc:x", "d.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//objc:x", "e.o").getArguments()).doesNotContain("-std=gnu++11");
    assertThat(compileAction("//objc:x", "f.o").getArguments()).doesNotContain("-std=gnu++11");
    assertThat(compileAction("//objc:x", "g.o").getArguments()).contains("-std=gnu++11");
  }

  @Test
  public void testAssetCatalogsAttributeErrorForNotInXcAssetsDir() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    scratch.file("lib/ac/notinxcassets1");
    scratch.file("lib/ac/notinxcassets2");
    scratch.file("lib/ac/foo.xcassets/isinxcassets");
    checkError("lib", "lib",
        String.format(ObjcCommon.NOT_IN_CONTAINER_ERROR_FORMAT,
            "lib/ac/notinxcassets2", ImmutableList.of(ObjcCommon.ASSET_CATALOG_CONTAINER_TYPE)),
        "objc_library(name = 'lib', srcs = ['src.m'], asset_catalogs = glob(['ac/**']))");
  }

  @Test
  public void testXcdatamodelsAttributeErrorForNotInXcdatamodelDir() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    scratch.file("lib/xcd/notinxcdatamodel1");
    scratch.file("lib/xcd/notinxcdatamodel2");
    scratch.file("lib/xcd/foo.xcdatamodel/isinxcdatamodel");
    scratch.file("lib/xcd/bar.xcdatamodeld/isinxcdatamodeld");
    checkError("lib", "lib",
        String.format(ObjcCommon.NOT_IN_CONTAINER_ERROR_FORMAT,
            "lib/xcd/notinxcdatamodel1", Xcdatamodels.CONTAINER_TYPES),
        "objc_library(name = 'lib', srcs = ['src.m'], datamodels = glob(['xcd/**']))");
  }

  @Test
  public void testProvidesStoryboardOptions() throws Exception {
    checkProvidesStoryboardObjects(RULE_TYPE);
  }

  @Test
  public void testDoesNotUseCxxUnfilteredFlags() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    createLibraryTargetWriter("//lib:lib")
        .setList("srcs", "a.m")
        .write();
    // -pthread is an unfiltered_cxx_flag in the osx crosstool.
    assertThat(compileAction("//lib:lib", "a.o").getArguments()).doesNotContain("-pthread");
  }

  @Test
  public void testDoesNotUseDotdPruning() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--objc_use_dotd_pruning=false");
    createLibraryTargetWriter("//lib:lib")
        .setList("srcs", "a.m")
        .write();
    CppCompileAction compileAction = (CppCompileAction) compileAction("//lib:lib", "a.o");
    assertThat(compileAction.discoverInputsFromDotdFiles(null, null, null)).isEmpty();
  }

  @Test
  public void testProvidesObjcLibraryAndHeaders() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:lib")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .setAndCreateFiles("hdrs", "a.h", "b.h")
            .write();
    ConfiguredTarget depender =
        createLibraryTargetWriter("//objc2:lib")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .setAndCreateFiles("hdrs", "c.h", "d.h")
            .setList("deps", "//objc:lib")
            .write();
    assertThat(getArifactPaths(target, LIBRARY)).containsExactly("objc/liblib.a");
    assertThat(getArifactPaths(depender, LIBRARY)).containsExactly(
        "objc/liblib.a", "objc2/liblib.a");
    assertThat(getArifactPaths(target, HEADER))
        .containsExactly("objc/a.h", "objc/b.h");
    assertThat(getArifactPaths(depender, HEADER))
        .containsExactly("objc/a.h", "objc/b.h", "objc2/c.h", "objc2/d.h");
  }

  private Iterable<String> getArifactPaths(ConfiguredTarget target, Key<Artifact> artifactKey) {
    return Artifact.toRootRelativePaths(
        target.get(ObjcProvider.SKYLARK_CONSTRUCTOR).get(artifactKey));
  }

  @Test
  public void testWeakSdkFrameworks_objcProvider() throws Exception {
    createLibraryTargetWriter("//base_lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("weak_sdk_frameworks", "foo")
        .write();
    createLibraryTargetWriter("//depender_lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("weak_sdk_frameworks", "bar")
        .setList("deps", "//base_lib:lib")
        .write();

    ObjcProvider baseProvider = providerForTarget("//base_lib:lib");
    ObjcProvider dependerProvider = providerForTarget("//depender_lib:lib");

    assertThat(baseProvider.get(WEAK_SDK_FRAMEWORK)).containsExactly(new SdkFramework("foo"));
    assertThat(dependerProvider.get(WEAK_SDK_FRAMEWORK))
        .containsExactly(new SdkFramework("foo"), new SdkFramework("bar"));
  }

  @Test
  public void testErrorIfDepDoesNotExist() throws Exception {
    checkErrorIfNotExist("deps", "[':nonexistent']");
  }

  @Test
  public void testArIsNotImplicitOutput() throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .write();
    try {
      reporter.removeHandler(failFastHandler);
      getTarget("//lib:liblib.a");
      fail("should have thrown");
    } catch (NoSuchTargetException expected) {
    }
  }

  @Test
  public void testErrorForAbsoluteIncludesPath() throws Exception {
    scratch.file("x/a.m");
    checkError(
        "x",
        "x",
        String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, "/absolute/path"),
        "objc_library(",
        "    name = 'x',",
        "    srcs = ['a.m'],",
        "    includes = ['/absolute/path'],",
        ")");
  }

  @Test
  public void testExportsBundleDependencies() throws Exception {
    scratch.file("bundle/bar/x.bundle/1");
    scratch.file(
        "bundle/BUILD",
        "objc_bundle(",
        "    name = 'bundle',",
        "    bundle_imports = glob(['bar/**']),",
        ")");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("bundles", "//bundle:bundle")
        .write();
    ObjcProvider provider = providerForTarget("//lib:lib");
    assertThat(provider.get(BUNDLE_FILE))
        .contains(new BundleableFile(getSourceArtifact("bundle/bar/x.bundle/1"), "x.bundle/1"));
  }

  @Test
  public void testDylibsProvided() throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("sdk_dylibs", "libdy1", "libdy2")
        .write();
    ObjcProvider provider = providerForTarget("//lib:lib");
    assertThat(provider.get(SDK_DYLIB)).containsExactly("libdy1", "libdy2").inOrder();
  }

  @Test
  public void testPopulatesCompilationArtifacts() throws Exception {
    checkPopulatesCompilationArtifacts(RULE_TYPE);
  }

  @Test
  public void testProvidesXcassetCatalogsTransitively() throws Exception {
    scratch.file("lib1/ac.xcassets/foo");
    scratch.file("lib1/ac.xcassets/bar");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("asset_catalogs", "glob(['ac.xcassets/**'])")
        .write();
    scratch.file("lib2/ac.xcassets/baz");
    scratch.file("lib2/ac.xcassets/42");
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("asset_catalogs", "glob(['ac.xcassets/**'])")
        .setList("deps", "//lib1:lib1")
        .write();

    ObjcProvider lib2Provider = providerForTarget("//lib2:lib2");
    assertThat(Artifact.toExecPaths(lib2Provider.get(ASSET_CATALOG)))
        .containsExactly(
            "lib1/ac.xcassets/foo",
            "lib1/ac.xcassets/bar",
            "lib2/ac.xcassets/baz",
            "lib2/ac.xcassets/42");
    assertThat(lib2Provider.get(XCASSETS_DIR))
        .containsExactly(
            PathFragment.create("lib1/ac.xcassets"), PathFragment.create("lib2/ac.xcassets"));

    ObjcProvider lib1Provider = providerForTarget("//lib1:lib1");
    assertThat(Artifact.toExecPaths(lib1Provider.get(ASSET_CATALOG)))
        .containsExactly("lib1/ac.xcassets/foo", "lib1/ac.xcassets/bar");
    assertThat(lib1Provider.get(XCASSETS_DIR))
        .containsExactly(PathFragment.create("lib1/ac.xcassets"))
        .inOrder();
  }

  @Test
  public void testObjcListFileInArchiveGeneration() throws Exception {
    scratch.file("lib/a.m");
    scratch.file("lib/b.m");
    scratch.file("lib/BUILD", "objc_library(name = 'lib1', srcs = ['a.m', 'b.m'])");
    ConfiguredTarget target = getConfiguredTarget("//lib:lib1");
    Artifact objlist = getBinArtifact("lib1-archive.objlist", target);
    ParameterFileWriteAction action = (ParameterFileWriteAction) getGeneratingAction(objlist);
    assertThat(action.getContents())
        .containsExactlyElementsIn(
            Artifact.toExecPaths(inputsEndingWith(archiveAction("//lib:lib1"), ".o")));
  }

  @Test
  public void testErrorsWrongFileTypeForSrcsWhenCompiling() throws Exception {
    checkErrorsWrongFileTypeForSrcsWhenCompiling(RULE_TYPE);
  }

  @Test
  public void testCompilationActionsForDebug() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG, CodeCoverageMode.NONE);
  }

  @Test
  public void testClangCoptsForDebugModeWithoutGlib() throws Exception {
    checkClangCoptsForDebugModeWithoutGlib(RULE_TYPE);
  }

  @Test
  public void testCompilationActionsForOptimized() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT, CodeCoverageMode.NONE);
  }

  @Test
  public void testUsesDefinesFromTransitiveCcDeps() throws Exception {
    scratch.file(
        "package/BUILD",
        "cc_library(",
        "    name = 'cc_lib',",
        "    srcs = ['a.cc'],",
        "    defines = ['FOO'],",
        ")",
        "",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['b.m'],",
        "    deps = [':cc_lib'],",
        ")");

    CommandAction compileAction = compileAction("//package:objc_lib", "b.o");
    assertThat(compileAction.getArguments()).contains("-DFOO");
  }

  @Test
  public void testAllowVariousNonBlacklistedTypesInHeaders() throws Exception {
    checkAllowVariousNonBlacklistedTypesInHeaders(RULE_TYPE);
  }

  @Test
  public void testWarningForBlacklistedTypesInHeaders() throws Exception {
    checkWarningForBlacklistedTypesInHeaders(RULE_TYPE);
  }

  @Test
  public void testBundleInformationPropagatedThroughLibraries() throws Exception {
    checkNestedBundleInformationPropagatedToDependers(RULE_TYPE);
  }

  @Test
  public void testAppleSdkVersionEnv() throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction action = compileAction("//objc:lib", "a.o");

    assertAppleSdkVersionEnv(action);
  }

  @Test
  public void testNonDefaultAppleSdkVersionEnv() throws Exception {
    useConfiguration("--ios_sdk_version=8.1");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction action = compileAction("//objc:lib", "a.o");

    assertAppleSdkVersionEnv(action, "8.1");
  }

  @Test
  public void testXcodeVersionEnv() throws Exception {
    useConfiguration("--xcode_version=5.8");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction action = compileAction("//objc:lib", "a.o");

    assertXcodeVersionEnv(action, "5.8");
  }

  @Test
  public void testIosSdkVersionCannotBeDefinedButEmpty() throws Exception {
    try {
      useConfiguration("--ios_sdk_version=");
      fail("Should fail for empty ios_sdk_version");
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessageThat().contains("--ios_sdk_version");
    }
  }

  private void checkErrorIfNotExist(String attribute, String value) throws Exception {
    scratch.file("x/a.m");
    checkError(
        "x",
        "x",
        "in "
            + attribute
            + " attribute of objc_library rule //x:x: rule '//x:nonexistent' does not exist",
        "objc_library(",
        "    name = 'x',",
        "    srcs = ['a.m'],",
        attribute + " = " + value,
        ")");
  }

  @Test
  public void testCompilesWithHdrs() throws Exception {
    checkCompilesWithHdrs(ObjcLibraryTest.RULE_TYPE);
  }

  @Test
  public void testCompilesAssemblyWithPreprocessing() throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.S")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileAction = compileAction("//objc:lib", "b.o");

    // Clang automatically preprocesses .S files, so the assembler-with-cpp flag is unnecessary.
    // Regression test for b/22636858.
    assertThat(compileAction.getArguments()).doesNotContain("-x");
    assertThat(compileAction.getArguments()).doesNotContain("assembler-with-cpp");
    assertThat(baseArtifactNames(compileAction.getOutputs())).containsExactly("b.o", "b.d");
    assertThat(baseArtifactNames(compileAction.getPossibleInputsForTesting()))
        .containsAllOf("c.h", "b.S");
  }

  @Test
  public void testReceivesTransitivelyPropagatedDefines() throws Exception {
    checkReceivesTransitivelyPropagatedDefines(RULE_TYPE);
  }

  @Test
  public void testSdkIncludesUsedInCompileAction() throws Exception {
    checkSdkIncludesUsedInCompileAction(RULE_TYPE);
  }

  @Test
  public void testUsesDotdPruning() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL, "--objc_use_dotd_pruning");
    createLibraryTargetWriter("//lib:lib").setList("srcs", "a.m").write();
    CppCompileAction compileAction = (CppCompileAction) compileAction("//lib:lib", "a.o");
    try {
      compileAction.discoverInputsFromDotdFiles(null, null, null);
      fail("Expected ActionExecutionException");
    } catch (ActionExecutionException expected) {
      assertThat(expected).hasMessageThat().contains("error while parsing .d file");
    }
  }

  @Test
  public void testAppleSdkDefaultPlatformEnv() throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction action = compileAction("//objc:lib", "a.o");

    assertAppleSdkPlatformEnv(action, "iPhoneSimulator");
  }

  @Test
  public void testAppleSdkDevicePlatformEnv() throws Exception {
    useConfiguration("--cpu=ios_arm64");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction action = compileAction("//objc:lib", "a.o");

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  @Test
  public void testApplePlatformEnvForCcLibraryDep() throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file("package/BUILD",
        "cc_library(",
        "    name = 'cc_lib',",
        "    srcs = ['a.cc'],",
        ")",
        "",
        "objc_binary(",
        "    name = 'objc_bin',",
        "    srcs = ['b.m'],",
        "    deps = [':cc_lib'],",
        ")");

    Action binLinkAction = linkAction("//package:objc_bin");
    Artifact artifact =
        ActionsTestUtil.getFirstArtifactEndingWith(binLinkAction.getInputs(), "libcc_lib.a");
    Action cppLibLinkAction = getGeneratingAction(artifact);
    Artifact cppLibArtifact =
        ActionsTestUtil.getFirstArtifactEndingWith(cppLibLinkAction.getInputs(), ".o");

    CppCompileAction action = (CppCompileAction) getGeneratingAction(cppLibArtifact);
    assertAppleSdkVersionEnv(action.getEnvironment());
  }

  @Test
  public void testDoesNotPropagateProtoIncludes() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    scratch.file(
        "x/BUILD",
        "proto_library(",
        "   name = 'protos',",
        "   srcs = ['data.proto'],",
        ")",
        "objc_proto_library(",
        "   name = 'objc_proto_lib',",
        "   deps = [':protos'],",
        "   portable_proto_filters = ['data_filter.pbascii'],",
        ")");
    createLibraryTargetWriter("//a:lib")
        .setList("srcs", "a.m")
        .setList("deps", "//x:objc_proto_lib")
        .write();
    createLibraryTargetWriter("//b:lib").setList("srcs", "b.m").setList("deps", "//a:lib").write();

    CommandAction compileAction1 = compileAction("//a:lib", "a.o");
    CommandAction compileAction2 = compileAction("//b:lib", "b.o");

    assertThat(Joiner.on(" ").join(compileAction1.getArguments())).contains("objc_proto_lib");
    assertThat(Joiner.on(" ").join(compileAction2.getArguments())).doesNotContain("objc_proto_lib");
  }

  @Test
  public void testExportsJ2ObjcProviders() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    ConfiguredTarget lib = createLibraryTargetWriter("//a:lib").write();
    assertThat(lib.getProvider(J2ObjcEntryClassProvider.class)).isNotNull();
    assertThat(lib.getProvider(J2ObjcMappingFileProvider.class)).isNotNull();
  }

  @Test
  public void testObjcProtoLibraryDoesNotCrash() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all");
    scratch.file(
        "x/BUILD",
        "objc_library(",
        "   name = 'objc',",
        "   srcs = ['source.m'],",
        "   deps = [':objc_proto_lib'],",
        ")",
        "proto_library(",
        "   name = 'protos',",
        "   srcs = ['data.proto'],",
        ")",
        "objc_proto_library(",
        "   name = 'objc_proto_lib',",
        "   deps = [':protos'],",
        "   portable_proto_filters = ['data_filter.pbascii'],",
        ")");
    assertThat(getConfiguredTarget("//x:objc")).isNotNull();
  }

  @Test
  public void testLegacyObjcProtoLibraryDoesNotCrash() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all");
    scratch.file(
        "x/BUILD",
        "objc_library(",
        "   name = 'objc',",
        "   srcs = ['source.m'],",
        "   deps = [':objc_proto_lib'],",
        ")",
        "proto_library(",
        "   name = 'protos',",
        "   srcs = ['data.proto'],",
        ")",
        "objc_proto_library(",
        "   name = 'objc_proto_lib',",
        "   deps = [':protos'],",
        ")");
    assertThat(getConfiguredTarget("//x:objc")).isNotNull();
  }

  @Test
  public void testObjcImportDoesNotCrash() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--experimental_objc_crosstool=all");
    scratch.file(
        "x/BUILD",
        "objc_library(",
        "   name = 'objc',",
        "   srcs = ['source.m'],",
        "   deps = [':import'],",
        ")",
        "objc_import(",
        "   name = 'import',",
        "   archives = ['archive.a'],",
        ")");
    assertThat(getConfiguredTarget("//x:objc")).isNotNull();
  }

  @Test
  public void testCompilationActionsWithIQuotesInCopts() throws Exception {
    useConfiguration(
        "--cpu=ios_i386",
        "--ios_cpu=i386");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setList("copts", "-iquote foo/bar", "-iquote bam/baz")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    String action = String.join(" ", compileActionA.getArguments());
    assertThat(action).contains("-iquote foo/bar");
    assertThat(action).contains("-iquote bam/baz");
  }
  @Test
  public void testCollectCodeCoverageWithGCOVFlags() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL, "--collect_code_coverage");
    createLibraryTargetWriter("//objc:x")
        .setAndCreateFiles("srcs", "a.mm", "b.cc", "c.mm", "d.cxx", "e.c", "f.m", "g.C")
        .write();
    List<String> copts = ImmutableList.of("-fprofile-arcs", "-ftest-coverage");
    assertThat(compileAction("//objc:x", "a.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "b.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "c.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "d.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "e.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "f.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "g.o").getArguments()).containsAllIn(copts);
  }

  @Test
  public void testCollectCodeCoverageWithLLVMCOVFlags() throws Exception {
    useConfiguration(
        "--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL,
        "--collect_code_coverage",
        "--experimental_use_llvm_covmap");
    createLibraryTargetWriter("//objc:x")
        .setAndCreateFiles("srcs", "a.mm", "b.cc", "c.mm", "d.cxx", "e.c", "f.m", "g.C")
        .write();
    List<String> copts = ImmutableList.of("-fprofile-instr-generate", "-fcoverage-mapping");
    assertThat(compileAction("//objc:x", "a.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "b.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "c.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "d.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "e.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "f.o").getArguments()).containsAllIn(copts);
    assertThat(compileAction("//objc:x", "g.o").getArguments()).containsAllIn(copts);
   }

  @Test
  public void testNoG0IfGeneratesDsym() throws Exception {
    useConfiguration("--apple_generate_dsym", "-c", "opt");
    createLibraryTargetWriter("//x:x").setList("srcs", "a.m").write();
    CommandAction compileAction = compileAction("//x:x", "a.o");
    assertThat(compileAction.getArguments()).doesNotContain("-g0");
  }

  @Test
  public void testFilesToCompileOutputGroup() throws Exception {
    checkFilesToCompileOutputGroup(RULE_TYPE);
  }

  @Test
  public void testSysrootArgSpecifiedWithGrteTopFlag() throws Exception {
    MockObjcSupport.setup(mockToolsConfig, "default_grte_top : '//x'");
    useConfiguration(
        ObjcCrosstoolMode.ALL,
        "--cpu=ios_x86_64",
        "--ios_cpu=x86_64");
    scratch.file(
        "x/BUILD",
        "objc_library(",
        "   name = 'objc',",
        "   srcs = ['source.m'],",
        ")",
        "filegroup(",
        "    name = 'everything',",
        "    srcs = ['header.h'],",
        ")");
    CommandAction compileAction = compileAction("//x:objc", "source.o");
    assertThat(compileAction.getArguments()).contains("--sysroot=x");
  }

  @Test
  public void testDefaultEnabledFeatureIsUsed() throws Exception {
    MockObjcSupport.setup(mockToolsConfig,
        "feature {",
        "  name: 'default'",
        "  enabled : true",
        "  flag_set {",
        "    action: 'objc-compile'",
        "    flag_group {",
        "      flag: '-dummy'",
        "    }",
        "  }",
        "}");
    useConfiguration(
        ObjcCrosstoolMode.ALL,
        "--cpu=ios_x86_64",
        "--ios_cpu=x86_64");
    scratch.file(
        "x/BUILD",
        "objc_library(",
        "   name = 'objc',",
        "   srcs = ['source.m'],",
        ")");
    CommandAction compileAction = compileAction("//x:objc", "source.o");
    assertThat(compileAction.getArguments()).contains("-dummy");
  }

  @Test
  public void testCustomModuleMap() throws Exception {
    checkCustomModuleMap(RULE_TYPE);
  }

  private boolean containsObjcFeature(String srcName) throws Exception {
     MockObjcSupport.setup(
        mockToolsConfig,
        "feature {",
        "  name: 'contains_objc_sources'",
        "  flag_set {",
        "    flag_group {",
        "      flag: 'DUMMY_FLAG'",
        "    }",
        "    action: 'c++-compile'",
        "  }",
        "}");
    createLibraryTargetWriter("//bottom:lib").setList("srcs", srcName).write();
    createLibraryTargetWriter("//middle:lib")
        .setList("srcs", "b.cc")
        .setList("deps", "//bottom:lib")
        .write();
    createLibraryTargetWriter("//top:lib")
        .setList("srcs", "a.cc")
        .setList("deps", "//middle:lib")
        .write();

    CommandAction compileAction = compileAction("//top:lib", "a.o");
    return compileAction.getArguments().contains("DUMMY_FLAG");
  }

  @Test
  public void testObjcSourcesFeatureCC() throws Exception {
    assertThat(containsObjcFeature("c.cc")).isFalse();
  }

  @Test
  public void testObjcSourcesFeatureObjc() throws Exception {
     assertThat(containsObjcFeature("c.m")).isTrue();
  }

  @Test
  public void testObjcSourcesFeatureObjcPlusPlus() throws Exception {
     assertThat(containsObjcFeature("c.mm")).isTrue();
  }
}
