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
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Legacy test case (that is, without the OSX crosstool) for objc_library. */
@RunWith(JUnit4.class)
@LegacyTest
public class LegacyObjcLibraryTest extends ObjcLibraryTest {
  private static final RuleType RULE_TYPE = new OnlyNeedsSourcesRuleType("objc_library");
  private static final String XCRUNWRAPPER = "xcrunwrapper";
  private static final String LIBTOOL = "libtool";

  @Override
  protected void useConfiguration(String... args) throws Exception {
    // Crosstool case is tested in {@link ObjcLibraryTest}
    useConfiguration(ObjcCrosstoolMode.OFF, args);
  }

  @Override
  @Test
  public void testObjcSourcesFeatureCC() throws Exception {
    // Features are not exported by legacy actions.
  }

  @Override
  @Test
  public void testObjcSourcesFeatureObjc() throws Exception {
    // Features are not exported by legacy actions.
  }

  @Override
  @Test
  public void testObjcSourcesFeatureObjcPlusPlus() throws Exception {
    // Features are not exported by legacy actions.
  }

  // Crosstool rules do not account for slashes in target names.
  @Test
  public void testLibFileIsCorrectForSlashInTargetName() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:dir/Target")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .write();
    Iterable<Artifact> outputArtifacts = getFilesToBuild(target);
    assertThat(Artifact.toRootRelativePaths(outputArtifacts)).containsExactly("objc/libTarget.a");
  }

  // Override required for distinct compiler path
  @Override
  @Test
  public void testCompilationActions_simulator() throws Exception {
    useConfiguration("--cpu=ios_i386", "--ios_minimum_os=1.0");
    ApplePlatform platform = ApplePlatform.IOS_SIMULATOR;
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setAndCreateFiles("non_arc_srcs", "not_arc.m")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    CommandAction compileActionNotArc = compileAction("//objc:lib", "not_arc.o");

    List<String> commonCompileFlags =
        new ImmutableList.Builder<String>()
            .add(MOCK_XCRUNWRAPPER_PATH)
            .add(ObjcRuleClasses.CLANG)
            .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
            .add("-fexceptions")
            .add("-fasm-blocks")
            .add("-fobjc-abi-version=2")
            .add("-fobjc-legacy-dispatch")
            .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
            .add("-mios-simulator-version-min=1.0")
            .add("-arch", "i386")
            .add("-isysroot", AppleToolchain.sdkDir())
            .add("-F", AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")
            .add("-F", frameworkDir(platform))
            .addAll(FASTBUILD_COPTS)
            .addAll(
                iquoteArgs(
                    getConfiguredTarget("//objc:lib").get(ObjcProvider.SKYLARK_CONSTRUCTOR),
                    getTargetConfiguration()))
            .build();

    assertThat(compileActionA.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .addAll(commonCompileFlags)
                .add("-fobjc-arc")
                .add("-c", "objc/a.m")
                .addAll(outputArgs(compileActionA.getOutputs()))
                .build());
    assertThat(compileActionNotArc.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .addAll(commonCompileFlags)
                .add("-fno-objc-arc")
                .add("-c", "objc/not_arc.m")
                .addAll(outputArgs(compileActionNotArc.getOutputs()))
                .build());

    assertRequiresDarwin(compileActionA);
    try {
      reporter.removeHandler(failFastHandler);
      getTarget("//objc:c.o");
      fail("should have thrown");
    } catch (NoSuchTargetException expected) {}
    assertThat(baseArtifactNames(compileActionA.getOutputs())).containsExactly("a.o", "a.d");
    assertThat(baseArtifactNames(compileActionA.getInputs()))
        .containsExactly("a.m", "c.h", "private.h", XCRUNWRAPPER);
  }

  // Override required for distinct compiler path
  @Override
  @Test
  public void testCompilationActions_device() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_minimum_os=1.0");
    ApplePlatform platform = ApplePlatform.IOS_DEVICE;

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setAndCreateFiles("non_arc_srcs", "not_arc.m")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    CommandAction compileActionNotArc = compileAction("//objc:lib", "not_arc.o");

    List<String> commonCompileFlags =
        new ImmutableList.Builder<String>()
            .add(MOCK_XCRUNWRAPPER_PATH)
            .add(ObjcRuleClasses.CLANG)
            .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
            .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
            .add("-miphoneos-version-min=1.0")
            .add("-arch", "armv7")
            .add("-isysroot", AppleToolchain.sdkDir())
            .add("-F", AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")
            .add("-F", frameworkDir(platform))
            .addAll(FASTBUILD_COPTS)
            .addAll(
                iquoteArgs(
                    getConfiguredTarget("//objc:lib").get(ObjcProvider.SKYLARK_CONSTRUCTOR),
                    getTargetConfiguration()))
            .build();

    assertThat(compileActionA.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .addAll(commonCompileFlags)
                .add("-fobjc-arc")
                .add("-c", "objc/a.m")
                .addAll(outputArgs(compileActionA.getOutputs()))
                .build());
    assertThat(compileActionNotArc.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .addAll(commonCompileFlags)
                .add("-fno-objc-arc")
                .add("-c", "objc/not_arc.m")
                .addAll(outputArgs(compileActionNotArc.getOutputs()))
                .build());

    assertRequiresDarwin(compileActionA);
    try {
      reporter.removeHandler(failFastHandler);
      getTarget("//objc:c.o");
      fail("should have thrown");
    } catch (NoSuchTargetException expected) {}
    assertThat(baseArtifactNames(compileActionA.getOutputs())).containsExactly("a.o", "a.d");
    assertThat(baseArtifactNames(compileActionA.getInputs()))
        .containsExactly("a.m", "c.h", "private.h", XCRUNWRAPPER);
  }

  // Override required for distinct compiler path, command line args.
  @Override
  @Test
  public void testCompilationActionsWithPch() throws Exception {
    ApplePlatform platform = ApplePlatform.IOS_SIMULATOR;
    scratch.file("objc/foo.pch");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .set("pch", "'some.pch'")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments())
        .containsExactlyElementsIn(
            new ImmutableList.Builder<String>()
                .add(MOCK_XCRUNWRAPPER_PATH)
                .add(ObjcRuleClasses.CLANG)
                .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
                .add("-fexceptions")
                .add("-fasm-blocks")
                .add("-fobjc-abi-version=2")
                .add("-fobjc-legacy-dispatch")
                .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
                .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
                .add("-arch", "x86_64")
                .add("-isysroot", AppleToolchain.sdkDir())
                .add("-F", AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")
                .add("-F", frameworkDir(platform))
                .addAll(FASTBUILD_COPTS)
                .addAll(
                    iquoteArgs(
                        getConfiguredTarget("//objc:lib").get(ObjcProvider.SKYLARK_CONSTRUCTOR),
                        getAppleCrosstoolConfiguration()))
                .add("-include", "objc/some.pch")
                .add("-fobjc-arc")
                .add("-c", "objc/a.m")
                .addAll(outputArgs(compileActionA.getOutputs()))
                .build())
        .inOrder();

    assertThat(compileActionA.getInputs()).contains(
        getFileConfiguredTarget("//objc:some.pch").getArtifact());
  }

  // Override required for distinct compiler path
  @Override
  @Test
  public void testCompilationActionsWithCopts() throws Exception {
    useConfiguration("--cpu=ios_i386");
    ApplePlatform platform = ApplePlatform.IOS_SIMULATOR;
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setList("copts", "-Ifoo", "--monkeys=$(TARGET_CPU)")
        .write();

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");

    assertThat(compileActionA.getArguments())
        .containsExactlyElementsIn(
            new ImmutableList.Builder<String>()
                .add(MOCK_XCRUNWRAPPER_PATH)
                .add(ObjcRuleClasses.CLANG)
                .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
                .add("-fexceptions")
                .add("-fasm-blocks")
                .add("-fobjc-abi-version=2")
                .add("-fobjc-legacy-dispatch")
                .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
                .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
                .add("-arch", "i386")
                .add("-isysroot", AppleToolchain.sdkDir())
                .add("-F", AppleToolchain.sdkDir() + "/Developer/Library/Frameworks")
                .add("-F", frameworkDir(platform))
                .addAll(FASTBUILD_COPTS)
                .addAll(
                    iquoteArgs(
                        getConfiguredTarget("//objc:lib").get(ObjcProvider.SKYLARK_CONSTRUCTOR),
                        getTargetConfiguration()))
                .add("-fobjc-arc")
                .add("-Ifoo")
                .add("--monkeys=ios_i386")
                .add("-c", "objc/a.m")
                .addAll(outputArgs(compileActionA.getOutputs()))
                .build())
        .inOrder();
  }

  // Override required since module map is not included in action inputs for the crosstool case
  @Override
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
        .contains("objc/lib.modulemaps/module.modulemap");
  }

  // Override required for distinct libtool path
  @Override
  @Test
  public void testArchiveAction_simulator() throws Exception {
    useConfiguration("--cpu=ios_i386");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction archiveAction = archiveAction("//objc:lib");
    assertThat(archiveAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                MOCK_LIBTOOL_PATH,
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
        .containsExactly("a.o", "b.o", "lib-archive.objlist", LIBTOOL);
    assertThat(baseArtifactNames(archiveAction.getOutputs())).containsExactly("liblib.a");
    assertRequiresDarwin(archiveAction);
  }

  // Override required for distinct libtool path
  @Override
  @Test
  public void testArchiveAction_device() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();
    CommandAction archiveAction = archiveAction("//objc:lib");
    assertThat(archiveAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                MOCK_LIBTOOL_PATH,
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
        .containsExactly("a.o", "b.o", "lib-archive.objlist", LIBTOOL);
    assertThat(baseArtifactNames(archiveAction.getOutputs())).containsExactly("liblib.a");
    assertRequiresDarwin(archiveAction);
  }

  // Override required for distinct libtool path
  @Override
  @Test
  public void testFullyLinkArchiveAction_simulator() throws Exception {
    useConfiguration("--cpu=ios_i386");
    createLibraryTargetWriter("//objc:lib_dep")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h", "b.h")
        .write();
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("deps", "//objc:lib_dep")
        .write();
    SpawnAction archiveAction = (SpawnAction) getGeneratingActionForLabel(
        "//objc2:lib_fully_linked.a");
    assertRequiresDarwin(archiveAction);
    assertThat(archiveAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                MOCK_LIBTOOL_PATH,
                "-static",
                "-arch_only",
                "i386",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                Iterables.getOnlyElement(archiveAction.getOutputs()).getExecPathString(),
                getBinArtifact("liblib.a", "//objc2:lib").getExecPathString(),
                getBinArtifact("liblib_dep.a", "//objc:lib_dep").getExecPathString()));
    assertThat(baseArtifactNames(archiveAction.getInputs()))
        .containsExactly("liblib_dep.a", "liblib.a", LIBTOOL);
  }

  // Override required for distinct libtool path
  @Override
  @Test
  public void testFullyLinkArchiveAction_device() throws Exception {
    useConfiguration("--cpu=ios_armv7");
    createLibraryTargetWriter("//objc:lib_dep")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "a.h", "b.h")
        .write();
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("deps", "//objc:lib_dep")
        .write();
    SpawnAction archiveAction = (SpawnAction) getGeneratingActionForLabel(
        "//objc2:lib_fully_linked.a");
    assertRequiresDarwin(archiveAction);
    assertThat(archiveAction.getArguments())
        .isEqualTo(
            ImmutableList.of(
                MOCK_LIBTOOL_PATH,
                "-static",
                "-arch_only",
                "armv7",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                Iterables.getOnlyElement(archiveAction.getOutputs()).getExecPathString(),
                getBinArtifact("liblib.a", "//objc2:lib").getExecPathString(),
                getBinArtifact("liblib_dep.a", "//objc:lib_dep").getExecPathString()));
    assertThat(baseArtifactNames(archiveAction.getInputs()))
        .containsExactly("liblib_dep.a", "liblib.a", LIBTOOL);
  }

  // Dotd pruning must be tested seperately for the legacy case, since it involves the
  // ObjcCompileAction.
  @Override
  @Test
  public void testUsesDotdPruning() throws Exception {
    createLibraryTargetWriter("//lib:lib").setList("srcs", "a.m").write();
    CommandAction compileAction = compileAction("//lib:lib", "a.o");
    assertThat(compileAction).isInstanceOf(ObjcCompileAction.class);
    assertThat(((ObjcCompileAction) compileAction).getDotdPruningPlan())
        .isEqualTo(HeaderDiscovery.DotdPruningMode.USE);
  }

  @Override
  @Test
  public void testDoesNotUseDotdPruning() throws Exception {
    useConfiguration("--objc_use_dotd_pruning=false");
    createLibraryTargetWriter("//lib:lib").setList("srcs", "a.m").write();
    CommandAction compileAction = compileAction("//lib:lib", "a.o");
    assertThat(compileAction).isInstanceOf(ObjcCompileAction.class);
    assertThat(((ObjcCompileAction) compileAction).getDotdPruningPlan())
        .isEqualTo(HeaderDiscovery.DotdPruningMode.DO_NOT_USE);
  }

  // Override required because CppCompileAction#getPossibleInputsForTesting is not available to
  // test for the presence of inputs that will be pruned.
  @Override
  @Test
  public void testPrecompiledHeaders() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
    scratch.file("objc/a.m");
    scratch.file("objc/c.pch");
    scratch.file(
        "objc/BUILD",
        RULE_TYPE.target(
            scratch, "objc", "x", "srcs", "['a.m']", "non_arc_srcs", "['b.m']", "pch", "'c.pch'"));
    CommandAction compileAction = compileAction("//objc:x", "a.o");
    assertThat(Joiner.on(" ").join(compileAction.getArguments())).contains("-include objc/c.pch");
  }

  // Override required because CppCompileAction#getPossibleInputsForTesting is not available to
  // test for the presence of inputs that will be pruned.
  @Override
  @Test
  public void testCompilesSources() throws Exception {
    useConfiguration("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);
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

    CommandAction compileA = compileAction("//objc:x", "a.o");

    assertThat(Artifact.toRootRelativePaths(compileA.getInputs()))
        .contains("objc/a.m");
    assertThat(Artifact.toRootRelativePaths(compileA.getOutputs()))
        .containsExactly("objc/_objs/x/objc/a.o", "objc/_objs/x/objc/a.d");
  }


  @Override
  @Test
  public void testCompilationModeDbg() throws Exception {
    // Feature requires crosstool
  }

  @Override
  @Test
  public void testCompilationModeFastbuild() throws Exception {
    // Feature requires crosstool
  }

  @Override
  @Test
  public void testCompilationModeOpt() throws Exception {
    // Feature requires crosstool
  }
}
