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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.CC_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Legacy test case (that is, without the OSX crosstool) for objc_library. */
@RunWith(JUnit4.class)
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

  @Test
  public void testLibFileIsCorrectForSlashInTargetName() throws Exception {
    ConfiguredTarget target =
        createLibraryTargetWriter("//objc:dir/Target")
            .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
            .write();
    Iterable<Artifact> outputArtifacts = getFilesToBuild(target);
    assertThat(Artifact.toRootRelativePaths(outputArtifacts)).containsExactly("objc/libTarget.a");
  }

  static Iterable<String> iquoteArgs(ObjcProvider provider, BuildConfiguration configuration) {
    return Interspersing.beforeEach(
        "-iquote",
        PathFragment.safePathStrings(ObjcCommon.userHeaderSearchPaths(provider, configuration)));
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

  // Test with ios device SDK version 9.0. Framework path differs from previous versions.
  @Test
  public void testCompilationActions_deviceSdk9() throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_minimum_os=1.0", "--ios_sdk_version=9.0");

    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileAction = compileAction("//objc:lib", "a.o");

    assertThat(compileAction.getArguments()).containsAllOf(
        "-F", AppleToolchain.sdkDir() + AppleToolchain.SYSTEM_FRAMEWORK_PATH).inOrder();
  }

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

  @Test
  public void testCompilationActionsWithCoptFmodules() throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .setList("copts", "-fmodules")
        .write();
    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    assertThat(compileActionA.getArguments()).containsAllOf("-fmodules",
        "-fmodules-cache-path=" + getModulesCachePath());
  }
  
  @Test
  public void testCompilationActionsWithCoptFmodulesCachePath() throws Exception {
    checkWarning("objc", "lib", CompilationSupport.MODULES_CACHE_PATH_WARNING,
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        "    copts = ['-fmodules', '-fmodules-cache-path=foobar']",
        ")");

    CommandAction compileActionA = compileAction("//objc:lib", "a.o");
    assertThat(compileActionA.getArguments()).containsAllOf("-fmodules",
        "-fmodules-cache-path=" + getModulesCachePath());
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

  @Test
  public void checkStoresCcLibsAsCc() throws Exception {
    ScratchAttributeWriter.fromLabelString(this, "cc_library", "//cc:lib")
        .setAndCreateFiles("srcs", "a.cc")
        .write();
    scratch.file(
        "third_party/cc_lib/BUILD",
        "licenses(['unencumbered'])",
        "cc_library(",
        "    name = 'cc_lib_impl',",
        "    srcs = [",
        "        'v1/a.c',",
        "        'v1/a.h',",
        "    ],",
        ")",
        "",
        "cc_inc_library(",
        "    name = 'cc_lib',",
        "    hdrs = ['v1/a.h'],",
        "    prefix = 'v1',",
        "    deps = [':cc_lib_impl'],",
        ")");
    createLibraryTargetWriter("//objc2:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m")
        .setAndCreateFiles("hdrs", "c.h", "d.h")
        .setList("deps", "//cc:lib", "//third_party/cc_lib:cc_lib_impl")
        .write();
    ObjcProvider objcProvider = providerForTarget("//objc2:lib");
    
    Iterable<Artifact> linkerInputArtifacts = 
        Iterables.transform(objcProvider.get(CC_LIBRARY), new Function<LinkerInput, Artifact>() {
      @Override
      public Artifact apply(LinkerInput library) {
        return library.getArtifact();
      }
    });

    assertThat(linkerInputArtifacts)
        .containsAllOf(
            getBinArtifact(
                "liblib.a", getConfiguredTarget("//cc:lib", getAppleCrosstoolConfiguration())),
            getBinArtifact(
                "libcc_lib_impl.a",
                getConfiguredTarget(
                    "//third_party/cc_lib:cc_lib_impl", getAppleCrosstoolConfiguration())));
  }

  @Test
  public void testCollectsSdkFrameworksTransitively() throws Exception {
    createLibraryTargetWriter("//base_lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("sdk_frameworks", "foo")
        .write();
    createLibraryTargetWriter("//depender_lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("sdk_frameworks", "bar")
        .setList("deps", "//base_lib:lib")
        .write();

    ObjcProvider baseProvider = providerForTarget("//base_lib:lib");
    ObjcProvider dependerProvider = providerForTarget("//depender_lib:lib");

    Set<SdkFramework> baseFrameworks = ImmutableSet.of(new SdkFramework("foo"));
    Set<SdkFramework> dependerFrameworks =
        ImmutableSet.of(new SdkFramework("foo"), new SdkFramework("bar"));
    assertThat(baseProvider.get(SDK_FRAMEWORK)).containsExactlyElementsIn(baseFrameworks);
    assertThat(dependerProvider.get(SDK_FRAMEWORK)).containsExactlyElementsIn(dependerFrameworks);

    // Make sure that the archive action does not actually include the frameworks. This is needed
    // for creating binaries but is ignored for libraries.
    CommandAction archiveAction = archiveAction("//depender_lib:lib");
    assertThat(archiveAction.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .add(MOCK_LIBTOOL_PATH)
                .add("-static")
                .add("-filelist")
                .add(
                    getBinArtifact("lib-archive.objlist", "//depender_lib:lib").getExecPathString())
                .add("-arch_only", "x86_64")
                .add("-syslibroot")
                .add(AppleToolchain.sdkDir())
                .add("-o")
                .addAll(Artifact.toExecPaths(archiveAction.getOutputs()))
                .build());
  }

  @Test
  public void testMultipleRulesCompilingOneSourceGenerateUniqueObjFiles() throws Exception {
    scratch.file("lib/a.m");
    scratch.file("lib/BUILD",
        "objc_library(name = 'lib1', srcs = ['a.m'], copts = ['-Ilib1flag'])",
        "objc_library(name = 'lib2', srcs = ['a.m'], copts = ['-Ilib2flag'])");
    Artifact obj1 = Iterables.getOnlyElement(
        inputsEndingWith(archiveAction("//lib:lib1"), ".o"));
    Artifact obj2 = Iterables.getOnlyElement(
        inputsEndingWith(archiveAction("//lib:lib2"), ".o"));

    // The exec paths of each obj file should be based on the objc_library target.
    assertThat(obj1.getExecPathString()).contains("lib1");
    assertThat(obj1.getExecPathString()).doesNotContain("lib2");
    assertThat(obj2.getExecPathString()).doesNotContain("lib1");
    assertThat(obj2.getExecPathString()).contains("lib2");

    SpawnAction compile1 = (SpawnAction) getGeneratingAction(obj1);
    SpawnAction compile2 = (SpawnAction) getGeneratingAction(obj2);
    assertThat(compile1.getArguments()).contains("-Ilib1flag");
    assertThat(compile2.getArguments()).contains("-Ilib2flag");
  }

  @Test
  public void testIncludesDirsOfTransitiveDepsGetPassedToCompileAction() throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("includes", "third_party/foo", "opensource/bar")
        .write();

    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("includes", "more_includes")
        .setList("deps", "//lib1:lib1")
        .write();
    CommandAction compileAction = compileAction("//lib2:lib2", "a.o");
    assertContainsSublist(
        compileAction.getArguments(),
        ImmutableList.copyOf(
            Interspersing.beforeEach(
                "-I",
                rootedIncludePaths(
                    getAppleCrosstoolConfiguration(),
                    "lib2/more_includes",
                    "lib1/third_party/foo",
                    "lib1/opensource/bar"))));
  }

  @Test
  public void testIncludesDirsOfTransitiveCcDepsGetPassedToCompileAction() throws Exception {
    scratch.file("package/BUILD",
        "cc_library(",
        "    name = 'cc_lib',",
        "    srcs = ['a.cc'],",
        "    includes = ['foo/bar'],",
        ")",
        "",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['b.m'],",
        "    deps = [':cc_lib'],",
        ")");

    CommandAction compileAction = compileAction("//package:objc_lib", "b.o");
    assertContainsSublist(
        compileAction.getArguments(),
        ImmutableList.copyOf(
            Interspersing.beforeEach(
                "-isystem",
                rootedIncludePaths(getAppleCrosstoolConfiguration(), "package/foo/bar"))));
  }

  @Test
  public void testIncludesDirsOfTransitiveCcIncDepsGetPassedToCompileAction() throws Exception {
    scratch.file(
        "third_party/cc_lib/BUILD",
        "licenses(['unencumbered'])",
        "cc_library(",
        "    name = 'cc_lib_impl',",
        "    srcs = [",
        "        'v1/a.c',",
        "        'v1/a.h',",
        "    ],",
        ")",
        "",
        "cc_inc_library(",
        "    name = 'cc_lib',",
        "    hdrs = ['v1/a.h'],",
        "    prefix = 'v1',",
        "    deps = [':cc_lib_impl'],",
        ")");

    scratch.file(
        "package/BUILD",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['b.m'],",
        "    deps = ['//third_party/cc_lib:cc_lib'],",
        ")");

    CommandAction compileAction = compileAction("//package:objc_lib", "b.o");
    String includeDir =
        getAppleCrosstoolConfiguration()
                .getIncludeDirectory(RepositoryName.MAIN)
                .getExecPathString()
            + "/third_party/cc_lib/_/cc_lib";
    assertContainsSublist(compileAction.getArguments(), ImmutableList.of("-I", includeDir));
  }

  @Test
  public void testIncludesIquoteFlagForGenFilesRoot() throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .write();
    CommandAction compileAction = compileAction("//lib:lib", "a.o");
    BuildConfiguration config = getAppleCrosstoolConfiguration();
    assertContainsSublist(compileAction.getArguments(), ImmutableList.of(
        "-iquote", config.getGenfilesFragment().getSafePathString()));
  }

  @Test
  public void testProvidesHdrsAndIncludes() throws Exception {
    checkProvidesHdrsAndIncludes(RULE_TYPE);
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
  public void testSdkIncludesUsedInCompileActionsOfDependers() throws Exception {
    checkSdkIncludesUsedInCompileActionsOfDependers(RULE_TYPE);
  }

  @Test
  public void testCompilesAssemblyS() throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.s")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileAction = compileAction("//objc:lib", "b.o");

    assertThat(compileAction.getArguments()).doesNotContain("-x");
    assertThat(compileAction.getArguments()).doesNotContain("assembler-with-cpp");
    assertThat(baseArtifactNames(compileAction.getOutputs())).containsExactly("b.o", "b.d");
    assertThat(baseArtifactNames(compileAction.getInputs()))
        .containsExactly("c.h", "b.s", XCRUNWRAPPER);
  }

  @Test
  public void testCompilesAssemblyAsm() throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setAndCreateFiles("srcs", "a.m", "b.asm")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileAction = compileAction("//objc:lib", "b.o");

    assertThat(compileAction.getArguments()).doesNotContain("-x");
    assertThat(compileAction.getArguments()).doesNotContain("assembler-with-cpp");
    assertThat(baseArtifactNames(compileAction.getOutputs())).containsExactly("b.o", "b.d");
    assertThat(baseArtifactNames(compileAction.getInputs()))
        .containsExactly("c.h", "b.asm", XCRUNWRAPPER);
  }

  // Converts output artifacts into expected command-line arguments.
  private List<String> outputArgs(Set<Artifact> outputs) {
    ImmutableList.Builder<String> result = new ImmutableList.Builder<>();
    for (String output : Artifact.toExecPaths(outputs)) {
      if (output.endsWith(".o")) {
        result.add("-o", output);
      } else if (output.endsWith(".d")) {
        result.add("-MD", "-MF", output);
      } else {
        throw new IllegalArgumentException(
            "output " + output + " has unknown ending (not in (.d, .o)");
      }
    }
    return result.build();
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
