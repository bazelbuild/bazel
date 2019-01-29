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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_proto_library. */
@RunWith(JUnit4.class)
public class ObjcProtoLibraryTest extends ObjcRuleTestCase {

  @Before
  public final void initializeToolsConfigMock() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    MockObjcSupport.setup(mockToolsConfig);
  }

  @Before
  public final void createFiles() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_binary(",
        "  name = 'opl_binary',",
        "  deps = [':opl_protobuf'],",
        "  platform_type = 'ios'",
        ")",
        "objc_library(",
        "  name = 'non_strict_lib',",
        "  deps = [':strict_lib'],",
        ")",
        "",
        "objc_library(",
        "  name = 'strict_lib',",
        "  deps = [':opl_protobuf'],",
        ")",
        "",
        "objc_proto_library(",
        "  name = 'nested_opl',",
        "  deps = [':opl_protobuf'],",
        "  portable_proto_filters = ['nested_filter.txt'],",
        ")",
        "",
        "objc_proto_library(",
        "  name = 'opl_protobuf',",
        "  deps = [':protolib'],",
        "  portable_proto_filters = [",
        "    'proto_filter.txt',",
        "    ':portable_proto_filters',",
        "  ],",
        ")",
        "",
        "objc_proto_library(",
        "  name = 'opl_protobuf_well_known_types',",
        "  deps = [':protolib_well_known_types'],",
        "  portable_proto_filters = [",
        "    'proto_filter.txt',",
        "  ],",
        ")",
        "",
        "filegroup(",
        "  name = 'portable_proto_filters',",
        "  srcs = [",
        "    'proto_filter2.txt',",
        "    'proto_filter3.txt',",
        "  ],",
        ")",
        "",
        "proto_library(",
        "  name = 'protolib',",
        "  srcs = ['file_a.proto', 'dir/file_b.proto'],",
        "  deps = ['//dep:dep_lib'],",
        ")",
        "",
        "objc_proto_library(",
        "  name = 'opl_protobuf_special_names',",
        "  deps = [':protolib_special_names'],",
        "  portable_proto_filters = [",
        "    'proto_filter.txt',",
        "  ],",
        ")",
        "objc_proto_library(",
        "  name = 'opl_pb2_special_names',",
        "  deps = [':protolib_special_names'],",
        ")",
        "",
        "proto_library(",
        "  name = 'protolib_special_names',",
        "  srcs = [",
        "    'j2objc-descriptor.proto',",
        "    'http.proto',",
        "    'https.proto',",
        "    'some_url_blah.proto',",
        "    'thumbnail_url.proto',",
        "    'url.proto',",
        "    'url2https.proto',",
        "    'urlbar.proto',",
        "  ],",
        "  deps = ['//dep:dep_lib'],",
        ")",
        "",
        "proto_library(",
        "  name = 'protolib_well_known_types',",
        "  srcs = ['file_a.proto'],",
        "  deps = ['" + TestConstants.TOOLS_REPOSITORY + "//objcproto:well_known_type_proto'],",
        ")",
        "",
        "genrule(",
        "  name = 'gen_proto',",
        "  srcs = ['file_a.proto'],",
        "  outs = ['file_a_genfile.proto'],",
        "  cmd  = 'cp $(location file_a.proto) $(location file_a_genfile.proto)')",
        "",
        "proto_library(",
        "  name = 'gen_protolib',",
        "  srcs = ['file_a_genfile.proto'],",
        "  deps = ['//dep:dep_lib'],",
        ")",
        "objc_proto_library(",
        "  name = 'gen_opl',",
        "  deps = [':gen_protolib'],",
        ")");
    scratch.file("dep/BUILD",
        "proto_library(",
        "  name = 'dep_lib',",
        "  srcs = ['file.proto'],",
        ")");
    scratch.file("package/file_a.proto");
    scratch.file("package/dir/file_b.proto");
    scratch.file("dep/file.proto");
    scratch.file("package/proto_filter.txt");
    scratch.file("package/proto_filter2.txt");
    scratch.file("package/proto_filter3.txt");
  }

  @Test
  public void testOutputs() throws Exception {
    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//package:opl_protobuf"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/opl_protobuf/package/FileA.pbobjc.h",
            "package/_generated_protos/opl_protobuf/package/FileA.pbobjc.m",
            "package/_generated_protos/opl_protobuf/package/dir/FileB.pbobjc.h",
            "package/_generated_protos/opl_protobuf/package/dir/FileB.pbobjc.m",
            "package/_generated_protos/opl_protobuf/dep/File.pbobjc.h");
  }

  @Test
  public void testDependingObjcProtoLibrary() throws Exception {
    NestedSet<Artifact> filesToBuild = getFilesToBuild(getConfiguredTarget("//package:nested_opl"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/nested_opl/package/FileA.pbobjc.h",
            "package/_generated_protos/nested_opl/package/FileA.pbobjc.m",
            "package/_generated_protos/nested_opl/package/dir/FileB.pbobjc.h",
            "package/_generated_protos/nested_opl/package/dir/FileB.pbobjc.m");
  }

  @Test
  public void testOutputsWithAutoUnion() throws Exception {
    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//package:opl_protobuf"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .doesNotContain("package/libopl_protobuf.a");
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/opl_protobuf/package/FileA.pbobjc.h",
            "package/_generated_protos/opl_protobuf/package/FileA.pbobjc.m",
            "package/_generated_protos/opl_protobuf/package/dir/FileB.pbobjc.h",
            "package/_generated_protos/opl_protobuf/package/dir/FileB.pbobjc.m",
            "package/_generated_protos/opl_protobuf/dep/File.pbobjc.h");
  }

  @Test
  public void testGeneratedFileNames() throws Exception {
    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//package:opl_protobuf_special_names"));
    String outputPath = "package/_generated_protos/opl_protobuf_special_names/package/";
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            outputPath + "J2ObjcDescriptor.pbobjc.h",
            outputPath + "J2ObjcDescriptor.pbobjc.m",
            outputPath + "HTTP.pbobjc.h",
            outputPath + "HTTP.pbobjc.m",
            outputPath + "HTTPS.pbobjc.h",
            outputPath + "HTTPS.pbobjc.m",
            outputPath + "SomeURLBlah.pbobjc.h",
            outputPath + "SomeURLBlah.pbobjc.m",
            outputPath + "ThumbnailURL.pbobjc.h",
            outputPath + "ThumbnailURL.pbobjc.m",
            outputPath + "URL.pbobjc.h",
            outputPath + "URL.pbobjc.m",
            outputPath + "URL2HTTPS.pbobjc.h",
            outputPath + "URL2HTTPS.pbobjc.m",
            outputPath + "Urlbar.pbobjc.h",
            outputPath + "Urlbar.pbobjc.m");
  }

  @Test
  public void testOutputsWithWellKnownTypes() throws Exception {
    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//package:opl_protobuf_well_known_types"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/opl_protobuf_well_known_types/package/FileA.pbobjc.h",
            "package/_generated_protos/opl_protobuf_well_known_types/package/FileA.pbobjc.m");
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .doesNotContain(
            "package/_generated_protos/opl_protobuf_well_known_types/objcproto/WellKnownType.pbobjc.h");
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .doesNotContain(
            "package/_generated_protos/opl_protobuf_well_known_types/objcproto/WellKnownType.pbobjc.m");
  }

  @Test
  public void testOutputsGenfile() throws Exception {
    NestedSet<Artifact> filesToBuild = getFilesToBuild(getConfiguredTarget("//package:gen_opl"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/gen_opl/package/FileAGenfile.pbobjc.h",
            "package/_generated_protos/gen_opl/package/FileAGenfile.pbobjc.m");
  }

  @Test
  public void testSourceGenerationAction() throws Exception {
    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//package:opl_protobuf")), "/FileA.pbobjc.m");
    SpawnAction action = (SpawnAction) getGeneratingAction(sourceFile);

    Artifact inputFileList =
        ActionsTestUtil.getFirstArtifactEndingWith(
            action.getInputs(), "/_proto_input_files_BundledProtos_0");

    ImmutableList<String> protoInputs =
        ImmutableList.of("dep/file.proto", "package/file_a.proto", "package/dir/file_b.proto");

    BuildConfiguration topLevelConfig = getAppleCrosstoolConfiguration();
    assertThat(action.getArguments())
        .containsExactly(
            TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + "tools/objc/protobuf_compiler_wrapper.sh",
            "--input-file-list",
            inputFileList.getExecPathString(),
            "--output-dir",
            // 2x parent directory because the package has one element ("package")
            sourceFile.getExecPath().getParentDirectory().getParentDirectory().toString(),
            "--force",
            "--proto-root-dir",
            topLevelConfig.getGenfilesFragment().toString(),
            "--proto-root-dir",
            ".",
            "--config",
            "package/proto_filter.txt",
            "--config",
            "package/proto_filter2.txt",
            "--config",
            "package/proto_filter3.txt")
        .inOrder();
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsAllOf(
            TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + "tools/objc/protobuf_compiler_wrapper.sh",
            TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + "tools/objc/protobuf_compiler_helper.py",
            TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + "tools/objc/proto_support");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllIn(protoInputs);
    assertThat(action.getInputs()).contains(inputFileList);

    FileWriteAction inputListAction = (FileWriteAction) getGeneratingAction(inputFileList);
    assertThat(inputListAction.getFileContents()).isEqualTo(sortedJoin(protoInputs));
  }

  @Test
  public void testWellKnownTypesProtoListInput() throws Exception {
    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//package:opl_protobuf_well_known_types")),
            "/FileA.pbobjc.m");
    SpawnAction action = (SpawnAction) getGeneratingAction(sourceFile);

    Artifact inputFileList =
        ActionsTestUtil.getFirstArtifactEndingWith(
            action.getInputs(), "/_proto_input_files_BundledProtos_0");

    ImmutableList<String> protoInputs = ImmutableList.of(
        "package/file_a.proto",
        TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + "objcproto/well_known_type.proto");

    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllIn(protoInputs);
    assertThat(action.getInputs()).contains(inputFileList);

    FileWriteAction inputListAction = (FileWriteAction) getGeneratingAction(inputFileList);
    assertThat(inputListAction.getFileContents()).contains("package/file_a.proto");
  }

  @Test
  public void testObjcProviderWithAutoUnion() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//package:opl_protobuf");
    Artifact headerFile =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "/FileA.pbobjc.h");

    ObjcProvider provider = providerForTarget("//package:opl_protobuf");
    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());

    assertThat(provider.get(ObjcProvider.LIBRARY).toSet())
        .doesNotContain(getBinArtifact("libopl_protobuf.a", target));

    assertThat(provider.get(ObjcProvider.HEADER).toSet()).contains(headerFile);
  }

  @Test
  public void testErrorForNoDepsAttribute() throws Exception {
    checkError(
        "x", "x", ProtoAttributes.NO_PROTOS_ERROR, "objc_proto_library(", "    name = 'x',", ")");
  }

  @Test
  public void testErrorForEmptyDepsAttribute() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.NO_PROTOS_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    deps = [],",
        ")");
  }

  @Test
  public void testErrorForFileInDeps() throws Exception {
    String expectedError =
        "filegroup rule '//x:protos' is misplaced here "
            + "(expected proto_library or objc_proto_library)";
    checkError(
        "x",
        "x",
        expectedError,
        "objc_proto_library(",
        "    name = 'x',",
        "    deps = [':protos'],",
        ")",
        "filegroup(",
        "    name = 'protos',",
        "    srcs = ['file.proto'],",
        ")");
  }

  @Test
  public void testErrorForPortableProtoFiltersEmpty() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.PORTABLE_PROTO_FILTERS_EMPTY_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    portable_proto_filters = [],",
        "    deps = [':protos'],",
        ")",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = ['file.proto'],",
        ")");
  }

  @Test
  public void testModulemapCreatedForNonLinkingTargets() throws Exception {
    // TODO(b/73943026): Remove this flag once everyone has migrated to the new strict behavior and
    // it is made the default.
    useConfiguration("--incompatible_strict_objc_module_maps");

    // The library target should propagate its module map.
    ObjcProvider provider = providerForTarget("//package:opl_protobuf");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.MODULE_MAP).toSet()))
        .containsExactly("package/opl_protobuf.modulemaps/module.modulemap");
  }

  @Test
  public void testModulemapNotCreatedForLinkingTargets() throws Exception {
    // TODO(b/73943026): Remove this flag once everyone has migrated to the new strict behavior and
    // it is made the default.
    useConfiguration("--incompatible_strict_objc_module_maps");

    // The binary target should not propagate the module map from the library it depends on.
    ObjcProvider provider = providerForTarget("//package:opl_binary");
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.MODULE_MAP).toSet()))
        .isEmpty();
  }

  private static String sortedJoin(Iterable<String> elements) {
    return Joiner.on('\n').join(Ordering.natural().immutableSortedCopy(elements));
  }

  @Test
  public void testObjcProvider() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//package:opl_protobuf");
    Artifact headerFile =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "/FileA.pbobjc.h");
    ObjcProvider provider = providerForTarget("//package:opl_protobuf");
    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());

    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.LIBRARY).toSet()))
        .containsExactly(TestConstants.TOOLS_REPOSITORY_PATH_PREFIX
            + "objcproto/libprotobuf_lib.a");

    assertThat(provider.get(ObjcProvider.HEADER).toSet()).contains(headerFile);

    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());
  }

  @Test
  public void testModuleMapActionFiltersHeaders() throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget("//package:opl_protobuf");
    Artifact moduleMap =
        getGenfilesArtifact("opl_protobuf.modulemaps/module.modulemap", configuredTarget);

    CppModuleMapAction genMap = (CppModuleMapAction) getGeneratingAction(moduleMap);
    assertThat(Artifact.toRootRelativePaths(genMap.getPrivateHeaders())).isEmpty();
    assertThat(Artifact.toRootRelativePaths(genMap.getPublicHeaders()))
        .containsExactly(
            "package/_generated_protos/opl_protobuf/package/FileA.pbobjc.h",
            "package/_generated_protos/opl_protobuf/package/dir/FileB.pbobjc.h",
            "package/_generated_protos/opl_protobuf/dep/File.pbobjc.h");
  }

  @Test
  public void testCompilationAction() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_i386");
    ApplePlatform platform = ApplePlatform.IOS_SIMULATOR;

    // Because protos are linked/compiled within the apple_binary context, we need to traverse the
    // action graph to find the linked protos (.a) and compiled protos (.o).
    ConfiguredTarget binaryTarget = getConfiguredTarget("//package:opl_binary");
    SymlinkAction symlinkAction =
        (SymlinkAction) getGeneratingAction(getBinArtifact("opl_binary_lipobin", binaryTarget));

    Artifact binaryInput = Iterables.getOnlyElement(symlinkAction.getInputs());

    CommandAction linkAction = (CommandAction) getGeneratingAction(binaryInput);

    Artifact linkedProtos =
        ActionsTestUtil.getFirstArtifactEndingWith(
            linkAction.getInputs(), "libopl_binary_BundledProtos.a");
    CommandAction linkedProtosAction = (CommandAction) getGeneratingAction(linkedProtos);

    Artifact objectFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            linkedProtosAction.getInputs(), "FileA.pbobjc.o");
    CommandAction compiledProtoAction = (CommandAction) getGeneratingAction(objectFile);

    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            compiledProtoAction.getInputs(), "/FileA.pbobjc.m");
    Artifact dotdFile =
        ActionsTestUtil.getFirstArtifactEndingWith(compiledProtoAction.getOutputs(), ".d");

    // We remove spaces since the crosstool rules do not use spaces in command line args.
    String compileArgs = Joiner.on("").join(compiledProtoAction.getArguments()).replace(" ", "");

    List<String> expectedArgs =
        new ImmutableList.Builder<String>()
            .addAll(AppleToolchain.DEFAULT_WARNINGS.values())
            .add("-fexceptions")
            .add("-fasm-blocks")
            .add("-fobjc-abi-version=2")
            .add("-fobjc-legacy-dispatch")
            .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
            .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
            .add("-arch", "i386")
            .add("-isysroot", AppleToolchain.sdkDir())
            .add("-F", AppleToolchain.sdkDir() + AppleToolchain.DEVELOPER_FRAMEWORK_PATH)
            .add("-F", frameworkDir(platform))
            .addAll(FASTBUILD_COPTS)
            .addAll(
                ObjcLibraryTest.iquoteArgs(
                    providerForTarget("//package:opl_binary"),
                    getAppleCrosstoolConfiguration()))
            .add("-I")
            .add(sourceFile.getExecPath().getParentDirectory().getParentDirectory().toString())
            .add("-fno-objc-arc")
            .add("-c", sourceFile.getExecPathString())
            .add("-o")
            .add(objectFile.getExecPathString())
            .add("-MD")
            .add("-MF")
            .add(dotdFile.getExecPathString())
            .build();

    for (String expectedArg : expectedArgs) {
      assertThat(compileArgs).contains(expectedArg);
    }

    assertRequiresDarwin(compiledProtoAction);
    assertThat(Artifact.toRootRelativePaths(compiledProtoAction.getInputs()))
        .containsAllOf(
            "package/_generated_protos/opl_binary/package/FileA.pbobjc.m",
            "package/_generated_protos/opl_binary/package/FileA.pbobjc.h",
            "package/_generated_protos/opl_binary/package/dir/FileB.pbobjc.h",
            "package/_generated_protos/opl_binary/dep/File.pbobjc.h");
  }

  @Test
  public void testLibraryLinkAction() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_armv7");

    // Because protos are linked within the apple_binary context, we need to traverse the action
    // graph to find the linked protos (.a).
    ConfiguredTarget binaryTarget = getConfiguredTarget("//package:opl_binary");
    SymlinkAction symlinkAction =
        (SymlinkAction) getGeneratingAction(getBinArtifact("opl_binary_lipobin", binaryTarget));

    Artifact binaryInput = Iterables.getOnlyElement(symlinkAction.getInputs());

    CommandAction linkAction = (CommandAction) getGeneratingAction(binaryInput);

    Artifact linkedProtos =
        ActionsTestUtil.getFirstArtifactEndingWith(
            linkAction.getInputs(), "libopl_binary_BundledProtos.a");
    CommandAction linkedProtosAction = (CommandAction) getGeneratingAction(linkedProtos);
    Artifact objListFile =
        ActionsTestUtil.getFirstArtifactEndingWith(linkedProtosAction.getInputs(), ".objlist");
    assertThat(linkedProtosAction.getArguments())
        .containsAllIn(
            ImmutableList.of(
                "-static",
                "-filelist",
                objListFile.getExecPathString(),
                "-arch_only",
                "armv7",
                "-syslibroot",
                AppleToolchain.sdkDir(),
                "-o",
                linkedProtos.getExecPathString()));
    assertRequiresDarwin(linkedProtosAction);
  }
}
