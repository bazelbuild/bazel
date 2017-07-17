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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
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
    MockObjcSupport.setupObjcProto(mockToolsConfig);
  }

  @Before
  public final void createFiles() throws Exception {
    scratch.file(
        "package/BUILD",
        "apple_binary(",
        "  name = 'opl_binary',",
        "  deps = [':opl_protobuf'],",
        "  platform_type = 'tvos'",
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
        "  name = 'opl',",
        "  deps = [':protolib'],",
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
        "objc_proto_library(",
        "  name = 'opl_well_known_types',",
        "  deps = [':protolib_well_known_types'],",
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
        "  deps = ['//objcproto:well_known_type_proto'],",
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
        "",
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
    NestedSet<Artifact> filesToBuild = getFilesToBuild(getConfiguredTarget("//package:opl"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/libopl.a",
            "package/_generated_protos/opl/package/FileA.pb.h",
            "package/_generated_protos/opl/package/FileA.pb.m",
            "package/_generated_protos/opl/package/dir/FileB.pb.h",
            "package/_generated_protos/opl/package/dir/FileB.pb.m",
            "package/_generated_protos/opl/dep/File.pb.h");
  }

  @Test
  public void testOutputsProtobuf() throws Exception {
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
  public void testOutputsWithAutoUnionExperiment() throws Exception {
    NestedSet<Artifact> filesToBuild = getFilesToBuild(getConfiguredTarget("//package:opl"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/libopl.a",
            "package/_generated_protos/opl/package/FileA.pb.h",
            "package/_generated_protos/opl/package/FileA.pb.m",
            "package/_generated_protos/opl/package/dir/FileB.pb.h",
            "package/_generated_protos/opl/package/dir/FileB.pb.m",
            "package/_generated_protos/opl/dep/File.pb.h");
  }

  @Test
  public void testDependingOnProtobufObjcProtoLibrary() throws Exception {
    NestedSet<Artifact> filesToBuild = getFilesToBuild(getConfiguredTarget("//package:nested_opl"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/nested_opl/package/FileA.pbobjc.h",
            "package/_generated_protos/nested_opl/package/FileA.pbobjc.m",
            "package/_generated_protos/nested_opl/package/dir/FileB.pbobjc.h",
            "package/_generated_protos/nested_opl/package/dir/FileB.pbobjc.m");
  }

  @Test
  public void testOutputsProtobufWithAutoUnionExperiment() throws Exception {
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
  public void testPB2GeneratedFileNames() throws Exception {
    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//package:opl_pb2_special_names"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/opl_pb2_special_names/package/J2ObjcDescriptor.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/J2ObjcDescriptor.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/Http.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/Http.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/Https.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/Https.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/SomeUrlBlah.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/SomeUrlBlah.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/ThumbnailUrl.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/ThumbnailUrl.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/Url.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/Url.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/Url2Https.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/Url2Https.pb.m",
            "package/_generated_protos/opl_pb2_special_names/package/Urlbar.pb.h",
            "package/_generated_protos/opl_pb2_special_names/package/Urlbar.pb.m");
  }

  @Test
  public void testProtobufGeneratedFileNames() throws Exception {
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
  public void testOutputsPB2WithWellKnownTypes() throws Exception {
    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//package:opl_well_known_types"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsAllOf(
            "package/_generated_protos/opl_well_known_types/package/FileA.pb.h",
            "package/_generated_protos/opl_well_known_types/package/FileA.pb.m");
    assertThat(Artifact.toRootRelativePaths(filesToBuild))
        .containsNoneOf(
            "package/_generated_protos/opl_well_known_types/objcproto/WellKnownType.pb.h",
            "package/_generated_protos/opl_well_known_types/objcproto/WellKnownType.pb.m");
  }

  @Test
  public void testOutputsProtobufWithWellKnownTypes() throws Exception {
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
            "package/libgen_opl.a",
            "package/_generated_protos/gen_opl/package/FileAGenfile.pb.h",
            "package/_generated_protos/gen_opl/package/FileAGenfile.pb.m");
  }

  @Test
  public void testSourceGenerationAction() throws Exception {
    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//package:opl")), "/FileA.pb.m");
    SpawnAction action = (SpawnAction) getGeneratingAction(sourceFile);

    Artifact inputFileList =
        ActionsTestUtil.getFirstArtifactEndingWith(action.getInputs(), "/_proto_input_files");

    ImmutableList<String> protoInputs =
        ImmutableList.of("dep/file.proto", "package/file_a.proto", "package/dir/file_b.proto");

    assertThat(action.getArguments())
        .containsExactly(
            "/usr/bin/python",
            "tools/objc/compile_protos.py",
            "--input-file-list",
            inputFileList.getExecPathString(),
            "--output-dir",
            // 2x parent directory because the package has one element ("package")
            sourceFile.getExecPath().getParentDirectory().getParentDirectory().toString(),
            "--working-dir", ".")
        .inOrder();
    assertRequiresDarwin(action);
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllOf(
        "tools/objc/compile_protos.py",
        "tools/objc/proto_support");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllIn(protoInputs);
    assertThat(action.getInputs()).contains(inputFileList);

    FileWriteAction inputListAction = (FileWriteAction) getGeneratingAction(inputFileList);
    assertThat(inputListAction.getFileContents()).isEqualTo(sortedJoin(protoInputs));
  }

  @Test
  public void testProtobufSourceGenerationAction() throws Exception {
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
            "tools/objc/protobuf_compiler_wrapper.sh",
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
            "tools/objc/protobuf_compiler_wrapper.sh",
            "tools/objc/protobuf_compiler_helper.py",
            "tools/objc/proto_support");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllIn(protoInputs);
    assertThat(action.getInputs()).contains(inputFileList);

    FileWriteAction inputListAction = (FileWriteAction) getGeneratingAction(inputFileList);
    assertThat(inputListAction.getFileContents()).isEqualTo(sortedJoin(protoInputs));
  }

  @Test
  public void testProtobufWithWellKnownTypesProtoListInput() throws Exception {
    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//package:opl_protobuf_well_known_types")),
            "/FileA.pbobjc.m");
    SpawnAction action = (SpawnAction) getGeneratingAction(sourceFile);

    Artifact inputFileList =
        ActionsTestUtil.getFirstArtifactEndingWith(
            action.getInputs(), "/_proto_input_files_BundledProtos_0");

    ImmutableList<String> protoInputs = ImmutableList.of("package/file_a.proto");

    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllIn(protoInputs);
    assertThat(action.getInputs()).contains(inputFileList);

    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .contains("objcproto/well_known_type.proto");

    FileWriteAction inputListAction = (FileWriteAction) getGeneratingAction(inputFileList);
    assertThat(inputListAction.getFileContents()).isEqualTo(sortedJoin(protoInputs));
  }

  @Test
  public void testUseObjcHeaders() throws Exception {
     scratch.file("objcheaderpackage/BUILD",
        "objc_proto_library(",
        "  name = 'opl',",
        "  deps = ['//package:protolib'],",
        "  use_objc_header_names = 1,",
        ")");

    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//objcheaderpackage:opl")), "/FileA.pb.m");
    SpawnAction action = (SpawnAction) getGeneratingAction(sourceFile);

    assertThat(action.getArguments()).contains("--use-objc-header-names");

    NestedSet<Artifact> filesToBuild =
        getFilesToBuild(getConfiguredTarget("//objcheaderpackage:opl"));
    assertThat(Artifact.toRootRelativePaths(filesToBuild)).containsAllOf(
        "objcheaderpackage/_generated_protos/opl/package/FileA.pbobjc.h",
        "objcheaderpackage/_generated_protos/opl/package/FileA.pb.m",
        "objcheaderpackage/_generated_protos/opl/package/dir/FileB.pbobjc.h",
        "objcheaderpackage/_generated_protos/opl/package/dir/FileB.pb.m"
    );
  }

  @Test
  public void testProtobufCompilationAction() throws Exception {
    useConfiguration("--ios_cpu=i386");

    ConfiguredTarget target = getConfiguredTarget("//package:opl_protobuf");
    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "/FileA.pbobjc.m");
    SpawnAction generateAction = (SpawnAction) getGeneratingAction(sourceFile);

    assertThat(generateAction).isNotNull();
  }


  @Test
  public void testProtobufObjcProviderWithAutoUnionExperiment() throws Exception {
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
  public void testPerProtoIncludes() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//package:opl");
    Artifact headerFile = ActionsTestUtil.getFirstArtifactEndingWith(
        getFilesToBuild(target), "/FileA.pb.h");

    ObjcProvider provider = providerForTarget("//package:opl");
    assertThat(provider.get(ObjcProvider.INCLUDE).toSet()).containsExactly(
        // 2x parent directory because the package has one element ("package")
        headerFile.getExecPath().getParentDirectory().getParentDirectory()
    );
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

  // This is a test for deprecated functionality.
  @Test
  public void testErrorForDepWithFilegroupWithoutProtoFiles() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.NO_PROTOS_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    deps = [':fg'],",
        ")",
        "filegroup(",
        "    name = 'fg',",
        "    srcs = ['file.dat'],",
        ")");
  }

  @Test
  public void testWarningForProtoSourceDeps() throws Exception {
    checkWarning(
        "x",
        "x",
        ProtoAttributes.FILES_DEPRECATED_WARNING,
        "objc_proto_library(",
        "    name = 'x',",
        "    deps = ['foo.proto'],",
        ")");
  }

  @Test
  public void testWarningForFilegroupDeps() throws Exception {
    checkWarning(
        "x",
        "x",
        ProtoAttributes.FILES_DEPRECATED_WARNING,
        "filegroup(",
        "    name = 'protos',",
        "    srcs = ['foo.proto'],",
        ")",
        "objc_proto_library(",
        "    name = 'x',",
        "    deps = [':protos'],",
        ")");
  }

  @Test
  public void testObjcCopts() throws Exception {
    useConfiguration("--objccopt=-foo");

    List<String> args = compileAction("//package:opl", "FileA.pb.o").getArguments();
    assertThat(args).contains("-foo");
  }

  @Test
  public void testObjcCopts_argumentOrdering() throws Exception {
    useConfiguration("--objccopt=-foo");

    List<String> args = compileAction("//package:opl", "FileA.pb.o").getArguments();
    assertThat(args).containsAllOf("-fno-objc-arc", "-foo").inOrder();
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
  public void testErrorWhenDependingOnPB2FromProtobufTarget() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.PROTOCOL_BUFFERS2_IN_PROTOBUF_DEPS_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    portable_proto_filters = ['filter.pbascii'],",
        "    deps = [':pb2', ':protos_b'],",
        ")",
        "objc_proto_library(",
        "    name = 'pb2',",
        "    deps = [':protos_a'],",
        ")",
        "proto_library(",
        "    name = 'protos_a',",
        "    srcs = ['a.proto'],",
        ")",
        "proto_library(",
        "    name = 'protos_b',",
        "    srcs = ['b.proto'],",
        ")");
  }

  @Test
  public void testErrorWhenDependingOnPB2FromPB2Target() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.OBJC_PROTO_LIB_DEP_IN_PROTOCOL_BUFFERS2_DEPS_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    deps = [':pb2', ':protos_b'],",
        ")",
        "objc_proto_library(",
        "    name = 'pb2',",
        "    deps = [':protos_a'],",
        ")",
        "proto_library(",
        "    name = 'protos_a',",
        "    srcs = ['a.proto'],",
        ")",
        "proto_library(",
        "    name = 'protos_b',",
        "    srcs = ['b.proto'],",
        ")");
  }

  @Test
  public void testErrorForPortableProtoFiltersWithUseObjcHeaderNames() throws Exception {
    checkErrorForPortableProtoFilterWithPb2Option("use_objc_header_names = 1");
  }

  @Test
  public void testErrorForPortableProtoFiltersWithPerProtoIncludes() throws Exception {
    checkErrorForPortableProtoFilterWithPb2Option("per_proto_includes = 1");
  }

  @Test
  public void testErrorForPortableProtoFiltersWithOptionsFile() throws Exception {
    checkErrorForPortableProtoFilterWithPb2Option("options_file = 'options_file.txt'");
  }

  @Test
  public void testErrorForUsesProtobufWithUseObjcHeaderNames() throws Exception {
    checkErrorForUsesProtobufWithPb2Option("use_objc_header_names = 1");
  }

  @Test
  public void testErrorForUsesProtobufWithPerProtoIncludes() throws Exception {
    checkErrorForUsesProtobufWithPb2Option("per_proto_includes = 1");
  }

  @Test
  public void testErrorForUsesProtobufWithOptionsFile() throws Exception {
    checkErrorForUsesProtobufWithPb2Option("options_file = 'options_file.txt'");
  }

  @Test
  public void testModulemapCreatedForNonLinkingTargets() throws Exception {
    checkOnlyLibModuleMapsArePresentForTarget("//package:opl_protobuf");
  }

  @Test
  public void testModulemapNotCreatedForLinkingTargets() throws Exception {
    checkOnlyLibModuleMapsArePresentForTarget("//package:opl_binary");
  }

  @Test
  public void testErrorForPortableProtoFilterFilesInDeps() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.FILES_NOT_ALLOWED_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    portable_proto_filters = ['proto_filter.txt'],",
        "    deps = [':protos'],",
        ")",
        "filegroup(",
        "    name = 'protos',",
        "    srcs = ['file.proto'],",
        ")");
  }

  @Test
  public void testErrorForUsesProtobufAsFalseWithFilters() throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.USES_PROTOBUF_CANT_BE_SPECIFIED_AS_FALSE,
        "objc_proto_library(",
        "    name = 'x',",
        "    uses_protobuf = 0,",
        "    portable_proto_filters = ['myfilter.pbascii'],",
        "    deps = [':protos'],",
        ")",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = ['file.proto'],",
        ")");
  }

  private void checkErrorForPortableProtoFilterWithPb2Option(String pb2Option) throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.PROTOBUF_ATTRIBUTES_NOT_EXCLUSIVE_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    portable_proto_filters = ['proto_filter.txt'],",
        "    " + pb2Option + ",",
        "    deps = [':protos'],",
        ")",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = ['file.proto'],",
        ")");
  }

  private void checkErrorForUsesProtobufWithPb2Option(String pb2Option) throws Exception {
    checkError(
        "x",
        "x",
        ProtoAttributes.PROTOBUF_ATTRIBUTES_NOT_EXCLUSIVE_ERROR,
        "objc_proto_library(",
        "    name = 'x',",
        "    uses_protobuf = 1,",
        "    " + pb2Option + ",",
        "    deps = [':protos'],",
        ")",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = ['file.proto'],",
        ")");
  }

  private static String sortedJoin(Iterable<String> elements) {
    return Joiner.on('\n').join(Ordering.natural().immutableSortedCopy(elements));
  }

  private void checkOnlyLibModuleMapsArePresentForTarget(String target) throws Exception {
    Artifact libModuleMap =
        getGenfilesArtifact(
            "opl_protobuf.modulemaps/module.modulemap",
            getConfiguredTarget("//package:opl_protobuf"));
    Artifact protolibModuleMap =
        getGenfilesArtifact(
            "protobuf_lib.modulemaps/module.modulemap",
            getConfiguredTarget("//objcproto:protobuf_lib"));

    ObjcProvider provider = providerForTarget(target);
    assertThat(Artifact.toRootRelativePaths(provider.get(ObjcProvider.MODULE_MAP).toSet()))
        .containsExactlyElementsIn(
            Artifact.toRootRelativePaths(ImmutableSet.of(libModuleMap, protolibModuleMap)));
  }

  @Test
  public void testObjcProvider() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//package:opl");
    Artifact headerFile =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "/FileA.pb.h");
    Artifact sourceFile =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "/FileA.pb.m");

    ObjcProvider provider = providerForTarget("//package:opl");
    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());

    ConfiguredTarget libProtoBufTarget = getConfiguredTarget("//objcproto:ProtocolBuffers_lib");
    assertThat(provider.get(ObjcProvider.LIBRARY).toSet())
        .containsExactly(
            getBinArtifact("libopl.a", target),
            getBinArtifact("libProtocolBuffers_lib.a", libProtoBufTarget));

    assertThat(provider.get(ObjcProvider.HEADER).toSet()).containsAllOf(headerFile, sourceFile);

    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());
  }

  @Test
  public void testProtobufObjcProvider() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//package:opl_protobuf");
    Artifact headerFile =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), "/FileA.pbobjc.h");
    ObjcProvider provider = providerForTarget("//package:opl_protobuf");
    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());

    ConfiguredTarget libProtoBufTarget = getConfiguredTarget("//objcproto:protobuf_lib");
    assertThat(provider.get(ObjcProvider.LIBRARY).toSet())
        .containsExactly(getBinArtifact("libprotobuf_lib.a", libProtoBufTarget));

    assertThat(provider.get(ObjcProvider.HEADER).toSet()).contains(headerFile);

    assertThat(provider.get(ObjcProvider.INCLUDE).toSet())
        .contains(headerFile.getExecPath().getParentDirectory().getParentDirectory());
  }

  @Test
  public void testCompilationActionInCoverageMode() throws Exception {
    useConfiguration("--collect_code_coverage");

    ConfiguredTarget target = getConfiguredTarget("//package:opl");
    CommandAction linkAction =
        (CommandAction) getGeneratingAction(getBinArtifact("libopl.a", target));

    CommandAction compileAction =
        (CommandAction)
            getGeneratingAction(
                ActionsTestUtil.getFirstArtifactEndingWith(linkAction.getInputs(), "/FileA.pb.o"));

    assertThat(Artifact.toRootRelativePaths(compileAction.getOutputs()))
        .containsAllOf(
            "package/_objs/opl/package/_generated_protos/opl/package/FileA.pb.o",
            "package/_objs/opl/package/_generated_protos/opl/package/FileA.pb.gcno");
  }

  @Test
  public void testModuleMapActionFiltersHeaders() throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget("//package:opl");
    Artifact moduleMap = getGenfilesArtifact("opl.modulemaps/module.modulemap", configuredTarget);

    CppModuleMapAction genMap = (CppModuleMapAction) getGeneratingAction(moduleMap);
    assertThat(Artifact.toRootRelativePaths(genMap.getPrivateHeaders())).isEmpty();
    assertThat(Artifact.toRootRelativePaths(genMap.getPublicHeaders()))
        .containsExactly(
            "package/_generated_protos/opl/package/FileA.pb.h",
            "package/_generated_protos/opl/package/dir/FileB.pb.h",
            "package/_generated_protos/opl/dep/File.pb.h");
  }
}
