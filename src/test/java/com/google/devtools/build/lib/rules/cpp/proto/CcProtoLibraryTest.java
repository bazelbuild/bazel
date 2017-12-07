// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp.proto;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CcProtoLibraryTest extends BuildViewTestCase {
  @Before
  public void setUp() throws Exception {
    mockToolsConfig.create("/protobuf/WORKSPACE");
    mockToolsConfig.overwrite(
        "/protobuf/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "exports_files(['protoc'])",
        "proto_lang_toolchain(",
        "    name = 'cc_toolchain',",
        "    command_line = '--cpp_out=$(OUT)',",
        "    blacklisted_protos = [],",
        ")");

    String existingWorkspace =
        new String(FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    mockToolsConfig.overwrite(
        "WORKSPACE",
        "local_repository(name = 'com_google_protobuf', path = '/protobuf/')",
        existingWorkspace);
    invalidatePackages(); // A dash of magic to re-evaluate the WORKSPACE file.
  }

  @Test
  public void basic() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_proto_library(name = 'foo_cc_proto', deps = ['foo_proto'])",
        "proto_library(name = 'foo_proto', srcs = ['foo.proto'])");
    assertThat(prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//x:foo_cc_proto"))))
        .containsExactly("x/foo.pb.h", "x/foo.pb.cc", "x/libfoo_proto.a", "x/libfoo_proto.so");
  }

  @Test
  public void canBeUsedFromCcRules() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_library(name = 'foo', srcs = ['foo.cc'], deps = ['foo_cc_proto'])",
        "cc_binary(name = 'bin', srcs = ['bin.cc'], deps = ['foo_cc_proto'])",
        "cc_proto_library(name = 'foo_cc_proto', deps = ['foo_proto'])",
        "proto_library(name = 'foo_proto', srcs = ['foo.proto'])");

    update(
        ImmutableList.of("//x:foo", "//x:bin"),
        false /* keepGoing */,
        1 /* loadingPhaseThreads */,
        true /* doAnalysis */,
        new EventBus());
  }

  @Test
  public void disallowMultipleDeps() throws Exception {
    checkError(
        "x",
        "foo_cc_proto",
        "'deps' attribute must contain exactly one label",
        "cc_proto_library(name = 'foo_cc_proto', deps = ['foo_proto', 'bar_proto'])",
        "proto_library(name = 'foo_proto', srcs = ['foo.proto'])",
        "proto_library(name = 'bar_proto', srcs = ['bar.proto'])");

    checkError(
        "y",
        "foo_cc_proto",
        "'deps' attribute must contain exactly one label",
        "cc_proto_library(name = 'foo_cc_proto', deps = [])");
  }

  @Test
  public void aliasProtos() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_proto_library(name = 'foo_cc_proto', deps = ['alias_proto'])",
        "proto_library(name = 'alias_proto', deps = [':foo_proto'])",
        "proto_library(name = 'foo_proto', srcs = ['foo.proto'])");

    CppCompilationContext context =
        getConfiguredTarget("//x:foo_cc_proto").getProvider(CppCompilationContext.class);
    assertThat(prettyArtifactNames(context.getDeclaredIncludeSrcs())).containsExactly("x/foo.pb.h");
  }

  @Test
  public void cppCompilationContext() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_proto_library(name = 'foo_cc_proto', deps = ['foo_proto'])",
        "proto_library(name = 'foo_proto', srcs = ['foo.proto'], deps = [':bar_proto'])",
        "proto_library(name = 'bar_proto', srcs = ['bar.proto'])");

    CppCompilationContext context =
        getConfiguredTarget("//x:foo_cc_proto").getProvider(CppCompilationContext.class);
    assertThat(prettyArtifactNames(context.getDeclaredIncludeSrcs()))
        .containsExactly("x/foo.pb.h", "x/bar.pb.h");
  }

  @Test
  public void outputDirectoryForProtoCompileAction() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_proto_library(name = 'foo_cc_proto', deps = [':bar_proto'])",
        "proto_library(name = 'bar_proto', srcs = ['bar.proto'])");

    Artifact hFile =
        getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//x:foo_cc_proto")), "bar.pb.h");
    SpawnAction protoCompileAction = getGeneratingSpawnAction(hFile);

    assertThat(protoCompileAction.getArguments())
        .contains(
            String.format(
                "--cpp_out=%s", getTargetConfiguration().getGenfilesFragment().toString()));
  }

  @Test
  public void outputDirectoryForProtoCompileAction_externalRepos() throws Exception {
    scratch.file(
        "x/BUILD", "cc_proto_library(name = 'foo_cc_proto', deps = ['@bla//foo:bar_proto'])");

    scratch.file("/bla/WORKSPACE");
    // Create the rule '@bla//foo:bar_proto'.
    scratch.file(
        "/bla/foo/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_library(name = 'bar_proto', srcs = ['bar.proto'])");
    String existingWorkspace =
        new String(FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    scratch.overwriteFile(
        "WORKSPACE", "local_repository(name = 'bla', path = '/bla/')", existingWorkspace);
    invalidatePackages(); // A dash of magic to re-evaluate the WORKSPACE file.

    Artifact hFile =
        getFirstArtifactEndingWith(
            getFilesToBuild(getConfiguredTarget("//x:foo_cc_proto")), "bar.pb.h");
    SpawnAction protoCompileAction = getGeneratingSpawnAction(hFile);

    assertThat(protoCompileAction.getArguments())
        .contains(
            String.format(
                "--cpp_out=%s/external/bla",
                getTargetConfiguration().getGenfilesFragment().toString()));
  }

  @Test
  public void commandLineControlsOutputFileSuffixes() throws Exception {
    useConfiguration(
        "--cc_proto_library_header_suffixes=.pb.h,.proto.h",
        "--cc_proto_library_source_suffixes=.pb.cc,.pb.cc.meta");
    scratch.file(
        "x/BUILD",
        "cc_proto_library(name = 'foo_cc_proto', deps = ['foo_proto'])",
        "proto_library(name = 'foo_proto', srcs = ['foo.proto'])");

    assertThat(prettyArtifactNames(getFilesToBuild(getConfiguredTarget("//x:foo_cc_proto"))))
        .containsExactly("x/foo.pb.cc", "x/foo.pb.h", "x/foo.pb.cc.meta", "x/foo.proto.h",
            "x/libfoo_proto.a", "x/libfoo_proto.so");
  }

  // TODO(carmi): test blacklisted protos. I don't currently understand what's the wanted behavior.
}
