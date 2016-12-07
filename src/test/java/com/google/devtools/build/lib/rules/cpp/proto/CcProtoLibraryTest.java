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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CcProtoLibraryTest extends BuildViewTestCase {
  @Before
  public void setUp() throws Exception {
    scratch.file("third_party/protobuf/BUILD", "licenses(['notice'])", "exports_files(['protoc'])");
    scratch.file(
        "protobuf/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_lang_toolchain(",
        "    name = 'cc_toolchain',",
        "    command_line = '--cpp_out=$(OUT)',",
        "    blacklisted_protos = [],",
        ")");

    String existingWorkspace =
        new String(FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    mockToolsConfig.overwrite(
        "WORKSPACE",
        "local_repository(name = 'com_google_protobuf_cc', path = 'protobuf/')",
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

    ProtoCcHeaderProvider headers =
        getConfiguredTarget("//x:foo_cc_proto").getProvider(ProtoCcHeaderProvider.class);
    assertThat(prettyArtifactNames(headers.getHeaders())).containsExactly("x/foo.pb.h");
  }

  // TODO(carmi): test blacklisted protos. I don't currently understand what's the wanted behavior.
}
