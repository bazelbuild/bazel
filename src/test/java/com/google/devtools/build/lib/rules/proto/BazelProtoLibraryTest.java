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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BazelProtoLibraryTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    scratch.file("third_party/protobuf/BUILD", "licenses(['notice'])", "exports_files(['protoc'])");
  }

  @Test
  public void testDescriptorSetOutput() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("x", "foo", "proto_library(name='foo', srcs=['foo.proto'])");
    Artifact file =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), ".proto.bin");
    assertThat(file.getRootRelativePathString()).isEqualTo("x/foo-descriptor-set.proto.bin");

    assertThat(getGeneratingSpawnAction(file).getRemainingArguments())
        .containsAllOf(
            "-Ix/foo.proto=x/foo.proto",
            "--descriptor_set_out=" + file.getExecPathString(),
            "x/foo.proto");
  }

  @Test
  public void testDescriptorSetOutput_aliasLibrary() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "x",
            "alias",
            "proto_library(name='alias', deps = [':second_alias'])",
            "proto_library(name='second_alias', deps = [':foo'])",
            "proto_library(name='foo', srcs=['foo.proto'])");
    Artifact file =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), ".proto.bin");
    assertThat(file.getRootRelativePathString()).isEqualTo("x/alias-descriptor-set.proto.bin");

    assertThat(getGeneratingSpawnAction(file).getRemainingArguments())
        .containsAllOf(
            "-Ix/foo.proto=x/foo.proto",
            "--descriptor_set_out=" + file.getExecPathString(),
            "x/foo.proto");
  }

  @Test
  public void testDescriptorSetOutput_noSrcs() throws Exception {
    ConfiguredTarget target = scratchConfiguredTarget("x", "foo", "proto_library(name='foo')");
    assertThat(ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(target), ".proto.bin"))
        .isNull();
  }

  @Test
  public void testDescriptorSetOutput_strictDeps() throws Exception {
    useConfiguration("--strict_proto_deps=error");
    scratch.file(
        "x/BUILD",
        "proto_library(name='nodeps', srcs=['nodeps.proto'])",
        "proto_library(name='withdeps', srcs=['withdeps.proto'], deps=[':dep1', ':dep2'])",
        "proto_library(name='depends_on_alias', srcs=['depends_on_alias.proto'], deps=[':alias'])",
        "proto_library(name='alias', deps=[':dep1', ':dep2'])",
        "proto_library(name='dep1', srcs=['dep1.proto'])",
        "proto_library(name='dep2', srcs=['dep2.proto'])");

    assertThat(getGeneratingSpawnAction(getDescriptorOutput("//x:nodeps")).getRemainingArguments())
        .contains("--direct_dependencies=");

    assertThat(
            getGeneratingSpawnAction(getDescriptorOutput("//x:withdeps")).getRemainingArguments())
        .contains("--direct_dependencies=x/dep1.proto:x/dep2.proto");

    assertThat(
            getGeneratingSpawnAction(getDescriptorOutput("//x:depends_on_alias"))
                .getRemainingArguments())
        .contains("--direct_dependencies=x/dep1.proto:x/dep2.proto");
  }

  @Test
  public void testDescriptorSetOutput_strictDeps_aliasLibrary() throws Exception {
    useConfiguration("--strict_proto_deps=error");
    scratch.file(
        "x/BUILD",
        "proto_library(name='alias', deps=[':dep1', ':subalias'])",
        "proto_library(name='dep1', srcs=['dep1.proto'], deps = [':subdep1'])",
        "proto_library(name='subdep1', srcs=['subdep1.proto'])",
        "proto_library(name='subalias', deps = [':dep2'])",
        "proto_library(name='dep2', srcs = ['dep2.proto'], deps = [':subdep2'])",
        "proto_library(name='subdep2', srcs=['subdep2.proto'])");

    assertThat(getGeneratingSpawnAction(getDescriptorOutput("//x:alias")).getRemainingArguments())
        .containsAllOf(
            "--direct_dependencies=x/subdep1.proto:x/subdep2.proto",
            "x/dep1.proto",
            "x/dep2.proto");
  }

  @Test
  public void testDescriptorSetOutput_strictDeps_disabled() throws Exception {
    useConfiguration("--strict_proto_deps=off");
    scratch.file("x/BUILD", "proto_library(name='foo', srcs=['foo.proto'])");

    for (String arg :
        getGeneratingSpawnAction(getDescriptorOutput("//x:foo")).getRemainingArguments()) {
      assertThat(arg).doesNotContain("--direct_dependencies=");
    }
  }

  private Artifact getDescriptorOutput(String label) throws Exception {
    return ActionsTestUtil.getFirstArtifactEndingWith(
        getFilesToBuild(getConfiguredTarget(label)), ".proto.bin");
  }
}
