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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code proto_library}. */
@RunWith(JUnit4.class)
public class BazelProtoLibraryTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler");
    scratch.file("proto/BUILD", "licenses(['notice'])", "exports_files(['compiler'])");
  }

  @Test
  public void createsDescriptorSets() throws Exception {
    scratch.file(
        "x/BUILD",
        "proto_library(name='alias', deps = ['foo'])",
        "proto_library(name='foo', srcs=['foo.proto'])",
        "proto_library(name='alias_to_no_srcs', deps = ['no_srcs'])",
        "proto_library(name='no_srcs')");

    assertThat(getDescriptorOutput("//x:alias").getRootRelativePathString())
        .isEqualTo("x/alias-descriptor-set.proto.bin");
    assertThat(getDescriptorOutput("//x:foo").getRootRelativePathString())
        .isEqualTo("x/foo-descriptor-set.proto.bin");
    assertThat(getDescriptorOutput("//x:alias_to_no_srcs").getRootRelativePathString())
        .isEqualTo("x/alias_to_no_srcs-descriptor-set.proto.bin");
    assertThat(getDescriptorOutput("//x:no_srcs").getRootRelativePathString())
        .isEqualTo("x/no_srcs-descriptor-set.proto.bin");
  }

  @Test
  public void descriptorSets_ruleWithSrcsCallsProtoc() throws Exception {
    scratch.file("x/BUILD", "proto_library(name='foo', srcs=['foo.proto'])");
    Artifact file = getDescriptorOutput("//x:foo");

    assertThat(getGeneratingSpawnAction(file).getRemainingArguments())
        .containsAllOf(
            "-Ix/foo.proto=x/foo.proto",
            "--descriptor_set_out=" + file.getExecPathString(),
            "x/foo.proto");
  }

  /** Asserts that we register a FileWriteAction with empty contents if there are no srcs. */
  @Test
  public void descriptorSets_ruleWithoutSrcsWritesEmptyFile() throws Exception {
    scratch.file("x/BUILD", "proto_library(name='no_srcs')");
    Action action = getDescriptorWriteAction("//x:no_srcs");
    assertThat(action).isInstanceOf(FileWriteAction.class);
    assertThat(((FileWriteAction) action).getFileContents()).isEmpty();
  }

  /**
   * Asserts that the actions creating descriptor sets for rule R, take as input (=depend on) all of
   * the descriptor sets of the transitive dependencies of R.
   *
   * <p>This is needed so that building R, that has a dependency R' which violates strict proto
   * deps, would break.
   */
  @Test
  public void descriptorSetsDependOnChildren() throws Exception {
    scratch.file(
        "x/BUILD",
        "proto_library(name='alias', deps = ['foo'])",
        "proto_library(name='foo', srcs=['foo.proto'], deps = ['bar'])",
        "proto_library(name='bar', srcs=['bar.proto'])",
        "proto_library(name='alias_to_no_srcs', deps = ['no_srcs'])",
        "proto_library(name='no_srcs')");

    assertThat(getDepsDescriptorSets(getDescriptorOutput("//x:alias")))
        .containsExactly("x/foo-descriptor-set.proto.bin", "x/bar-descriptor-set.proto.bin");
    assertThat(getDepsDescriptorSets(getDescriptorOutput("//x:foo")))
        .containsExactly("x/bar-descriptor-set.proto.bin");
    assertThat(getDepsDescriptorSets(getDescriptorOutput("//x:bar"))).isEmpty();
    assertThat(getDepsDescriptorSets(getDescriptorOutput("//x:alias_to_no_srcs")))
        .containsExactly("x/no_srcs-descriptor-set.proto.bin");
    assertThat(getDepsDescriptorSets(getDescriptorOutput("//x:no_srcs"))).isEmpty();
  }

  /**
   * Returns all of the inputs of the action that generated 'descriptorSet', and which are
   * themselves descriptor sets.
   */
  private ImmutableList<String> getDepsDescriptorSets(Artifact descriptorSet) {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (String input : prettyArtifactNames(getGeneratingAction(descriptorSet).getInputs())) {
      if (input.endsWith("-descriptor-set.proto.bin")) {
        result.add(input);
      }
    }
    return result.build();
  }

  @Test
  public void descriptorSetsAreExposedInProvider() throws Exception {
    scratch.file(
        "x/BUILD",
        "proto_library(name='alias', deps = ['foo'])",
        "proto_library(name='foo', srcs=['foo.proto'], deps = ['bar'])",
        "proto_library(name='bar', srcs=['bar.proto'])",
        "proto_library(name='alias_to_no_srcs', deps = ['no_srcs'])",
        "proto_library(name='no_srcs')");

    {
      ProtoSourcesProvider provider =
          getConfiguredTarget("//x:alias").getProvider(ProtoSourcesProvider.class);
      assertThat(provider.directDescriptorSet().getRootRelativePathString())
          .isEqualTo("x/alias-descriptor-set.proto.bin");
      assertThat(prettyArtifactNames(provider.transitiveDescriptorSets()))
          .containsExactly(
              "x/alias-descriptor-set.proto.bin",
              "x/foo-descriptor-set.proto.bin",
              "x/bar-descriptor-set.proto.bin");
    }

    {
      ProtoSourcesProvider provider =
          getConfiguredTarget("//x:foo").getProvider(ProtoSourcesProvider.class);
      assertThat(provider.directDescriptorSet().getRootRelativePathString())
          .isEqualTo("x/foo-descriptor-set.proto.bin");
      assertThat(prettyArtifactNames(provider.transitiveDescriptorSets()))
          .containsExactly("x/foo-descriptor-set.proto.bin", "x/bar-descriptor-set.proto.bin");
    }

    {
      ProtoSourcesProvider provider =
          getConfiguredTarget("//x:bar").getProvider(ProtoSourcesProvider.class);
      assertThat(provider.directDescriptorSet().getRootRelativePathString())
          .isEqualTo("x/bar-descriptor-set.proto.bin");
      assertThat(prettyArtifactNames(provider.transitiveDescriptorSets()))
          .containsExactly("x/bar-descriptor-set.proto.bin");
    }

    {
      ProtoSourcesProvider provider =
          getConfiguredTarget("//x:alias_to_no_srcs").getProvider(ProtoSourcesProvider.class);
      assertThat(provider.directDescriptorSet().getRootRelativePathString())
          .isEqualTo("x/alias_to_no_srcs-descriptor-set.proto.bin");
      assertThat(prettyArtifactNames(provider.transitiveDescriptorSets()))
          .containsExactly(
              "x/alias_to_no_srcs-descriptor-set.proto.bin", "x/no_srcs-descriptor-set.proto.bin");
    }

    {
      ProtoSourcesProvider provider =
          getConfiguredTarget("//x:no_srcs").getProvider(ProtoSourcesProvider.class);
      assertThat(provider.directDescriptorSet().getRootRelativePathString())
          .isEqualTo("x/no_srcs-descriptor-set.proto.bin");
      assertThat(prettyArtifactNames(provider.transitiveDescriptorSets()))
          .containsExactly("x/no_srcs-descriptor-set.proto.bin");
    }
  }

  @Test
  public void testDescriptorSetOutput_strictDeps() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler", "--strict_proto_deps=error");
    scratch.file(
        "x/BUILD",
        "proto_library(name='nodeps', srcs=['nodeps.proto'])",
        "proto_library(name='withdeps', srcs=['withdeps.proto'], deps=[':dep1', ':dep2'])",
        "proto_library(name='depends_on_alias', srcs=['depends_on_alias.proto'], deps=[':alias'])",
        "proto_library(name='alias', deps=[':dep1', ':dep2'])",
        "proto_library(name='dep1', srcs=['dep1.proto'])",
        "proto_library(name='dep2', srcs=['dep2.proto'])");

    assertThat(getGeneratingSpawnAction(getDescriptorOutput("//x:nodeps")).getRemainingArguments())
        .containsAllOf("--direct_dependencies", "x/nodeps.proto")
        .inOrder();

    assertThat(
            getGeneratingSpawnAction(getDescriptorOutput("//x:withdeps")).getRemainingArguments())
        .containsAllOf("--direct_dependencies", "x/dep1.proto:x/dep2.proto:x/withdeps.proto")
        .inOrder();

    assertThat(
            getGeneratingSpawnAction(getDescriptorOutput("//x:depends_on_alias"))
                .getRemainingArguments())
        .containsAllOf(
            "--direct_dependencies", "x/dep1.proto:x/dep2.proto:x/depends_on_alias.proto")
        .inOrder();
  }

  /**
   * When building a proto_library with multiple srcs (say foo.proto and bar.proto), we should allow
   * foo.proto to import bar.proto without tripping strict-deps checking. This means that
   * --direct_dependencies should list the srcs.
   */
  @Test
  public void testDescriptorSetOutput_strict_deps_multipleSrcs() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler", "--strict_proto_deps=error");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "x", "foo", "proto_library(name='foo', srcs=['foo.proto', 'bar.proto'])");
    Artifact file = getFirstArtifactEndingWith(getFilesToBuild(target), ".proto.bin");
    assertThat(file.getRootRelativePathString()).isEqualTo("x/foo-descriptor-set.proto.bin");

    assertThat(getGeneratingSpawnAction(file).getRemainingArguments())
        .containsAllOf("--direct_dependencies", "x/foo.proto:x/bar.proto")
        .inOrder();
  }

  @Test
  public void testDescriptorSetOutput_strictDeps_disabled() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler", "--strict_proto_deps=off");
    scratch.file("x/BUILD", "proto_library(name='foo', srcs=['foo.proto'])");

    for (String arg :
        getGeneratingSpawnAction(getDescriptorOutput("//x:foo")).getRemainingArguments()) {
      assertThat(arg).doesNotContain("--direct_dependencies=");
    }
  }

  @Test
  public void testProtoSourceRootWithoutDeps() throws Exception {
    scratch.file(
        "x/foo/BUILD",
        "proto_library(",
        "    name = 'nodeps',",
        "    srcs = ['foo/nodeps.proto'],",
        "    proto_source_root = 'x/foo',",
        ")"
    );
    ConfiguredTarget protoTarget = getConfiguredTarget("//x/foo:nodeps");
    ProtoSourcesProvider sourcesProvider = protoTarget.getProvider(ProtoSourcesProvider.class);
    assertThat(sourcesProvider.getTransitiveProtoSourceRoots()).containsExactly("x/foo");

    assertThat(getGeneratingSpawnAction(getDescriptorOutput("//x/foo:nodeps"))
        .getRemainingArguments())
        .contains("--proto_path=x/foo");
  }

  @Test
  public void testProtoSourceRootWithoutDeps_notPackageName() throws Exception {
    scratch.file(
        "x/foo/BUILD",
        "proto_library(",
        "    name = 'nodeps',",
        "    srcs = ['foo/nodeps.proto'],",
        "    proto_source_root = 'something/else',",
        ")"
    );

    try {
      getConfiguredTarget("//x/foo:nodeps");
    } catch (AssertionError error) {
      assertThat(error)
          .hasMessageThat()
          .contains("proto_source_root must be the same as the package name (x/foo)");
      return;
    }
    throw new Exception("Target should have failed building.");
  }

  @Test
  public void testProtoSourceRootWithDepsDuplicate() throws Exception {
    scratch.file(
        "x/foo/BUILD",
        "proto_library(",
        "    name = 'withdeps',",
        "    srcs = ['foo/withdeps.proto'],",
        "    proto_source_root = 'x/foo',",
        "    deps = [':dep'],",
        ")",
        "proto_library(",
        "    name = 'dep',",
        "    srcs = ['foo/dep.proto'],",
        "    proto_source_root = 'x/foo',",
        ")"
    );
    ConfiguredTarget protoTarget = getConfiguredTarget("//x/foo:withdeps");
    ProtoSourcesProvider sourcesProvider = protoTarget.getProvider(ProtoSourcesProvider.class);
    assertThat(sourcesProvider.getTransitiveProtoSourceRoots()).containsExactly("x/foo");

    assertThat(getGeneratingSpawnAction(getDescriptorOutput("//x/foo:withdeps"))
        .getRemainingArguments())
        .contains("--proto_path=x/foo");
  }

  @Test
  public void testProtoSourceRootWithDeps() throws Exception {
    scratch.file(
        "x/foo/BUILD",
        "proto_library(",
        "    name = 'withdeps',",
        "    srcs = ['foo/withdeps.proto'],",
        "    proto_source_root = 'x/foo',",
        "    deps = ['//x/bar:dep', ':dep'],",
        ")",
        "proto_library(",
        "    name = 'dep',",
        "    srcs = ['foo/dep.proto'],",
        ")"
    );
    scratch.file(
        "x/bar/BUILD",
        "proto_library(",
        "    name = 'dep',",
        "    srcs = ['foo/dep.proto'],",
        "    proto_source_root = 'x/bar',",
        ")"
    );
    ConfiguredTarget protoTarget = getConfiguredTarget("//x/foo:withdeps");
    ProtoSourcesProvider sourcesProvider = protoTarget.getProvider(ProtoSourcesProvider.class);
    assertThat(sourcesProvider.getTransitiveProtoSourceRoots())
        .containsExactly("x/foo", "x/bar", ".");
  }

  @Test
  public void testExportedProtoSourceRoots() throws Exception {
    scratch.file("ad/BUILD",
        "proto_library(name='ad', proto_source_root='ad', srcs=['ad.proto'])");
    scratch.file("ae/BUILD",
        "proto_library(name='ae', proto_source_root='ae', srcs=['ae.proto'])");
    scratch.file("bd/BUILD",
        "proto_library(name='bd', proto_source_root='bd', srcs=['bd.proto'])");
    scratch.file("be/BUILD",
        "proto_library(name='be', proto_source_root='be', srcs=['be.proto'])");
    scratch.file("a/BUILD",
        "proto_library(",
        "    name='a',",
        "    proto_source_root='a',",
        "    srcs=['a.proto'],",
        "    exports=['//ae:ae'],",
        "    deps=['//ad:ad'])");
    scratch.file("b/BUILD",
        "proto_library(",
        "    name='b',",
        "    proto_source_root='b',",
        "    srcs=['b.proto'],",
        "    exports=['//be:be'],",
        "    deps=['//bd:bd'])");
    scratch.file("c/BUILD",
        "proto_library(",
        "    name='c',",
        "    proto_source_root='c',",
        "    srcs=['c.proto'],",
        "    exports=['//a:a'],",
        "    deps=['//b:b'])");

    ConfiguredTarget c = getConfiguredTarget("//c:c");
    // exported proto source roots should be the source root of the rule and the direct source roots
    // of its exports and nothing else (not the exports of its exports or the deps of its exports
    // or the exports of its deps)
    assertThat(c.getProvider(ProtoSourcesProvider.class).getExportedProtoSourceRoots())
        .containsExactly("a", "c");
  }

  @Test
  public void testProtoSourceRoot() throws Exception {
    scratch.file(
        "x/foo/BUILD",
        "proto_library(",
        "    name = 'banana',",
        "    srcs = ['foo.proto'],",
        "    proto_source_root = 'x/foo',",
        ")");

    ConfiguredTarget protoTarget = getConfiguredTarget("//x/foo:banana");
    ProtoSourcesProvider sourcesProvider = protoTarget.getProvider(ProtoSourcesProvider.class);

    assertThat(sourcesProvider.getProtoSourceRoot()).isEqualTo("x/foo");
  }

  private Artifact getDescriptorOutput(String label) throws Exception {
    return getFirstArtifactEndingWith(getFilesToBuild(getConfiguredTarget(label)), ".proto.bin");
  }

  private Action getDescriptorWriteAction(String label) throws Exception {
    return getGeneratingAction(getDescriptorOutput(label));
  }
}
