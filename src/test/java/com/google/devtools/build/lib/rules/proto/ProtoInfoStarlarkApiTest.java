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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoCommonApi;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProtoInfoStarlarkApiTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler"); // TODO check do we need that.
    scratch.file("proto/BUILD", "licenses(['notice'])", "exports_files(['compiler'])");
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");
    scratch.file("myinfo/BUILD");
    MockProtoSupport.setup(mockToolsConfig);

    MockProtoSupport.setupWorkspace(scratch);
    invalidatePackages();
  }

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//myinfo:myinfo.bzl", ImmutableMap.of()), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  @Test
  public void testProtoCommon() throws Exception {
    scratch.file(
        "foo/test.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "  return MyInfo(proto_common=proto_common)",
        "test = rule(implementation = _impl, attrs = {})");

    scratch.file("foo/BUILD", "load(':test.bzl', 'test')", "test(name='test')");

    ConfiguredTarget test = getConfiguredTarget("//foo:test");
    Object protoCommon = getMyInfoFromTarget(test).getValue("proto_common");
    assertThat(protoCommon).isInstanceOf(ProtoCommonApi.class);
  }

  @Test
  public void testProvider() throws Exception {
    scratch.file(
        "foo/test.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "  provider = ctx.attr.dep[ProtoInfo]", // NB: This is the modern provider
        "  return MyInfo(direct_sources=provider.direct_sources)",
        "test = rule(implementation = _impl, attrs = {'dep': attr.label()})");

    scratch.file(
        "foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load(':test.bzl', 'test')",
        "test(name='test', dep=':proto')",
        "proto_library(name='proto', srcs=['p.proto'])");

    ConfiguredTarget test = getConfiguredTarget("//foo:test");
    @SuppressWarnings("unchecked")
    Iterable<Artifact> directSources =
        (Iterable<Artifact>) getMyInfoFromTarget(test).getValue("direct_sources");
    assertThat(ActionsTestUtil.baseArtifactNames(directSources)).containsExactly("p.proto");
  }

  @Test
  public void testProtoSourceRootExportedInStarlark() throws Exception {
    scratch.file(
        "third_party/foo/my_proto_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "",
        "def _my_proto_rule_impl(ctx):",
        "    return MyInfo(",
        "        fetched_proto_source_root = ctx.attr.protodep[ProtoInfo].proto_source_root",
        "    )",
        "",
        "my_proto_rule = rule(",
        "    implementation = _my_proto_rule_impl,",
        "    attrs = {'protodep': attr.label()},",
        ")");

    scratch.file(
        "third_party/foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "load(':my_proto_rule.bzl', 'my_proto_rule')",
        "my_proto_rule(",
        "  name = 'my_proto',",
        "  protodep = ':dep_proto',",
        ")",
        "proto_library(",
        "  name = 'dep_proto',",
        "  srcs = ['dep.proto'],",
        "  strip_import_prefix = '/third_party/foo',",
        ")");

    ConfiguredTarget ct = getConfiguredTarget("//third_party/foo:my_proto");
    String protoSourceRoot = (String) getMyInfoFromTarget(ct).getValue("fetched_proto_source_root");
    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();

    assertThat(protoSourceRoot).isEqualTo(genfiles + "/third_party/foo/_virtual_imports/dep_proto");
  }

  @Test
  public void testProtoInfoMinimal() throws Exception {
    scratch.file(
        "third_party/foo/my_proto_rule.bzl",
        "",
        "result = provider()",
        "def _my_proto_rule_impl(ctx):",
        "    descriptor_set = ctx.actions.declare_file('descriptor-set.proto.bin')",
        "    ctx.actions.write(output = descriptor_set, content = 'descriptor set content')",
        "    protoInfo = ProtoInfo(",
        "        descriptor_set = descriptor_set,",
        "        proto_source_root = 'my_proto_root',",
        "    )",
        "    return [result(property = protoInfo), protoInfo]",
        "",
        "my_proto_rule = rule(",
        "    implementation = _my_proto_rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(providers=[ProtoInfo]),",
        "        'srcs': attr.label_list(allow_files=['.proto']),",
        "     },",
        "    provides = [ProtoInfo],",
        ")");

    scratch.file(
        "third_party/foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "load(':my_proto_rule.bzl', 'my_proto_rule')",
        "my_proto_rule(",
        "  name = 'my_proto',",
        "  srcs = [':src_proto_src'],",
        "  deps = [':dep_proto'],",
        ")",
        "proto_library(",
        "  name = 'dep_proto',",
        "  srcs = ['dep.proto'],",
        "  strip_import_prefix = '/third_party/foo',",
        ")",
        "filegroup(",
        "    name = 'src_proto_src',",
        "    srcs = ['src.proto'],",
        ")");

    ProtoInfo protoInfo = fetchMyProtoInfo(getConfiguredTarget("//third_party/foo:my_proto"));
    assertThat(protoInfo.getDirectProtoSourceRoot()).isEqualTo("my_proto_root");
    assertThat(protoInfo.getDirectDescriptorSet().prettyPrint()).isEqualTo("third_party/foo/descriptor-set.proto.bin");
  }

  @Test
  public void testProtoInfoInStarlark() throws Exception {
    scratch.file(
        "third_party/foo/my_proto_rule.bzl",
        "",
        "result = provider()",
        "def _my_proto_rule_impl(ctx):",
        "    descriptor_set = ctx.actions.declare_file('descriptor-set.proto.bin')",
        "    ctx.actions.write(output = descriptor_set, content = 'descriptor set content')",
        "    protoInfo = ProtoInfo(",
        "        descriptor_set = descriptor_set,",
        "        proto_source_root = 'third_party',",
        "        sources = ctx.files.srcs,",
        "        deps = [dep[ProtoInfo] for dep in ctx.attr.deps],",
        "        exports = [export[ProtoInfo] for export in ctx.attr.exports],",
        "    )",
        "    return [result(property = protoInfo), protoInfo]",
        "",
        "my_proto_rule = rule(",
        "    implementation = _my_proto_rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(providers=[ProtoInfo]),",
        "        'exports': attr.label_list(providers=[ProtoInfo]),",
        "        'srcs': attr.label_list(allow_files=['.proto']),",
        "     },",
        "    provides = [ProtoInfo],",
        ")");

    scratch.file(
        "third_party/foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "load(':my_proto_rule.bzl', 'my_proto_rule')",
        "my_proto_rule(",
        "  name = 'my_proto',",
        "  srcs = ['src.proto'],",
        "  deps = [':dep_proto'],",
        "  exports = [':export_proto'],",
        ")",
        "proto_library(",
        "  name = 'dep_proto',",
        "  srcs = ['dep.proto'],",
        "  strip_import_prefix = '/third_party/foo',",
        ")",
        "proto_library(",
        "  name = 'export_proto',",
        "  srcs = ['export.proto'],",
        "  strip_import_prefix = '/third_party/foo',",
        ")");

    ProtoInfo protoInfo = fetchMyProtoInfo(getConfiguredTarget("//third_party/foo:my_proto"));
    assertThat(protoInfo.getDirectProtoSourceRoot()).isEqualTo("third_party");
    assertThat(protoInfo.getDirectDescriptorSet().prettyPrint())
        .isEqualTo("third_party/foo/descriptor-set.proto.bin");
    assertThat(prettyArtifactNames(protoInfo.getDirectProtoSources()))
        .containsExactly("third_party/foo/src.proto");
    assertThat(prettyArtifactNames(protoInfo.getStrictImportableProtoSourcesForDependents()))
        .containsExactly("third_party/foo/src.proto");
    assertThat(prettyArtifactNames(protoInfo.getTransitiveDescriptorSets()))
        .containsExactly("third_party/foo/descriptor-set.proto.bin",
            "third_party/foo/dep_proto-descriptor-set.proto.bin");
    List<String> transitiveProtoSourceRoots = protoInfo.getTransitiveProtoSourceRoots().toList();
    assertThat(transitiveProtoSourceRoots).hasSize(2);
    assertThat(transitiveProtoSourceRoots.get(0)).endsWith("third_party/foo/_virtual_imports/dep_proto");
    assertThat(transitiveProtoSourceRoots.get(1)).isEqualTo("third_party");
    assertThat(prettyArtifactNames(protoInfo.getTransitiveProtoSources()))
        .containsExactly("third_party/foo/src.proto",
            "third_party/foo/_virtual_imports/dep_proto/dep.proto");
    List<ProtoSource> exportedProtoSources =
        protoInfo.getExportedSources().toList();
    assertThat(exportedProtoSources).hasSize(1);
    assertThat(exportedProtoSources.get(0).getSourceRoot().toString())
        .isEqualTo("third_party");
    assertThat(exportedProtoSources.get(0).getSourceFile().prettyPrint())
        .isEqualTo("third_party/foo/src.proto");
    assertThat(exportedProtoSources.get(0).getImportPath().toString())
        .isEqualTo("foo/src.proto");
  }

  @Test
  public void testProtoInfoProxyInStarlark() throws Exception {
    scratch.file(
        "third_party/foo/my_proto_rule.bzl",
        "",
        "result = provider()",
        "def _my_proto_rule_impl(ctx):",
        "    descriptor_set = ctx.actions.declare_file('descriptor-set.proto.bin')",
        "    ctx.actions.write(output = descriptor_set, content = 'descriptor set content')",
        "    protoInfo = ProtoInfo(",
        "        descriptor_set = descriptor_set,",
        "        proto_source_root = 'my_proto_root',",
        "        sources = ctx.files.srcs,",
        "        deps = [dep[ProtoInfo] for dep in ctx.attr.deps],",
        "    )",
        "    return [result(property = protoInfo), protoInfo]",
        "",
        "my_proto_rule = rule(",
        "    implementation = _my_proto_rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(providers=[ProtoInfo]),",
        "        'srcs': attr.label_list(allow_files=['.proto']),",
        "     },",
        "    provides = [ProtoInfo],",
        ")");

    scratch.file(
        "third_party/foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "load(':my_proto_rule.bzl', 'my_proto_rule')",
        "my_proto_rule(",
        "  name = 'my_proto',",
        "  deps = [':dep_proto'],",
        ")",
        "proto_library(",
        "  name = 'dep_proto',",
        "  srcs = ['dep.proto'],",
        "  strip_import_prefix = '/third_party/foo',",
        ")");

    ProtoInfo protoInfo = fetchMyProtoInfo(getConfiguredTarget("//third_party/foo:my_proto"));
    assertThat(protoInfo.getDirectProtoSourceRoot()).isEqualTo("my_proto_root");
    assertThat(protoInfo.getDirectDescriptorSet().prettyPrint())
        .isEqualTo("third_party/foo/descriptor-set.proto.bin");
    assertThat(prettyArtifactNames(protoInfo.getStrictImportableProtoSourcesForDependents()))
        .containsExactly("third_party/foo/_virtual_imports/dep_proto/dep.proto");
    assertThat(prettyArtifactNames(protoInfo.getDirectProtoSources())).isEmpty();
    assertThat(prettyArtifactNames(protoInfo.getTransitiveDescriptorSets()))
        .containsExactly("third_party/foo/descriptor-set.proto.bin",
            "third_party/foo/dep_proto-descriptor-set.proto.bin");
    assertThat(protoInfo.getTransitiveProtoSourceRoots().toList().get(0))
        .endsWith("third_party/foo/_virtual_imports/dep_proto");
    assertThat(prettyArtifactNames(protoInfo.getTransitiveProtoSources()))
        .containsExactly("third_party/foo/_virtual_imports/dep_proto/dep.proto");
  }

  @Test
  public void testProtoInfoUsedAsDependencyInStarlark() throws Exception {
    scratch.file(
        "third_party/foo/my_proto_rule.bzl",
        "",
        "result = provider()",
        "def _my_proto_rule_impl(ctx):",
        "    descriptor_set = ctx.actions.declare_file('descriptor-set.proto.bin')",
        "    ctx.actions.write(output = descriptor_set, content = 'descriptor set content')",
        "    protoInfo = ProtoInfo(",
        "        descriptor_set = descriptor_set,",
        "        proto_source_root = 'third_party',",
        "        sources = ctx.files.srcs,",
        "        deps = [dep[ProtoInfo] for dep in ctx.attr.deps],",
        "        exports = [export[ProtoInfo] for export in ctx.attr.exports],",
        "    )",
        "    return [result(property = protoInfo), protoInfo]",
        "",
        "my_proto_rule = rule(",
        "    implementation = _my_proto_rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(providers=[ProtoInfo]),",
        "        'exports': attr.label_list(providers=[ProtoInfo]),",
        "        'srcs': attr.label_list(allow_files=['.proto']),",
        "     },",
        "    provides = [ProtoInfo],",
        ")");

    scratch.file(
        "third_party/foo/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "load(':my_proto_rule.bzl', 'my_proto_rule')",
        "my_proto_rule(",
        "  name = 'my_proto',",
        "  srcs = ['my.proto'],",
        ")",
        "proto_library(",
        "  name = 'new_proto',",
        "  srcs = ['new.proto'],",
        "  deps = ['my_proto'],",
        "  exports = ['my_proto'],",
        "  strip_import_prefix = '/third_party/foo',",
        ")");

    ProtoInfo protoInfo = fetchProtoInfo(getConfiguredTarget("//third_party/foo:new_proto"));
    assertThat(prettyArtifactNames(protoInfo.getTransitiveProtoSources()))
        .contains("third_party/foo/my.proto");
  }

  private ProtoInfo fetchMyProtoInfo(ConfiguredTarget configuredTarget) throws Exception {
    StructImpl info =
        (StructImpl)
            configuredTarget.get(
                new StarlarkProvider.Key(
                    Label.parseAbsolute("//third_party/foo:my_proto_rule.bzl", ImmutableMap.of()), "result"));

    @SuppressWarnings("unchecked")
    ProtoInfo protoInfo = (ProtoInfo) info.getValue("property");
    return protoInfo;
  }

  private ProtoInfo fetchProtoInfo(ConfiguredTarget configuredTarget) {
    return (ProtoInfo) configuredTarget.get(StarlarkProviderIdentifier.forKey(ProtoInfo.PROVIDER.getKey()));
  }
}
