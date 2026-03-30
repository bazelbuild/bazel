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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.proto.ProtoInfo.ProtoInfoProvider;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code proto_library}. */
@RunWith(JUnit4.class)
public class BazelProtoLibraryTest extends BuildViewTestCase {
  private boolean isThisBazel() {
    return getAnalysisMock().isThisBazel();
  }

  @Before
  public void setUp() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler");
    MockProtoSupport.setup(mockToolsConfig);
    scratch.file(
        "proto/BUILD",
        """
        licenses(["notice"])

        exports_files(["compiler"])
        """);

    invalidatePackages();
  }

  @Test
  public void protoToolchainResolution_enabled() throws Exception {
    setBuildLanguageOptions("--incompatible_enable_proto_toolchain_resolution");
    scratch.file(
        "x/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='foo', srcs=['foo.proto'])");

    getDescriptorOutput("//x:foo");

    assertNoEvents();
  }

  private void testExternalRepoWithGeneratedProto(boolean siblingRepoLayout) throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'foo')",
        "local_path_override(module_name = 'foo', path = '/foo')");
    if (siblingRepoLayout) {
      setBuildLanguageOptions("--experimental_sibling_repository_layout");
    }

    scratch.file("/foo/MODULE.bazel", "module(name = 'foo')");
    scratch.file(
        "/foo/x/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='x', srcs=['generated.proto'])",
        "genrule(name='g', srcs=[], outs=['generated.proto'], cmd='')");
    scratch.file(
        "a/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='a', srcs=['a.proto'], deps=['@foo//x:x'])");
    invalidatePackages();

    String genfiles =
        getTargetConfiguration()
            .getGenfilesFragment(
                siblingRepoLayout ? RepositoryName.create("foo+") : RepositoryName.MAIN)
            .toString();
    String fooProtoRoot;
    fooProtoRoot = (siblingRepoLayout ? genfiles : genfiles + "/external/foo+");
    ConfiguredTarget a = getConfiguredTarget("//a:a");
    ProtoInfo aInfo = getProtoInfo(a);
    assertThat(aInfo.getTransitiveProtoSourceRoots().toList()).containsExactly(".", fooProtoRoot);

    ConfiguredTarget x = getConfiguredTarget("@@foo+//x:x");
    ProtoInfo xInfo = getProtoInfo(x);
    assertThat(xInfo.getTransitiveProtoSourceRoots().toList()).containsExactly(fooProtoRoot);
  }

  @Test
  public void testExternalRepoWithGeneratedProto_withSubdirRepoLayout() throws Exception {
    testExternalRepoWithGeneratedProto(/* siblingRepoLayout= */ false);
  }

  @Test
  public void test_siblingRepoLayout_externalRepoWithGeneratedProto() throws Exception {
    testExternalRepoWithGeneratedProto(/* siblingRepoLayout= */ true);
  }

  private void testImportPrefixInExternalRepo(boolean siblingRepoLayout) throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'yolo_repo')",
        "local_path_override(module_name = 'yolo_repo', path = '/yolo_repo')");

    if (siblingRepoLayout) {
      setBuildLanguageOptions("--experimental_sibling_repository_layout");
    }

    scratch.file("/yolo_repo/MODULE.bazel", "module(name = 'yolo_repo')");
    scratch.file("/yolo_repo/yolo_pkg/yolo.proto");
    scratch.file(
        "/yolo_repo/yolo_pkg/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "  name = 'yolo_proto',",
        "  srcs = ['yolo.proto'],",
        "  import_prefix = 'bazel.build/yolo',",
        "  visibility = ['//visibility:public'],",
        ")");
    invalidatePackages();

    ConfiguredTarget target = getConfiguredTarget("@@yolo_repo+//yolo_pkg:yolo_proto");
    assertThat(
            Iterables.getOnlyElement(
                    getProtoInfo(target).getStrictImportableProtoSourcesForDependents().toList())
                .getExecPathString())
        .endsWith("/_virtual_imports/yolo_proto/bazel.build/yolo/yolo_pkg/yolo.proto");
  }

  @Test
  public void testImportPrefixInExternalRepo_withSubdirRepoLayout() throws Exception {
    testImportPrefixInExternalRepo(/*siblingRepoLayout=*/ false);
  }

  @Test
  public void testImportPrefixInExternalRepo_withSiblingRepoLayout() throws Exception {
    testImportPrefixInExternalRepo(/*siblingRepoLayout=*/ true);
  }

  private void testImportPrefixAndStripInExternalRepo(boolean siblingRepoLayout) throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'yolo_repo')",
        "local_path_override(module_name = 'yolo_repo', path = '/yolo_repo')");

    if (siblingRepoLayout) {
      setBuildLanguageOptions("--experimental_sibling_repository_layout");
    }

    scratch.file("/yolo_repo/MODULE.bazel", "module(name = 'yolo_repo')");
    scratch.file("/yolo_repo/yolo_pkg_to_be_stripped/yolo_pkg/yolo.proto");
    scratch.file(
        "/yolo_repo/yolo_pkg_to_be_stripped/yolo_pkg/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "  name = 'yolo_proto',",
        "  srcs = ['yolo.proto'],",
        "  import_prefix = 'bazel.build/yolo',",
        "  strip_import_prefix = '/yolo_pkg_to_be_stripped',",
        "  visibility = ['//visibility:public'],",
        ")");
    invalidatePackages();

    ConfiguredTarget target =
        getConfiguredTarget("@@yolo_repo+//yolo_pkg_to_be_stripped/yolo_pkg:yolo_proto");
    assertThat(
            Iterables.getOnlyElement(
                    getProtoInfo(target).getStrictImportableProtoSourcesForDependents().toList())
                .getExecPathString())
        .endsWith("/_virtual_imports/yolo_proto/bazel.build/yolo/yolo_pkg/yolo.proto");
  }

  @Test
  public void testImportPrefixAndStripInExternalRepo_withSubdirRepoLayout() throws Exception {
    testImportPrefixAndStripInExternalRepo(/*siblingRepoLayout=*/ false);
  }

  @Test
  public void testImportPrefixAndStripInExternalRepo_withSiblingRepoLayout() throws Exception {
    testImportPrefixAndStripInExternalRepo(/*siblingRepoLayout=*/ true);
  }

  private void testStripImportPrefixInExternalRepo(boolean siblingRepoLayout) throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'yolo_repo')",
        "local_path_override(module_name = 'yolo_repo', path = '/yolo_repo')");

    if (siblingRepoLayout) {
      setBuildLanguageOptions("--experimental_sibling_repository_layout");
    }

    scratch.file("/yolo_repo/MODULE.bazel", "module(name = 'yolo_repo')");
    scratch.file("/yolo_repo/yolo_pkg_to_be_stripped/yolo_pkg/yolo.proto");
    scratch.file(
        "/yolo_repo/yolo_pkg_to_be_stripped/yolo_pkg/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "  name = 'yolo_proto',",
        "  srcs = ['yolo.proto'],",
        "  strip_import_prefix = '/yolo_pkg_to_be_stripped',",
        "  visibility = ['//visibility:public'],",
        ")");
    invalidatePackages();

    ConfiguredTarget target =
        getConfiguredTarget("@@yolo_repo+//yolo_pkg_to_be_stripped/yolo_pkg:yolo_proto");
    assertThat(
            Iterables.getOnlyElement(
                    getProtoInfo(target).getStrictImportableProtoSourcesForDependents().toList())
                .getExecPathString())
        .endsWith("/_virtual_imports/yolo_proto/yolo_pkg/yolo.proto");
  }

  @Test
  public void testStripImportPrefixInExternalRepo_withSubdirRepoLayout() throws Exception {
    testStripImportPrefixInExternalRepo(/*siblingRepoLayout=*/ false);
  }

  @Test
  public void testStripImportPrefixInExternalRepo_withSiblingRepoLayout() throws Exception {
    testStripImportPrefixInExternalRepo(/*siblingRepoLayout=*/ true);
  }

  private void testRelativeStripImportPrefixInExternalRepo(boolean siblingRepoLayout)
      throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'yolo_repo')",
        "local_path_override(module_name = 'yolo_repo', path = '/yolo_repo')");

    if (siblingRepoLayout) {
      setBuildLanguageOptions("--experimental_sibling_repository_layout");
    }

    scratch.file("/yolo_repo/MODULE.bazel", "module(name = 'yolo_repo')");
    scratch.file("/yolo_repo/yolo_pkg_to_be_stripped/yolo_pkg/yolo.proto");
    scratch.file(
        "/yolo_repo/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "  name = 'yolo_proto',",
        "  srcs = ['yolo_pkg_to_be_stripped/yolo_pkg/yolo.proto'],",
        "  strip_import_prefix = 'yolo_pkg_to_be_stripped',",
        "  visibility = ['//visibility:public'],",
        ")");
    invalidatePackages();

    ConfiguredTarget target = getConfiguredTarget("@@yolo_repo+//:yolo_proto");
    assertThat(
            Iterables.getOnlyElement(
                    getProtoInfo(target).getStrictImportableProtoSourcesForDependents().toList())
                .getExecPathString())
        .endsWith("/_virtual_imports/yolo_proto/yolo_pkg/yolo.proto");
  }

  @Test
  public void testRelativeStripImportPrefixInExternalRepo_withSubdirRepoLayout() throws Exception {
    testRelativeStripImportPrefixInExternalRepo(/*siblingRepoLayout=*/ false);
  }

  @Test
  public void testRelativeStripImportPrefixInExternalRepo_withSiblingRepoLayout() throws Exception {
    testRelativeStripImportPrefixInExternalRepo(/*siblingRepoLayout=*/ true);
  }

  @Test
  public void testIllegalImportPrefix() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file(
        "a/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'a',",
        "    srcs = ['a.proto'],",
        "    import_prefix = '/foo')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a");
    assertContainsEvent("should be a relative path");
  }

  @Test
  public void testStripImportPrefixAndImportPrefix() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file(
        "a/b/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'd',",
        "    srcs = ['c/d.proto'],",
        "    import_prefix = 'foo',",
        "    strip_import_prefix = 'c')");

    ImmutableList<String> commandLine =
        allArgsForAction((SpawnAction) getDescriptorWriteAction("//a/b:d"));
    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    assertThat(commandLine).contains("-I" + genfiles + "/a/b/_virtual_imports/d");
  }

  @Test
  public void testImportPrefixWithoutStripImportPrefix() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file(
        "a/b/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'd',",
        "    srcs = ['c/d.proto'],",
        "    import_prefix = 'foo')");

    ImmutableList<String> commandLine =
        allArgsForAction((SpawnAction) getDescriptorWriteAction("//a/b:d"));
    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    assertThat(commandLine).contains("-I" + genfiles + "/a/b/_virtual_imports/d");
  }

  @Test
  public void testDotInImportPrefix() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file(
        "a/b/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'd',",
        "    srcs = ['c/d.proto'],",
        "    import_prefix = './e')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a/b:d");
    assertContainsEvent("should be normalized");
  }

  @Test
  public void testDotDotInImportPrefix() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file(
        "a/b/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'd',",
        "    srcs = ['c/d.proto'],",
        "    import_prefix = '../e')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a/b:d");
    assertContainsEvent("should be normalized");
  }

  @Test
  public void testStripImportPrefixAndImportPrefixWithStrictProtoDeps() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    useConfiguration("--strict_proto_deps=STRICT");
    scratch.file(
        "a/b/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'd',",
        "    srcs = ['c/d.proto'],",
        "    import_prefix = 'foo',",
        "    strip_import_prefix = 'c')");

    ImmutableList<String> commandLine =
        allArgsForAction((SpawnAction) getDescriptorWriteAction("//a/b:d"));
    assertThat(commandLine).containsAtLeast("--direct_dependencies", "foo/d.proto").inOrder();
  }

  @Test
  public void testStripImportPrefixForExternalRepositories() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'foo')",
        "local_path_override(module_name = 'foo', path = '/foo')");

    scratch.file("/foo/MODULE.bazel", "module(name = 'foo')");
    scratch.file(
        "/foo/x/y/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(",
        "    name = 'q',",
        "    srcs = ['z/q.proto'],",
        "    strip_import_prefix = '/x')");

    scratch.file(
        "a/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='a', srcs=['a.proto'], deps=['@foo//x/y:q'])");
    invalidatePackages();

    ImmutableList<String> commandLine =
        allArgsForAction((SpawnAction) getDescriptorWriteAction("//a:a"));
    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    assertThat(commandLine).contains("-I" + genfiles + "/external/foo+/x/y/_virtual_imports/q");
  }

  @CanIgnoreReturnValue
  private Artifact getDescriptorOutput(String label) throws Exception {
    return getFirstArtifactEndingWith(getFilesToBuild(getConfiguredTarget(label)), ".proto.bin");
  }

  private Action getDescriptorWriteAction(String label) throws Exception {
    return getGeneratingAction(getDescriptorOutput(label));
  }

  @Test
  public void testDependencyOnProtoSourceInExternalRepo() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file("third_party/foo/MODULE.bazel", "module(name = 'foo')");
    scratch.file(
        "third_party/foo/BUILD.bazel",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='a', srcs=['a.proto'])",
        "proto_library(name='c', srcs=['a/b/c.proto'])");
    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'foo')",
        "local_path_override(module_name = 'foo', path = 'third_party/foo')");
    invalidatePackages();

    scratch.file(
        "x/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='a', srcs=['a.proto'], deps=['@foo//:a'])",
        "proto_library(name='c', srcs=['c.proto'], deps=['@foo//:c'])");

    {
      ImmutableList<String> commandLine =
          allArgsForAction((SpawnAction) getDescriptorWriteAction("//x:a"));
      assertThat(commandLine).containsAtLeast("-Iexternal/foo+", "-I.");
    }

    {
      ImmutableList<String> commandLine =
          allArgsForAction((SpawnAction) getDescriptorWriteAction("//x:c"));
      assertThat(commandLine).containsAtLeast("-Iexternal/foo+", "-I.");
    }
  }

  @Test
  public void testProtoLibraryWithVirtualProtoSourceRoot() throws Exception {
    if (!isThisBazel()) {
      return;
    }

    scratch.file(
        "x/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='foo', srcs=['a.proto'], import_prefix='foo')");

    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    ProtoInfo provider = getProtoInfo(getConfiguredTarget("//x:foo"));
    assertThat(Iterables.transform(provider.getDirectProtoSources(), s -> s.getExecPathString()))
        .containsExactly(genfiles + "/x/_virtual_imports/foo/foo/x/a.proto");
  }


  @Test
  public void protoLibrary_reexport_allowed() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        """
        proto_library(
            name = "foo",
            srcs = ["foo.proto"],
            allow_exports = ":test",
        )

        package_group(
            name = "test",
            packages = ["//allowed"],
        )
        """);
    scratch.file(
        "allowed/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        """
        proto_library(
            name = "test1",
            deps = ["//x:foo"],
        )

        proto_library(
            name = "test2",
            srcs = ["A.proto"],
            exports = ["//x:foo"],
        )
        """);

    getConfiguredTarget("//allowed:test1");
    getConfiguredTarget("//allowed:test2");

    assertNoEvents();
  }

  @Test
  public void protoLibrary_implcitReexport_fails() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        """
        proto_library(
            name = "foo",
            srcs = ["foo.proto"],
            allow_exports = ":test",
        )

        package_group(
            name = "test",
            packages = ["//allowed"],
        )
        """);
    scratch.file(
        "notallowed/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='test', deps = ['//x:foo'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//notallowed:test");

    assertContainsEvent("proto_library '@@//x:foo' can't be reexported in package '//notallowed'");
  }

  @Test
  public void protoLibrary_explicitExport_fails() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')
        proto_library(
            name = "foo",
            srcs = ["foo.proto"],
            allow_exports = ":test",
        )

        package_group(
            name = "test",
            packages = ["//allowed"],
        )
        """);
    scratch.file(
        "notallowed/BUILD",
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "proto_library(name='test', srcs = ['A.proto'], exports = ['//x:foo'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//notallowed:test");

    assertContainsEvent("proto_library '@@//x:foo' can't be reexported in package '//notallowed'");
  }

  private ProtoInfo getProtoInfo(ConfiguredTarget target) throws Exception {
    for (var key : ProtoConstants.EXTERNAL_PROTO_INFO_KEYS) {
      ProtoInfoProvider providerClass = new ProtoInfoProvider(key);
      ProtoInfo provider = target.get(providerClass);
      if (provider != null) {
        return provider;
      }
    }
    throw new IllegalStateException("ProtoInfo not found in " + target.toString());
  }
}
