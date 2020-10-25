// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for Bazel's *AND* Blaze's {@code proto_library}. */
@RunWith(JUnit4.class)
public class SharedProtoLibraryTest extends BuildViewTestCase {
  private boolean isThisBazel() {
    return getAnalysisMock().isThisBazel();
  }

  /** If this is Blaze, simulate behaviour of Bazel's {@code proto_library} rule. */
  private void simulateBazel() throws Exception {
    if (!isThisBazel()) {
      useConfiguration("--incompatible_generated_protos_in_virtual_imports=true");
    }
  }

  /** If this is Bazel, simulate behaviour of Blaze's {@code proto_library} rule. */
  private void simulateBlaze() throws Exception {
    if (isThisBazel()) {
      useConfiguration("--incompatible_generated_protos_in_virtual_imports=false");
    }
  }

  @Before
  public void setUp() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler");
    scratch.file("proto/BUILD", "licenses(['notice'])", "exports_files(['compiler'])");

    if (isThisBazel()) {
      scratch.file(
          "proto/defs.bzl",
          "load('@rules_proto//proto:defs.bzl', proto_lib = 'proto_library')",
          "def proto_library(**kwargs):",
          "  proto_lib(**kwargs)");
    } else {
      // This is Blaze.
      scratch.file(
          "proto/defs.bzl", "def proto_library(**kwargs):", "  native.proto_library(**kwargs)");
    }

    MockProtoSupport.setupWorkspace(scratch);
    invalidatePackages();
  }

  @Test
  public void testProtoLibrary() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'proto_library')",
        "proto_library(name='foo', srcs=['a.proto', 'b.proto', 'c.proto'])");

    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceFile().getExecPath().getPathString()))
        .containsExactly("x/a.proto", "x/b.proto", "x/c.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(),
                s -> s.getOriginalSourceFile().getExecPath().getPathString()))
        .containsExactly("x/a.proto", "x/b.proto", "x/c.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceRoot().getSafePathString()))
        .containsExactly(".", ".", ".");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getImportPath().getPathString()))
        .containsExactly("x/a.proto", "x/b.proto", "x/c.proto");
  }

  @Test
  public void testProtoLibraryWithoutSources() throws Exception {
    scratch.file(
        "x/BUILD", "load('//proto:defs.bzl', 'proto_library')", "proto_library(name='foo')");

    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(provider.getDirectSources()).isEmpty();
  }

  @Test
  public void testProtoLibraryWithVirtualProtoSourceRoot() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'proto_library')",
        "proto_library(name='foo', srcs=['a.proto'], import_prefix='foo')");

    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/_virtual_imports/foo/foo/x/a.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(),
                s -> s.getOriginalSourceFile().getExecPath().getPathString()))
        .containsExactly("x/a.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceRoot().getSafePathString()))
        .containsExactly(genfiles + "/x/_virtual_imports/foo");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getImportPath().getPathString()))
        .containsExactly("foo/x/a.proto");
  }

  @Test
  public void testProtoLibraryWithGeneratedSources_Bazel() throws Exception {
    simulateBazel();

    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'proto_library')",
        "genrule(name='g', srcs=[], outs=['generated.proto'], cmd='')",
        "proto_library(name='foo', srcs=['generated.proto'])");

    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/_virtual_imports/foo/x/generated.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(),
                s -> s.getOriginalSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/generated.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceRoot().getSafePathString()))
        .containsExactly(genfiles + "/x/_virtual_imports/foo");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getImportPath().getPathString()))
        .containsExactly("x/generated.proto");
  }

  @Test
  public void testProtoLibraryWithGeneratedSources_Blaze() throws Exception {
    simulateBlaze();

    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'proto_library')",
        "genrule(name='g', srcs=[], outs=['generated.proto'], cmd='')",
        "proto_library(name='foo', srcs=['generated.proto'])");

    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/generated.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(),
                s -> s.getOriginalSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/generated.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceRoot().getSafePathString()))
        .containsExactly(genfiles);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getImportPath().getPathString()))
        .containsExactly("x/generated.proto");
  }

  @Test
  public void testProtoLibraryWithMixedSources_Bazel() throws Exception {
    simulateBazel();

    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'proto_library')",
        "genrule(name='g', srcs=[], outs=['generated.proto'], cmd='')",
        "proto_library(name='foo', srcs=['generated.proto', 'a.proto'])");

    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceFile().getExecPath().getPathString()))
        .containsExactly(
            genfiles + "/x/_virtual_imports/foo/x/generated.proto",
            genfiles + "/x/_virtual_imports/foo/x/a.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(),
                s -> s.getOriginalSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/generated.proto", "x/a.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceRoot().getSafePathString()))
        .containsExactly(
            genfiles + "/x/_virtual_imports/foo", genfiles + "/x/_virtual_imports/foo");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getImportPath().getPathString()))
        .containsExactly("x/generated.proto", "x/a.proto");
  }

  @Test
  public void testProtoLibraryWithMixedSources_Blaze() throws Exception {
    simulateBlaze();

    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'proto_library')",
        "genrule(name='g', srcs=[], outs=['generated.proto'], cmd='')",
        "proto_library(name='foo', srcs=['generated.proto', 'a.proto'])");

    String genfiles = getTargetConfiguration().getGenfilesFragment(RepositoryName.MAIN).toString();
    ProtoInfo provider = getConfiguredTarget("//x:foo").get(ProtoInfo.PROVIDER);
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/generated.proto", "x/a.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(),
                s -> s.getOriginalSourceFile().getExecPath().getPathString()))
        .containsExactly(genfiles + "/x/generated.proto", "x/a.proto");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getSourceRoot().getSafePathString()))
        .containsExactly(genfiles, ".");
    assertThat(
            Iterables.transform(
                provider.getDirectSources(), s -> s.getImportPath().getPathString()))
        .containsExactly("x/generated.proto", "x/a.proto");
  }
}
