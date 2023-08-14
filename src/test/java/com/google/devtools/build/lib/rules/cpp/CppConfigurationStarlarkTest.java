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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.io.IOException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for C++ fragments in Starlark. */
@RunWith(JUnit4.class)
public final class CppConfigurationStarlarkTest extends BuildViewTestCase {

  @Test
  public void testMinimumOsVersion() throws Exception {
    useConfiguration("--minimum_os_version=-wololoo");
    writeRuleReturning("ctx.fragments.cpp.minimum_os_version()");

    String result = (String) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).isEqualTo("-wololoo");
  }

  @Test
  public void testNullMinimumOsVersion() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.minimum_os_version()");

    Object result = getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).isInstanceOf(NoneType.class);
  }

  @Test
  public void testCopts() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.copts");
    useConfiguration("--copt=-wololoo");

    @SuppressWarnings("unchecked")
    Sequence<String> result = (Sequence<String>) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).containsExactly("-wololoo");
  }

  @Test
  public void testCxxopts() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.cxxopts");
    useConfiguration("--cxxopt=-wololoo");

    @SuppressWarnings("unchecked")
    Sequence<String> result = (Sequence<String>) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).containsExactly("-wololoo");
  }

  @Test
  public void testConlyopts() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.conlyopts");
    useConfiguration("--conlyopt=-wololoo");

    @SuppressWarnings("unchecked")
    Sequence<String> result = (Sequence<String>) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).containsExactly("-wololoo");
  }

  @Test
  public void testObjcopts() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.objccopts");
    useConfiguration("--objccopt=-wololoo");

    @SuppressWarnings("unchecked")
    Sequence<String> result = (Sequence<String>) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).containsExactly("-wololoo");
  }

  @Test
  public void testLinkopts() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.linkopts");
    useConfiguration("--linkopt=-wololoo");

    @SuppressWarnings("unchecked")
    Sequence<String> result = (Sequence<String>) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).containsExactly("-wololoo");
  }

  private static void assertBlockedFeature(AssertionError e, String feature) {
    assertThat(e)
        .hasMessageThat()
        .contains(
            String.format("cannot use private API (feature '%s' in CppConfiguration)", feature));
  }

  @Test
  public void testExpandedApiBlocked() throws Exception {
    writeRuleReturning("foo", "pic.bzl", "pic", "ctx.fragments.cpp.force_pic()");
    writeRuleReturning("foo", "lcov.bzl", "lcov", "ctx.fragments.cpp.generate_llvm_lcov()");
    writeRuleReturning("foo", "fdo.bzl", "fdo", "ctx.fragments.cpp.fdo_instrument()");
    writeRuleReturning(
        "foo", "hdr_deps.bzl", "hdr_deps", "ctx.fragments.cpp.process_headers_in_dependencies()");
    writeRuleReturning("foo", "save.bzl", "save", "ctx.fragments.cpp.save_feature_state()");
    writeRuleReturning(
        "foo",
        "fission.bzl",
        "fission",
        "ctx.fragments.cpp.fission_active_for_current_compilation_mode()");
    AssertionError e;
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:pic"));
    assertBlockedFeature(e, "force_pic");
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:lcov"));
    assertBlockedFeature(e, "generate_llvm_lcov");
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:fdo"));
    assertBlockedFeature(e, "fdo_instrument");
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:hdr_deps"));
    assertThat(e).hasMessageThat().contains("cannot use private API");
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:save"));
    assertThat(e).hasMessageThat().contains("cannot use private API");
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:fission"));
    assertBlockedFeature(e, "fission_active_for_current_compilation_mode");
  }

  private void writeRuleReturning(String returns) throws IOException {
    writeRuleReturning("foo", "lib.bzl", "bar", returns);
  }

  private void writeRuleReturning(String path, String lib, String target, String returns)
      throws IOException {
    scratch.file(
        path + "/" + lib,
        "def _impl(ctx):",
        "  return struct(",
        "    result = " + returns,
        "  )",
        "foo = rule(implementation=_impl, fragments = ['cpp'])");
    scratch.appendFile(
        path + "/BUILD", "load(':" + lib + "', 'foo')", "foo(name='" + target + "')");
  }
}
