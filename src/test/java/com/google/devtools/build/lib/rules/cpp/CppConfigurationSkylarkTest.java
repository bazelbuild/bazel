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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for C++ fragments in Starlark. */
@RunWith(JUnit4.class)
public final class CppConfigurationSkylarkTest extends BuildViewTestCase {

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
  public void testLinkopts() throws Exception {
    writeRuleReturning("ctx.fragments.cpp.linkopts");
    useConfiguration("--linkopt=-wololoo");

    @SuppressWarnings("unchecked")
    Sequence<String> result = (Sequence<String>) getConfiguredTarget("//foo:bar").get("result");
    assertThat(result).containsExactly("-wololoo");
  }

  private void writeRuleReturning(String s) throws IOException {
    scratch.file(
        "foo/lib.bzl",
        "def _impl(ctx):",
        "  return struct(",
        "    result = " + s,
        "  )",
        "foo = rule(implementation=_impl, fragments = ['cpp'])");
    scratch.file("foo/BUILD", "load(':lib.bzl', 'foo')", "foo(name='bar')");
  }
}
