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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RequiredConfigFragmentsProvider}. */
@RunWith(JUnit4.class)
public final class RequiredConfigFragmentsTest extends BuildViewTestCase {
  @Test
  public void provideTransitiveRequiredFragmentsMode() throws Exception {
    useConfiguration("--include_config_fragments_provider=transitive");
    scratch.file(
        "a/BUILD",
        "config_setting(name = 'config', values = {'start_end_lib': '1'})",
        "py_library(name = 'pylib', srcs = ['pylib.py'])",
        "cc_library(name = 'a', srcs = ['A.cc'], data = [':pylib'])");

    ImmutableSet<String> ccLibTransitiveFragments =
        getConfiguredTarget("//a:a")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(ccLibTransitiveFragments).containsAtLeast("CppConfiguration", "PythonConfiguration");

    ImmutableSet<String> configSettingTransitiveFragments =
        getConfiguredTarget("//a:config")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(configSettingTransitiveFragments).contains("CppOptions");
  }

  @Test
  public void provideDirectRequiredFragmentsMode() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct");
    scratch.file(
        "a/BUILD",
        "config_setting(name = 'config', values = {'start_end_lib': '1'})",
        "py_library(name = 'pylib', srcs = ['pylib.py'])",
        "cc_library(name = 'a', srcs = ['A.cc'], data = [':pylib'])");

    ImmutableSet<String> ccLibDirectFragments =
        getConfiguredTarget("//a:a")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(ccLibDirectFragments).contains("CppConfiguration");
    assertThat(ccLibDirectFragments).doesNotContain("PythonConfiguration");

    ImmutableSet<String> configSettingDirectFragments =
        getConfiguredTarget("//a:config")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(configSettingDirectFragments).contains("CppOptions");
  }

  /**
   * Helper method that returns a combined set of the common fragments all genrules require plus
   * instance-specific requirements passed here.
   */
  private ImmutableSortedSet<String> genRuleFragments(String... targetSpecificRequirements)
      throws Exception {
    scratch.file(
        "base_genrule/BUILD",
        "genrule(",
        "    name = 'base_genrule',",
        "    srcs = [],",
        "    outs = ['base_genrule.out'],",
        "    cmd = 'echo hi > $@')");
    ImmutableSortedSet.Builder<String> builder = ImmutableSortedSet.naturalOrder();
    builder.add(targetSpecificRequirements);
    builder.addAll(
        getConfiguredTarget("//base_genrule")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments());
    return builder.build();
  }

  @Test
  public void requiresMakeVariablesSuppliedByDefine() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct", "--define", "myvar=myval");
    scratch.file(
        "a/BUILD",
        "genrule(",
        "    name = 'myrule',",
        "    srcs = [],",
        "    outs = ['myrule.out'],",
        "    cmd = 'echo $(myvar) $(COMPILATION_MODE) > $@')");
    ImmutableSet<String> requiredFragments =
        getConfiguredTarget("//a:myrule")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(requiredFragments)
        .containsExactlyElementsIn(genRuleFragments("--define:myvar"))
        .inOrder();
  }

  @Test
  public void starlarkExpandMakeVariables() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct", "--define", "myvar=myval");
    scratch.file(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  print(ctx.expand_make_variables('dummy attribute', 'string with $(myvar)!', {}))",
        "",
        "simple_rule = rule(",
        "  implementation = _impl,",
        "   attrs = {}",
        ")");
    scratch.file("a/BUILD", "load('//a:defs.bzl', 'simple_rule')", "simple_rule(name = 'simple')");
    ImmutableSet<String> requiredFragments =
        getConfiguredTarget("//a:simple")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(requiredFragments)
        .containsExactlyElementsIn(genRuleFragments("--define:myvar"))
        .inOrder();
  }
}
