// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the package() function's transitive_visibility argument. */
@RunWith(JUnit4.class)
public class TransitiveVisibilityTest extends BuildViewTestCase {

  @Test
  public void targetsInPackageWithTransitiveVisibility_allHaveProvider() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file("pkg/message.txt", "Hello, world!");
    scratch.file(
        "pkg/BUILD",
        "package(transitive_visibility = ':tv1')",
        "package_group(name = 'tv1', packages = ['//pkg/...'])",
        "package_group(name = 'other_package_group', packages = ['//pkg/...'])",
        "genrule(name = 'target', srcs = ['message.txt'], outs = ['target.out'], cmd = 'cat"
            + " message.txt > $@')");

    // genrule
    ConfiguredTarget target = getConfiguredTarget("//pkg:target");
    TransitiveVisibilityProvider provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//pkg:tv1"));

    // output_file
    target = getConfiguredTarget("//pkg:target.out");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//pkg:tv1"));

    // input_file
    target = getConfiguredTarget("//pkg:message.txt");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//pkg:tv1"));

    // package_group
    target = getConfiguredTarget("//pkg:other_package_group");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNull();
  }

  @Test
  public void targetsDependingOnTargetWithTransitiveVisibility_allHaveProvider() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "tv_pkg/BUILD",
        "package(transitive_visibility = ':tv1')",
        "package_group(name = 'tv1', packages = ['//tv_pkg/...', '//pkg/...'])",
        "genrule(name = 'dep', outs = ['dep.out'], cmd = 'touch $@')");
    scratch.file(
        "pkg/BUILD",
        """
        genrule(
            name = 'target',
            outs = ['target.out'],
            srcs = ['//tv_pkg:dep', 'message.txt'],
            cmd = 'touch $@'
        )
        """);

    // genrule
    ConfiguredTarget target = getConfiguredTarget("//pkg:target");
    TransitiveVisibilityProvider provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//tv_pkg:tv1"));

    // Also check that the dep itself has the provider.
    ConfiguredTarget depTarget = getConfiguredTarget("//tv_pkg:dep");
    provider = depTarget.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//tv_pkg:tv1"));

    // output_file
    target = getConfiguredTarget("//pkg:target.out");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//tv_pkg:tv1"));
  }

  @Test
  public void transitiveVisibilityDeclaredWithWrongType() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "pkg/BUILD",
        "package(transitive_visibility = [':a_list'])",
        "genrule(name='a', outs=['a.out'], cmd='touch $@')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//pkg:a");
    assertContainsEvent(
        "Error in package: expected value of type 'string' for package() argument"
            + " 'transitive_visibility', but got [\":a_list\"] (list)");
  }

  @Test
  public void targetInheritsProviderFromOutputFileDep() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "tv_pkg/BUILD",
        "package(transitive_visibility = ':tv1')",
        "package_group(name = 'tv1', packages = ['//tv_pkg/...', '//pkg/...'])",
        "genrule(name='gen', outs=['out'], cmd='touch $@')");
    scratch.file(
        "pkg/BUILD",
        "genrule(name = 'target', outs = ['target.out'], srcs = ['//tv_pkg:out'], cmd = 'touch"
            + " $@')");
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget target = getConfiguredTarget("//pkg:target");
    TransitiveVisibilityProvider provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getTransitiveVisibility())
        .containsExactly(Label.parseCanonical("//tv_pkg:tv1"));
  }

  @Test
  public void providerNotCreatedWhenEnforcementDisabled() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=false");
    scratch.file(
        "tv_pkg/BUILD",
        "package(transitive_visibility = ':tv1')",
        "package_group(name = 'tv1', packages = ['//tv_pkg/...', '//pkg/...'])",
        "genrule(name = 'dep', outs = ['dep.out'], cmd = 'touch $@')");
    scratch.file(
        "pkg/BUILD",
        "genrule(name = 'target', outs = ['target.out'], srcs = ['//tv_pkg:dep'], cmd = 'touch"
            + " $@')");
    ConfiguredTarget target = getConfiguredTarget("//pkg:target");
    assertThat(target.getProvider(TransitiveVisibilityProvider.class)).isNull();
  }
}
