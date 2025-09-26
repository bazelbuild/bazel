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
    assertTransitiveVisibilityContainsPackages(provider, "//pkg");

    // output_file
    target = getConfiguredTarget("//pkg:target.out");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertTransitiveVisibilityContainsPackages(provider, "//pkg");

    // input_file
    target = getConfiguredTarget("//pkg:message.txt");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertTransitiveVisibilityContainsPackages(provider, "//pkg");

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
    assertTransitiveVisibilityContainsPackages(provider, "//pkg", "//tv_pkg");

    // Also check that the dep itself has the provider.
    ConfiguredTarget depTarget = getConfiguredTarget("//tv_pkg:dep");
    provider = depTarget.getProvider(TransitiveVisibilityProvider.class);
    assertTransitiveVisibilityContainsPackages(provider, "//pkg", "//tv_pkg");

    // output_file
    target = getConfiguredTarget("//pkg:target.out");
    provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertTransitiveVisibilityContainsPackages(provider, "//pkg", "//tv_pkg");
  }

  @Test
  public void targetsDependingOnTargetWithTransitiveVisibility_failIfNotTransitivelyVisible()
      throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file("tv_pkg/message.txt", "Hello, world!");
    scratch.file(
        "tv_pkg/BUILD",
        "package(transitive_visibility = ':tv1')",
        "exports_files(['message.txt'])",
        "package_group(name = 'tv1', packages = ['//tv_pkg/...'])",
        "genrule(name = 'dep', outs = ['dep.out'], cmd = 'touch $@')");
    scratch.file(
        "pkg/BUILD",
        """
        genrule(
            name = 'depends_on_genrule',
            outs = ['depends_on_genrule.out'],
            srcs = ['//tv_pkg:dep'],
            cmd = 'touch $@'
        )
        genrule(
            name = 'depends_on_output_file',
            outs = ['depends_on_output_file.out'],
            srcs = ['//tv_pkg:dep.out'],
            cmd = 'touch $@'
        )

        genrule(
            name = 'depends_on_input_file',
            outs = ['depends_on_input_file.out'],
            srcs = ['//tv_pkg:message.txt'],
            cmd = 'touch $@'
        )
        package_group(
            name = 'depends_on_package_group',
            includes = ['//tv_pkg:tv1']
        )
        """);

    // genrule depends on genrule
    reporter.removeHandler(failFastHandler);
    reporter.clearEventBus();
    getConfiguredTarget("//pkg:depends_on_genrule");
    assertContainsEvent(
        "Transitive visibility error: //tv_pkg:dep is not transitively visible from"
            + " //pkg:depends_on_genrule. //tv_pkg:dep inherits a transitive_visibility declaration"
            + " from its package or one of its dependencies that does not allow"
            + " //pkg:depends_on_genrule");

    // genrule depends on output_file
    getConfiguredTarget("//pkg:depends_on_output_file");
    assertContainsEvent(
        "Transitive visibility error: //tv_pkg:dep.out is not transitively visible from"
            + " //pkg:depends_on_output_file. //tv_pkg:dep.out inherits a transitive_visibility"
            + " declaration from its package or one of its dependencies that does not allow"
            + " //pkg:depends_on_output_file");

    // genrule depends on input_file
    getConfiguredTarget("//pkg:depends_on_input_file");
    assertContainsEvent(
        "Transitive visibility error: //tv_pkg:message.txt is not transitively visible from"
            + " //pkg:depends_on_input_file. //tv_pkg:message.txt inherits a transitive_visibility"
            + " declaration from its package or one of its dependencies that does not allow"
            + " //pkg:depends_on_input_file");

    // genrule depends on package_group -- no enforcement by design
    reporter.addHandler(failFastHandler);
    ConfiguredTarget target = getConfiguredTarget("//pkg:depends_on_package_group");
    // Mostly just checking that this target exists.
    TransitiveVisibilityProvider provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertThat(provider).isNull();
  }

  @Test
  public void targetsDependingOnTargetWithTransitiveVisibility_failIfTransitivelyNotVisible()
      throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "a/BUILD",
        "package(transitive_visibility = ':tv1')",
        "package_group(name = 'tv1', packages = ['//a/...', '//b/...'])",
        "genrule(name = 'a', outs = ['a.out'], cmd = 'touch $@')");
    scratch.file(
        "b/BUILD", "genrule(name = 'b', srcs = ['//a'], outs = ['b.out'], cmd = 'touch $@')");
    scratch.file(
        "c/BUILD", "genrule(name = 'c', srcs = ['//b'], outs = ['c.out'], cmd = 'touch $@')");

    reporter.removeHandler(failFastHandler);
    reporter.clearEventBus();
    getConfiguredTarget("//c:c");
    assertContainsEvent(
        "Transitive visibility error: //b:b is not transitively visible from //c:c. //b:b"
            + " inherits a transitive_visibility declaration from its package or one of its"
            + " dependencies that does not allow //c:c");
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
  public void transitiveVisibilityUsedInADifferentPackage() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "pkg/BUILD",
        "package(transitive_visibility = ':tv1')",
        "package_group(name = 'tv1', packages = ['//pkg/...'])",
        "package_group(name = 'other_package_group', packages = ['//pkg2/...'])",
        "genrule(name = 'target', outs = ['target.out'], cmd = 'touch $@')");
    scratch.file(
        "pkg2/BUILD", "package(transitive_visibility = ['//pkg:tv1'])", "fail I am a bad package");
    ConfiguredTarget target = getConfiguredTarget("//pkg:target");
    assertTransitiveVisibilityContainsPackages(
        target.getProvider(TransitiveVisibilityProvider.class), "//pkg");
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

  @Test
  public void diamondDep_topLevelAllowed() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "d/BUILD",
        """
        package(transitive_visibility = ':tv_d')
        package_group(name = 'tv_d', packages = ['//a/...', '//b/...', '//c/...', '//d/...'])
        genrule(name = 'd', outs = ['d.out'], cmd = 'touch $@')
        """);
    scratch.file(
        "b/BUILD", "genrule(name = 'b', srcs = ['//d'], outs = ['b.out'], cmd = 'touch $@')");
    scratch.file(
        "c/BUILD", "genrule(name = 'c', srcs = ['//d'], outs = ['c.out'], cmd = 'touch $@')");
    scratch.file(
        "a/BUILD",
        "genrule(name = 'a', srcs = ['//b', '//c'], outs = ['a.out'], cmd = 'touch $@')");

    ConfiguredTarget target = getConfiguredTarget("//a:a");
    TransitiveVisibilityProvider provider = target.getProvider(TransitiveVisibilityProvider.class);
    assertTransitiveVisibilityContainsPackages(provider, "//a", "//b", "//c", "//d");
  }

  @Test
  public void diamondDep_topLevelNotAllowed() throws Exception {
    useConfiguration("--experimental_enforce_transitive_visibility=true");
    scratch.file(
        "d/BUILD",
        """
        package(transitive_visibility = ':tv_d')
        package_group(name = 'tv_d', packages = ['//b/...', '//c/...', '//d/...'])
        genrule(name = 'd', outs = ['d.out'], cmd = 'touch $@')
        """);
    scratch.file(
        "b/BUILD", "genrule(name = 'b', srcs = ['//d'], outs = ['b.out'], cmd = 'touch $@')");
    scratch.file(
        "c/BUILD", "genrule(name = 'c', srcs = ['//d'], outs = ['c.out'], cmd = 'touch $@')");
    scratch.file(
        "a/BUILD",
        "genrule(name = 'a', srcs = ['//b', '//c'], outs = ['a.out'], cmd = 'touch $@')");

    reporter.removeHandler(failFastHandler);
    reporter.clearEventBus();
    getConfiguredTarget("//a:a");
    assertContainsEvent(
        "Transitive visibility error: //b:b is not transitively visible from //a:a. //b:b"
            + " inherits a transitive_visibility declaration from its package or one of its"
            + " dependencies that does not allow //a:a");
  }

  private void assertTransitiveVisibilityContainsPackages(
      TransitiveVisibilityProvider provider, String... packages) throws Exception {
    assertThat(provider).isNotNull();
    for (String pkg : packages) {
      for (PackageSpecificationProvider restrictionSet : provider.getTransitiveVisibility()) {
        assertThat(restrictionSet.targetInAllowlist(pkg)).isTrue();
      }
    }
  }
}
