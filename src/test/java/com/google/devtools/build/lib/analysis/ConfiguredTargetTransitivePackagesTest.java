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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorRepositoryHelpersHolder;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test that checks collected list of transitive targets of configured targets. */
@RunWith(JUnit4.class)
public final class ConfiguredTargetTransitivePackagesTest extends AnalysisTestCase {

  @Before
  public void setUpToolsConfigMock() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
  }

  @Override
  protected SkyframeExecutorRepositoryHelpersHolder getRepositoryHelpersHolder() {
    // Transitive packages are only stored when external repositories are enabled.
    return SkyframeExecutorRepositoryHelpersHolder.create(
        new RepositoryDirectoryDirtinessChecker());
  }

  private void assertTransitiveClosureOfTargetContainsPackages(
      String target, BuildConfigurationValue config, String... packages) throws Exception {
    ConfiguredTargetValue ctValue =
        SkyframeExecutorTestUtils.getExistingConfiguredTargetValue(
            skyframeExecutor, Label.parseCanonical(target), config);
    ImmutableSet<String> packageNames =
        ctValue.getTransitivePackages().toList().stream()
            .map(pkg -> pkg.getPackageIdentifier().toString())
            .collect(toImmutableSet());
    assertThat(packageNames).containsAtLeastElementsIn(Sets.newHashSet(packages));
  }

  @Test
  public void testSimpleConfiguredTarget() throws Exception {
    scratch.file("a/BUILD", "sh_library(name = 'a', deps = [ '//a/b:b' ])");
    scratch.file("a/b/BUILD", "sh_library(name = 'b', deps = [ '//c:c', '//d:d'] )");
    scratch.file("c/BUILD", "sh_library(name = 'c')");
    scratch.file("d/BUILD", "sh_library(name = 'd')");

    ConfiguredTarget target = Iterables.getOnlyElement(update("//a:a").getTargetsToBuild());
    BuildConfigurationValue config = getConfiguration(target);

    assertTransitiveClosureOfTargetContainsPackages("//a:a", config, "a", "a/b", "c", "d");
    assertTransitiveClosureOfTargetContainsPackages("//a/b:b", config, "a/b", "c", "d");
    assertTransitiveClosureOfTargetContainsPackages("//c:c", config, "c");
    assertTransitiveClosureOfTargetContainsPackages("//d:d", config, "d");
  }

  @Test
  public void testPackagesFromAspects() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TestAspects.EXTRA_ATTRIBUTE_ASPECT_RULE);
    scratch.file("extra/BUILD", "base(name = 'extra')");
    scratch.file(
        "a/c/BUILD",
        """
        rule_with_extra_deps_aspect(
            name = "foo",
            foo = [":bar"],
        )

        base(name = "bar")
        """);

    ConfiguredTarget target = Iterables.getOnlyElement(update("//a/c:foo").getTargetsToBuild());
    BuildConfigurationValue config = getConfiguration(target);

    // We expect 'extra' package because rule_with_extra_deps adds an aspect on attribute 'foo' with
    // '//extra:extra' dependency.
    assertTransitiveClosureOfTargetContainsPackages("//a/c:foo", config, "a/c", "extra");
  }

  @Test
  public void testTargetsWithConfiguration() throws Exception {
    scratch.file("a/BUILD", "cc_library(name = 'a', srcs = [ 'some.cpp' ])");

    ConfiguredTarget target = Iterables.getOnlyElement(update("//a:a").getTargetsToBuild());
    BuildConfigurationValue config = getConfiguration(target);

    // We expect to get the mock crosstool in transitive dependencies, because it's required for c++
    // configuration.
    assertTransitiveClosureOfTargetContainsPackages(
        "//a:a",
        config,
        "a",
        PackageIdentifier.create(
                TestConstants.TOOLS_REPOSITORY,
                PathFragment.create(TestConstants.MOCK_CC_CROSSTOOL_PATH))
            .toString());
  }
}
