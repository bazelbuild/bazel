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
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link PackageGroupConfiguredTarget}.
 */
@RunWith(JUnit4.class)
public final class PackageGroupBuildViewTest extends BuildViewTestCase {
  @Override
  protected boolean allowExternalRepositories() {
    return true;
  }

  /** Regression test for bug #3445835. */
  @Test
  public void testPackageGroupInDeps() throws Exception {
    checkError(
        "foo",
        "bar",
        "in deps attribute of cc_library rule //foo:bar: "
            + "package group '//foo:foo' is misplaced here ",
        "package_group(name = 'foo', packages = ['//none'])",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name = 'bar', deps = [':foo'])");
  }

  @Test
  public void testPackageGroupInData() throws Exception {
    checkError(
        "foo",
        "bar",
        "in data attribute of cc_library rule //foo:bar: "
            + "package group '//foo:foo' is misplaced here ",
        "package_group(name = 'foo', packages = ['//none'])",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name = 'bar', data = [':foo'])");
  }

  @Test
  public void testPackageGroupWithAllPackagesInMainRepository() throws Exception {
    scratch.file(
        "fruits/BUILD", "package_group(", "    name = 'apple',", "    packages = ['@//...'],", ")");

    PackageGroupConfiguredTarget pg =
        (PackageGroupConfiguredTarget) getConfiguredTarget("//fruits:apple");
    PackageSpecificationProvider provider = pg.getProvider(PackageSpecificationProvider.class);
    assertThat(provider.targetInAllowlist(Label.parseCanonical("//any/pkg:target"))).isTrue();
  }

  @Test
  public void testPackageGroupWithRepoMapping() throws Exception {
    registry.addModule(createModuleKey("veggies", "1.0"), "module(name='veggies', version='1.0')");

    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='main', version='1.0')",
        "bazel_dep(name='veggies', version='1.0', repo_name='my_veggies')");

    invalidatePackages();

    scratch.file(
        "fruits/BUILD",
        "package_group(",
        "    name = 'banana',",
        "    packages = ['@my_veggies//cucumber'],",
        ")");

    PackageGroupConfiguredTarget pg =
        (PackageGroupConfiguredTarget) getConfiguredTarget("//fruits:banana");
    PackageSpecificationProvider provider = pg.getProvider(PackageSpecificationProvider.class);

    assertThat(
            provider.targetInAllowlist(
                Label.parseWithRepoContext(
                    "@my_veggies//cucumber:something",
                    Label.RepoContext.of(
                        pg.getLabel().getRepository(),
                        skyframeExecutor.getMainRepoMapping(reporter)))))
        .isTrue();
  }
}
