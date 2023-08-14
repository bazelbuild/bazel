// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.YankedVersionsUtil;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class RepoFileFunctionTest extends BuildViewTestCase {

  private Path moduleRoot;
  private FakeRegistry registry;

  @Override
  protected ImmutableList<Injected> extraPrecomputedValues() {
    try {
      moduleRoot = scratch.dir("modules");
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    registry = FakeRegistry.DEFAULT_FACTORY.newFakeRegistry(moduleRoot.getPathString());
    return ImmutableList.of(
        PrecomputedValue.injected(
            ModuleFileFunction.REGISTRIES, ImmutableList.of(registry.getUrl())),
        PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, false),
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, ImmutableMap.of()),
        PrecomputedValue.injected(YankedVersionsUtil.ALLOWED_YANKED_VERSIONS, ImmutableList.of()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, CheckDirectDepsMode.WARNING),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE, BazelCompatibilityMode.ERROR),
        PrecomputedValue.injected(BazelLockFileFunction.LOCKFILE_MODE, LockfileMode.UPDATE));
  }

  @Test
  public void defaultVisibility() throws Exception {
    scratch.overwriteFile("REPO.bazel", "repo(default_visibility=['//some:thing'])");
    scratch.overwriteFile("p/BUILD", "sh_library(name = 't')");
    Target t = getTarget("//p:t");
    assertThat(t.getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonical("//some:thing"));
  }

  @Test
  public void repoFileInTheMainRepo() throws Exception {
    scratch.overwriteFile("REPO.bazel", "repo(default_deprecation='EVERYTHING IS DEPRECATED')");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    assertThat(
            getRuleContext(getConfiguredTarget("//abc/def:what"))
                .attributes()
                .get("deprecation", Type.STRING))
        .isEqualTo("EVERYTHING IS DEPRECATED");
  }

  @Test
  public void repoFileInAnExternalRepo() throws Exception {
    setBuildLanguageOptions("--enable_bzlmod");
    scratch.overwriteFile("MODULE.bazel", "bazel_dep(name='foo',version='1.0')");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    registry.addModule(createModuleKey("foo", "1.0"), "module(name='foo',version='1.0')");
    scratch.overwriteFile(moduleRoot.getRelative("foo~1.0/WORKSPACE.bazel").getPathString());
    scratch.overwriteFile(
        moduleRoot.getRelative("foo~1.0/REPO.bazel").getPathString(),
        "repo(default_deprecation='EVERYTHING IS DEPRECATED')");
    scratch.overwriteFile(
        moduleRoot.getRelative("foo~1.0/abc/def/BUILD").getPathString(), "filegroup(name='what')");

    assertThat(
            getRuleContext(getConfiguredTarget("//abc/def:what"))
                .attributes()
                .get("deprecation", Type.STRING))
        .isNull();
    assertThat(
            getRuleContext(getConfiguredTarget("@@foo~1.0//abc/def:what"))
                .attributes()
                .get("deprecation", Type.STRING))
        .isEqualTo("EVERYTHING IS DEPRECATED");
  }

  @Test
  public void cantCallRepoTwice() throws Exception {
    scratch.overwriteFile(
        "REPO.bazel",
        "repo(default_deprecation='EVERYTHING IS DEPRECATED')",
        "repo(features=['abc'])");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    reporter.removeHandler(failFastHandler);
    assertTargetError("//abc/def:what", "'repo' can only be called once");
  }

  @Test
  public void featureMerger() throws Exception {
    scratch.overwriteFile("REPO.bazel", "repo(features=['a', 'b', 'c', '-d'])");
    scratch.overwriteFile(
        "abc/def/BUILD",
        "package(features=['-a','-b','d'])",
        "filegroup(name='what', features=['b'])");
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//abc/def:what"));
    assertThat(ruleContext.getFeatures()).containsExactly("b", "c", "d");
    assertThat(ruleContext.getDisabledFeatures()).containsExactly("a");
  }

  @Test
  public void restrictedSyntax() throws Exception {
    scratch.overwriteFile(
        "REPO.bazel", "if 3+5>7: repo(default_deprecation='EVERYTHING IS DEPRECATED')");
    scratch.overwriteFile("abc/def/BUILD", "filegroup(name='what')");
    reporter.removeHandler(failFastHandler);
    assertTargetError(
        "//abc/def:what",
        "`if` statements are not allowed in REPO.bazel files. You may use an `if` expression for"
            + " simple cases.");
  }
}
