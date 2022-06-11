// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepositoryMappingFunction} and {@link RepositoryMappingValue}. */
@RunWith(JUnit4.class)
public class RepositoryMappingFunctionTest extends BuildViewTestCase {
  private FakeRegistry registry;

  private EvaluationResult<RepositoryMappingValue> eval(SkyKey key)
      throws InterruptedException, AbruptExitException {
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("WORKSPACE")).build(),
            Root.fromPath(rootDirectory));
    return SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
  }

  @Override
  protected boolean enableBzlmod() {
    return true;
  }

  @Before
  public void setUpForBzlmod() throws IOException, ParseException {
    scratch.file("MODULE.bazel", "module()");
    registry = FakeRegistry.DEFAULT_FACTORY.newFakeRegistry(scratch.dir("modules").getPathString());
    ModuleFileFunction.REGISTRIES.set(
        getSkyframeExecutor().getDifferencerForTesting(), ImmutableList.of(registry.getUrl()));
    ModuleFileFunction.IGNORE_DEV_DEPS.set(getSkyframeExecutor().getDifferencerForTesting(), false);
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        getSkyframeExecutor().getDifferencerForTesting(), CheckDirectDepsMode.WARNING);
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    // Make sure we don't have built-in modules affecting the dependency graph.
    return new AnalysisMock.Delegate(super.getAnalysisMock()) {
      @Override
      public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
          BlazeDirectories directories) {
        return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
            .putAll(
                Maps.filterKeys(
                    super.getSkyFunctions(directories),
                    fnName -> !fnName.equals(SkyFunctions.MODULE_FILE)))
            .put(
                SkyFunctions.MODULE_FILE,
                new ModuleFileFunction(
                    FakeRegistry.DEFAULT_FACTORY, directories.getWorkspace(), ImmutableMap.of()))
            .buildOrThrow();
      }
    };
  }

  public static RepositoryMappingValue withMappingAllowingFallback(
      ImmutableMap<String, RepositoryName> repositoryMapping) {
    return RepositoryMappingValue.withMapping(
        RepositoryMapping.createAllowingFallback(repositoryMapping));
  }

  public static RepositoryMappingValue withMapping(
      ImmutableMap<String, RepositoryName> repositoryMapping, RepositoryName ownerRepo) {
    return RepositoryMappingValue.withMapping(
        RepositoryMapping.create(repositoryMapping, ownerRepo));
  }

  public RepositoryMappingValue withMappingForRootModule(
      ImmutableMap<String, RepositoryName> repositoryMapping, RepositoryName ownerRepo) {
    ImmutableMap.Builder<String, RepositoryName> allMappings = ImmutableMap.builder();
    allMappings.putAll(repositoryMapping);
    for (String name : analysisMock.getWorkspaceRepos()) {
      allMappings.put(name, RepositoryName.createUnvalidated(name));
    }
    return RepositoryMappingValue.withMapping(
        RepositoryMapping.create(allMappings.buildOrThrow(), ownerRepo));
  }

  @Test
  public void testSimpleMapping() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    repo_mapping = {'@a' : '@b'},",
        ")");
    RepositoryName name = RepositoryName.create("a_remote_repo");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    assertThat(result.hasError()).isFalse();
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingAllowingFallback(
                ImmutableMap.of(
                    "a", RepositoryName.create("b"), "good", RepositoryName.create(""))));
  }

  @Test
  public void testRepoNameMapping_asMainModule() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0', repo_name = 'com_foo_bar_b')");
    registry.addModule(createModuleKey("B", "1.0"), "module(name='B', version='1.0')");

    RepositoryName name = RepositoryName.MAIN;
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    assertThat(result.hasError()).isFalse();
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingForRootModule(
                ImmutableMap.of(
                    "", RepositoryName.create(""),
                    "A", RepositoryName.create(""),
                    "com_foo_bar_b", RepositoryName.create("@B.1.0")),
                name));
  }

  @Test
  public void testRepoNameMapping_asDependency() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='1.0', repo_name = 'com_foo_bar_c')");
    registry
        .addModule(createModuleKey("B", "1.0"), "module(name='B', version='1.0')")
        .addModule(
            createModuleKey("C", "1.0"),
            "module(name='C', version='1.0')",
            "bazel_dep(name='B', version='1.0', repo_name='com_foo_bar_b')");

    RepositoryName name = RepositoryName.create("@C.1.0");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    assertThat(result.hasError()).isFalse();
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMapping(
                ImmutableMap.of(
                    "C", RepositoryName.create("@C.1.0"),
                    "com_foo_bar_b", RepositoryName.create("@B.1.0")),
                name));
  }

  @Test
  public void testRepoNameMapping_dependencyOnRootModule() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel", "module(name='A',version='0.1')", "bazel_dep(name='B',version='1.0')");
    registry.addModule(
        createModuleKey("B", "1.0"),
        "module(name='B', version='1.0')",
        "bazel_dep(name='A',version='3.0')");

    RepositoryName name = RepositoryName.create("@B.1.0");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    assertThat(result.hasError()).isFalse();
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMapping(
                ImmutableMap.of(
                    "B", RepositoryName.create("@B.1.0"), "A", RepositoryName.create("")),
                name));
  }

  @Test
  public void testRepoNameMapping_multipleVersionOverride_fork() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0',repo_name='B1')",
        "bazel_dep(name='B',version='2.0',repo_name='B2')",
        "multiple_version_override(module_name='B',versions=['1.0','2.0'])");
    registry
        .addModule(createModuleKey("B", "1.0"), "module(name='B', version='1.0')")
        .addModule(createModuleKey("B", "2.0"), "module(name='B', version='2.0')");

    RepositoryName name = RepositoryName.MAIN;
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    if (result.hasError()) {
      fail(result.getError().toString());
    }
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingForRootModule(
                ImmutableMap.of(
                    "", RepositoryName.create(""),
                    "A", RepositoryName.create(""),
                    "B1", RepositoryName.create("@B.1.0"),
                    "B2", RepositoryName.create("@B.2.0")),
                name));
  }

  @Test
  public void testRepoNameMapping_multipleVersionOverride_diamond() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0')",
        "multiple_version_override(module_name='D',versions=['1.0','2.0'])");
    registry
        .addModule(
            createModuleKey("B", "1.0"),
            "module(name='B', version='1.0');bazel_dep(name='D', version='1.0')")
        .addModule(
            createModuleKey("C", "2.0"),
            "module(name='C', version='2.0');bazel_dep(name='D', version='2.0')")
        .addModule(createModuleKey("D", "1.0"), "module(name='D', version='1.0')")
        .addModule(createModuleKey("D", "2.0"), "module(name='D', version='2.0')");

    RepositoryName name = RepositoryName.create("@B.1.0");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    if (result.hasError()) {
      fail(result.getError().toString());
    }
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMapping(
                ImmutableMap.of(
                    "B", RepositoryName.create("@B.1.0"),
                    "D", RepositoryName.create("@D.1.0")),
                name));
  }

  @Test
  public void testRepoNameMapping_multipleVersionOverride_lookup() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0',repo_name='B1')",
        "bazel_dep(name='B',version='2.0',repo_name='B2')",
        "multiple_version_override(module_name='B',versions=['1.0','2.0'])");
    registry
        .addModule(
            createModuleKey("B", "1.0"),
            "module(name='B', version='1.0');"
                + "bazel_dep(name='C', version='1.0', repo_name='com_foo_bar_c')")
        .addModule(createModuleKey("B", "2.0"), "module(name='B', version='2.0')")
        .addModule(createModuleKey("C", "1.0"), "module(name='C', version='1.0')");

    RepositoryName name = RepositoryName.create("@B.1.0");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    if (result.hasError()) {
      fail(result.getError().toString());
    }
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMapping(
                ImmutableMap.of(
                    "B", RepositoryName.create("@B.1.0"),
                    "com_foo_bar_c", RepositoryName.create("@C.1.0")),
                name));
  }

  @Test
  public void testMultipleRepositoriesWithMapping() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    repo_mapping = {'@a' : '@b'},",
        ")",
        "local_repository(",
        "    name = 'other_remote_repo',",
        "    path = '/other_remote_repo',",
        "    repo_mapping = {'@x' : '@y'},",
        ")");
    RepositoryName name1 = RepositoryName.create("a_remote_repo");
    SkyKey skyKey1 = RepositoryMappingValue.key(name1);
    RepositoryName name2 = RepositoryName.create("other_remote_repo");
    SkyKey skyKey2 = RepositoryMappingValue.key(name2);

    assertThatEvaluationResult(eval(skyKey1))
        .hasEntryThat(skyKey1)
        .isEqualTo(
            withMappingAllowingFallback(
                ImmutableMap.of(
                    "a", RepositoryName.create("b"),
                    "good", RepositoryName.create(""))));
    assertThatEvaluationResult(eval(skyKey2))
        .hasEntryThat(skyKey2)
        .isEqualTo(
            withMappingAllowingFallback(
                ImmutableMap.of(
                    "x", RepositoryName.create("y"),
                    "good", RepositoryName.create(""))));
  }

  @Test
  public void testRepositoryWithMultipleMappings() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    repo_mapping = {'@a' : '@b', '@x' : '@y'},",
        ")");
    RepositoryName name = RepositoryName.create("a_remote_repo");
    SkyKey skyKey = RepositoryMappingValue.key(name);

    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingAllowingFallback(
                ImmutableMap.of(
                    "a", RepositoryName.create("b"),
                    "x", RepositoryName.create("y"),
                    "good", RepositoryName.create(""))));
  }

  @Test
  public void testMixtureOfBothSystems_workspaceRepo() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'root')",
        "local_repository(",
        "    name = 'ws_repo',",
        "    path = '/ws_repo',",
        "    repo_mapping = {",
        "        '@B_alias' : '@B',",
        "        '@B_alias2' : '@B',",
        "        '@D_alias' : '@D',",
        "        '@E_alias' : '@E',",
        "    },",
        ")");
    scratch.overwriteFile(
        "MODULE.bazel",
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0')",
        "multiple_version_override(module_name='D',versions=['1.0','2.0'])");
    registry
        .addModule(
            createModuleKey("B", "1.0"),
            "module(name='B', version='1.0');bazel_dep(name='D', version='1.0')")
        .addModule(
            createModuleKey("C", "2.0"),
            "module(name='C', version='2.0');bazel_dep(name='D', version='2.0')")
        .addModule(createModuleKey("D", "1.0"), "module(name='D', version='1.0')")
        .addModule(createModuleKey("D", "2.0"), "module(name='D', version='2.0')");

    RepositoryName name = RepositoryName.create("ws_repo");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingAllowingFallback(
                ImmutableMap.<String, RepositoryName>builder()
                    .put("root", RepositoryName.MAIN)
                    // mappings to @B get remapped to @B.1.0 because of module B@1.0
                    .put("B_alias", RepositoryName.create("@B.1.0"))
                    .put("B_alias2", RepositoryName.create("@B.1.0"))
                    // mapping from @B to @B.1.0 is also created
                    .put("B", RepositoryName.create("@B.1.0"))
                    // mapping from @C to @C.2.0 is created despite not being mentioned
                    .put("C", RepositoryName.create("@C.2.0"))
                    // mapping to @D is untouched because D has a multiple-version override
                    .put("D_alias", RepositoryName.create("D"))
                    // mapping to @E is untouched because E is not a module
                    .put("E_alias", RepositoryName.create("E"))
                    .buildOrThrow()));
  }

  @Test
  public void testMixtureOfBothSystems_mainRepo() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'root')",
        "local_repository(",
        "    name = 'ws_repo',",
        "    path = '/ws_repo',",
        ")");
    scratch.overwriteFile(
        "MODULE.bazel", "module(name='A',version='0.1')", "bazel_dep(name='B',version='1.0')");
    registry
        .addModule(
            createModuleKey("B", "1.0"),
            "module(name='B', version='1.0');bazel_dep(name='C', version='2.0')")
        .addModule(createModuleKey("C", "2.0"), "module(name='C', version='2.0')");

    SkyKey skyKey = RepositoryMappingValue.key(RepositoryName.MAIN);
    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingForRootModule(
                ImmutableMap.of(
                    "", RepositoryName.MAIN,
                    "A", RepositoryName.MAIN,
                    "B", RepositoryName.create("@B.1.0"),
                    "root", RepositoryName.create("root"),
                    "ws_repo", RepositoryName.create("ws_repo")),
                RepositoryName.MAIN));
  }

  @Test
  public void testMixtureOfBothSystems_mainRepo_shouldNotSeeWorkspaceRepos() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'root')",
        "local_repository(",
        "    name = 'ws_repo',",
        "    path = '/ws_repo',",
        ")");
    scratch.overwriteFile(
        "MODULE.bazel", "module(name='A',version='0.1')", "bazel_dep(name='B',version='1.0')");
    registry
        .addModule(
            createModuleKey("B", "1.0"),
            "module(name='B', version='1.0');bazel_dep(name='C', version='2.0')")
        .addModule(createModuleKey("C", "2.0"), "module(name='C', version='2.0')");

    SkyKey skyKey = RepositoryMappingValue.KEY_FOR_ROOT_MODULE_WITHOUT_WORKSPACE_REPOS;
    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMapping(
                ImmutableMap.of(
                    "", RepositoryName.MAIN,
                    "A", RepositoryName.MAIN,
                    "B", RepositoryName.create("@B.1.0")),
                RepositoryName.MAIN));
  }

  @Test
  public void testErrorWithMapping() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.overwriteFile(
        "WORKSPACE",
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    repo_mapping = {'x' : '@b'},",
        ")");
    RepositoryName name = RepositoryName.create("a_remote_repo");
    SkyKey skyKey = RepositoryMappingValue.key(name);

    assertThatEvaluationResult(eval(skyKey))
        .hasErrorEntryForKeyThat(skyKey)
        .hasExceptionThat()
        .isInstanceOf(NoSuchPackageException.class);
    assertContainsEvent(
        "invalid repository name 'x': repo names used in the repo_mapping attribute must start with"
            + " '@'");
  }

  @Test
  public void testDefaultMainRepoNameInMapping() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    repo_mapping = {},",
        ")");
    RepositoryName name = RepositoryName.create("a_remote_repo");
    SkyKey skyKey = RepositoryMappingValue.key(name);

    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMappingAllowingFallback(
                ImmutableMap.of(TestConstants.WORKSPACE_NAME, RepositoryName.MAIN)));
  }

  @Test
  public void testExplicitMainRepoNameInMapping() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        ")");
    RepositoryName name = RepositoryName.create("a_remote_repo");
    SkyKey skyKey = RepositoryMappingValue.key(name);

    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(withMappingAllowingFallback(ImmutableMap.of("good", RepositoryName.MAIN)));
  }

  @Test
  public void builtinsRepo() throws Exception {
    scratch.overwriteFile(
        "MODULE.bazel",
        // Add an explicit dependency on @bazel_tools since we're not using built-in modules in this
        // test
        "bazel_dep(name='bazel_tools',version='1.0')");
    registry
        .addModule(
            createModuleKey("bazel_tools", "1.0"),
            "module(name='bazel_tools',version='1.0')",
            "bazel_dep(name='foo', version='1.0')")
        .addModule(createModuleKey("foo", "1.0"), "module(name='foo', version='1.0')");

    RepositoryName name = RepositoryName.create("_builtins");
    SkyKey skyKey = RepositoryMappingValue.key(name);
    EvaluationResult<RepositoryMappingValue> result = eval(skyKey);

    assertThat(result.hasError()).isFalse();
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            withMapping(
                ImmutableMap.of(
                    "bazel_tools", RepositoryName.BAZEL_TOOLS, // bazel_tools is a well-known module
                    "foo", RepositoryName.create("@foo.1.0"),
                    "_builtins", RepositoryName.create("_builtins")),
                name));
  }

  @Test
  public void testEqualsAndHashCode() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            withMappingAllowingFallback(ImmutableMap.of("foo", RepositoryName.create("bar"))),
            withMappingAllowingFallback(ImmutableMap.of("foo", RepositoryName.create("bar"))))
        .addEqualityGroup(
            withMappingAllowingFallback(ImmutableMap.of("fizz", RepositoryName.create("buzz"))),
            withMappingAllowingFallback(ImmutableMap.of("fizz", RepositoryName.create("buzz"))))
        .testEquals();
  }
}
