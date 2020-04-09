// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.rules.repository.ManagedDirectoriesKnowledgeImpl;
import com.google.devtools.build.lib.rules.repository.ManagedDirectoriesKnowledgeImpl.ManagedDirectoriesListener;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import org.hamcrest.BaseMatcher;
import org.hamcrest.Description;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.hamcrest.MockitoHamcrest;

/**
 * Test for {@link WorkspaceFileFunction}.
 */
@RunWith(JUnit4.class)
public class WorkspaceFileFunctionTest extends BuildViewTestCase {

  private WorkspaceFileFunction workspaceSkyFunc;
  private ExternalPackageFunction externalSkyFunc;
  private WorkspaceASTFunction astSkyFunc;
  private FakeFileValue fakeWorkspaceFileValue;
  private TestManagedDirectoriesKnowledge testManagedDirectoriesKnowledge;

  static class FakeFileValue extends FileValue {
    private boolean exists;
    private long size;

    FakeFileValue() {
      super();
      exists = true;
      size = 0L;
    }

    @Override
    public RootedPath realRootedPath() {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileStateValue realFileStateValue() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean exists() {
      return exists;
    }

    @Override
    public boolean isFile() {
      return exists;
    }

    @Override
    public ImmutableList<RootedPath> logicalChainDuringResolution() {
      throw new UnsupportedOperationException();
    }

    void setExists(boolean exists) {
      this.exists = exists;
    }

    @Override
    public long getSize() {
      return size;
    }

    void setSize(long size) {
      this.size = size;
    }
  }

  @Before
  public final void setUp() throws Exception {
    ConfiguredRuleClassProvider ruleClassProvider =
        TestRuleClassProvider.getRuleClassProvider(true);
    workspaceSkyFunc =
        new WorkspaceFileFunction(
            ruleClassProvider,
            pkgFactory,
            directories,
            /*skylarkImportLookupFunctionForInlining=*/ null);
    externalSkyFunc = new ExternalPackageFunction();
    astSkyFunc = new WorkspaceASTFunction(ruleClassProvider);
    fakeWorkspaceFileValue = new FakeFileValue();
  }

  @Override
  protected ManagedDirectoriesKnowledge getManagedDirectoriesKnowledge() {
    testManagedDirectoriesKnowledge = new TestManagedDirectoriesKnowledge();
    return testManagedDirectoriesKnowledge;
  }

  @Override
  protected Iterable<EnvironmentExtension> getEnvironmentExtensions() {
    return ImmutableList.of();
  }

  private static Label getLabelMapping(Package pkg, String name) throws NoSuchTargetException {
    return (Label) ((Rule) pkg.getTarget(name)).getAttributeContainer().getAttr("actual");
  }

  private RootedPath createWorkspaceFile(String... contents) throws IOException {
    Path workspacePath = scratch.overwriteFile("WORKSPACE", contents);
    fakeWorkspaceFileValue.setSize(workspacePath.getFileSize());
    return RootedPath.toRootedPath(
        Root.fromPath(workspacePath.getParentDirectory()),
        PathFragment.create(workspacePath.getBaseName()));
  }

  // Dummy hamcrest matcher that match the function name of a skykey
  static class SkyKeyMatchers extends BaseMatcher<SkyKey> {
    private final SkyFunctionName functionName;

    public SkyKeyMatchers(SkyFunctionName functionName) {
      this.functionName = functionName;
    }
    @Override
    public boolean matches(Object item) {
      if (item instanceof SkyKey) {
        return ((SkyKey) item).functionName().equals(functionName);
      }
      return false;
    }

    @Override
    public void describeTo(Description description) {}
  }

  private SkyFunction.Environment getEnv() throws InterruptedException {
    PathPackageLocator locator = Mockito.mock(PathPackageLocator.class);
    Mockito.when(locator.getPathEntries())
        .thenReturn(ImmutableList.of(Root.fromPath(directories.getWorkspace())));

    SkyFunction.Environment env = Mockito.mock(SkyFunction.Environment.class);
    Mockito.when(env.getValue(MockitoHamcrest.argThat(new SkyKeyMatchers(FileValue.FILE))))
        .then(
            invocation -> {
              SkyKey key = (SkyKey) invocation.getArguments()[0];
              String path = ((RootedPath) key.argument()).getRootRelativePath().getPathString();
              FakeFileValue result = new FakeFileValue();
              result.setExists(path.equals("WORKSPACE"));
              return result;
            });
    Mockito.when(
            env.getValue(
                MockitoHamcrest.argThat(new SkyKeyMatchers(WorkspaceFileValue.WORKSPACE_FILE))))
        .then(
            invocation -> {
              SkyKey key = (SkyKey) invocation.getArguments()[0];
              return workspaceSkyFunc.compute(key, getEnv());
            });
    Mockito.when(
            env.getValue(MockitoHamcrest.argThat(new SkyKeyMatchers(SkyFunctions.WORKSPACE_AST))))
        .then(
            invocation -> {
              SkyKey key = (SkyKey) invocation.getArguments()[0];
              return astSkyFunc.compute(key, getEnv());
            });
    Mockito.when(
            env.getValue(MockitoHamcrest.argThat(new SkyKeyMatchers(SkyFunctions.PRECOMPUTED))))
        .then(
            invocation -> {
              SkyKey key = (SkyKey) invocation.getArguments()[0];
              if (key.equals(PrecomputedValue.STARLARK_SEMANTICS.getKeyForTesting())) {
                return new PrecomputedValue(StarlarkSemantics.DEFAULT_SEMANTICS);
              } else if (key.equals(
                  RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE
                      .getKeyForTesting())) {
                return new PrecomputedValue(Optional.<RootedPath>absent());
              } else if (key.equals(PrecomputedValue.PATH_PACKAGE_LOCATOR.getKeyForTesting())) {
                return new PrecomputedValue(locator);
              } else {
                return null;
              }
            });
    return env;
  }

  private EvaluationResult<WorkspaceFileValue> eval(SkyKey key) throws InterruptedException {
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("WORKSPACE")).build(),
            Root.fromPath(rootDirectory));
    return SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
  }

  @Test
  public void testImportToChunkMapSimple() throws Exception {
    scratch.file("a.bzl", "a = 'a'");
    scratch.file("b.bzl", "b = 'b'");
    scratch.file("BUILD", "");
    RootedPath workspace =
        createWorkspaceFile(
            "WORKSPACE",
            "workspace(name = 'good')",
            "load('//:a.bzl', 'a')",
            "x = 1  #for chunk break",
            "load('//:b.bzl', 'b')");
    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    assertThat(value1.getImportToChunkMap()).containsEntry("//:a.bzl", 1);

    SkyKey key2 = WorkspaceFileValue.key(workspace, 2);
    EvaluationResult<WorkspaceFileValue> result2 = eval(key2);
    WorkspaceFileValue value2 = result2.get(key2);
    assertThat(value2.getImportToChunkMap()).containsEntry("//:a.bzl", 1);
    assertThat(value2.getImportToChunkMap()).containsEntry("//:b.bzl", 2);
  }

  @Test
  public void testImportToChunkMapDoesNotOverrideDuplicate() throws Exception {
    scratch.file("a.bzl", "a = 'a'");
    scratch.file("BUILD", "");
    RootedPath workspace =
        createWorkspaceFile(
            "WORKSPACE",
            "workspace(name = 'good')",
            "load('//:a.bzl', 'a')",
            "x = 1  #for chunk break",
            "load('//:a.bzl', 'a')");
    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    assertThat(value1.getImportToChunkMap()).containsEntry("//:a.bzl", 1);

    SkyKey key2 = WorkspaceFileValue.key(workspace, 2);
    EvaluationResult<WorkspaceFileValue> result2 = eval(key2);
    WorkspaceFileValue value2 = result2.get(key2);
    assertThat(value2.getImportToChunkMap()).containsEntry("//:a.bzl", 1);
    assertThat(value2.getImportToChunkMap()).doesNotContainEntry("//:a.bzl", 2);
  }

  @Test
  public void testRepositoryMappingInChunks() throws Exception {
    scratch.file("b.bzl", "b = 'b'");
    scratch.file("BUILD", "");
    RootedPath workspace =
        createWorkspaceFile(
            "workspace(name = 'good')",
            "local_repository(name = 'a', path = '../a', repo_mapping = {'@x' : '@y'})",
            "load('//:b.bzl', 'b')",
            "local_repository(name = 'b', path = '../b', repo_mapping = {'@x' : '@y'})");
    RepositoryName a = RepositoryName.create("@a");
    RepositoryName b = RepositoryName.create("@b");
    RepositoryName x = RepositoryName.create("@x");
    RepositoryName y = RepositoryName.create("@y");
    RepositoryName good = RepositoryName.create("@good");
    RepositoryName main = RepositoryName.create("@");

    SkyKey key0 = WorkspaceFileValue.key(workspace, 0);
    EvaluationResult<WorkspaceFileValue> result0 = eval(key0);
    WorkspaceFileValue value0 = result0.get(key0);
    assertThat(value0.getRepositoryMapping()).containsEntry(a, ImmutableMap.of(x, y, good, main));

    SkyKey key1 = WorkspaceFileValue.key(workspace, 1);
    EvaluationResult<WorkspaceFileValue> result1 = eval(key1);
    WorkspaceFileValue value1 = result1.get(key1);
    assertThat(value1.getRepositoryMapping()).containsEntry(a, ImmutableMap.of(x, y, good, main));
    assertThat(value1.getRepositoryMapping()).containsEntry(b, ImmutableMap.of(x, y, good, main));
  }

  @Test
  public void setTestManagedDirectoriesKnowledge() throws Exception {
    PrecomputedValue precomputedValue =
        (PrecomputedValue)
            getEnv().getValue(PrecomputedValue.STARLARK_SEMANTICS.getKeyForTesting());
    StarlarkSemantics semantics =
        (StarlarkSemantics) Preconditions.checkNotNull(precomputedValue).get();
    Injectable injectable = getSkyframeExecutor().injectable();
    try {
      StarlarkSemantics semanticsWithManagedDirectories =
          StarlarkSemantics.builderWithDefaults()
              .experimentalAllowIncrementalRepositoryUpdates(true)
              .build();
      PrecomputedValue.STARLARK_SEMANTICS.set(injectable, semanticsWithManagedDirectories);

      TestManagedDirectoriesListener listener = new TestManagedDirectoriesListener();
      ManagedDirectoriesKnowledgeImpl knowledge = new ManagedDirectoriesKnowledgeImpl(listener);

      RepositoryName one = RepositoryName.create("@repo1");
      RepositoryName two = RepositoryName.create("@repo2");
      RepositoryName three = RepositoryName.create("@repo3");

      PathFragment pf1 = PathFragment.create("dir1");
      PathFragment pf2 = PathFragment.create("dir2");
      PathFragment pf3 = PathFragment.create("dir3");

      assertThat(knowledge.getManagedDirectories(one)).isEmpty();
      assertThat(knowledge.getOwnerRepository(pf1)).isNull();

      WorkspaceFileValue workspaceFileValue = createWorkspaceFileValueForTest();
      boolean isChanged = knowledge.workspaceHeaderReloaded(null, workspaceFileValue);

      assertThat(isChanged).isTrue();
      assertThat(listener.getRepositoryNames()).containsExactly(one, two);

      assertThat(knowledge.getManagedDirectories(one)).containsExactly(pf1, pf2);
      assertThat(knowledge.getManagedDirectories(two)).containsExactly(pf3);
      assertThat(knowledge.getManagedDirectories(three)).isEmpty();

      assertThat(knowledge.getOwnerRepository(pf1)).isEqualTo(one);
      assertThat(knowledge.getOwnerRepository(pf2)).isEqualTo(one);
      assertThat(knowledge.getOwnerRepository(pf3)).isEqualTo(two);

      // Nothing changed, let's test the behavior.
      listener.reset();
      isChanged = knowledge.workspaceHeaderReloaded(workspaceFileValue, workspaceFileValue);
      assertThat(isChanged).isFalse();
      assertThat(listener.getRepositoryNames()).containsExactly(one, two);

      assertThat(knowledge.getManagedDirectories(one)).containsExactly(pf1, pf2);
      assertThat(knowledge.getManagedDirectories(two)).containsExactly(pf3);
      assertThat(knowledge.getManagedDirectories(three)).isEmpty();

      assertThat(knowledge.getOwnerRepository(pf1)).isEqualTo(one);
      assertThat(knowledge.getOwnerRepository(pf2)).isEqualTo(one);
      assertThat(knowledge.getOwnerRepository(pf3)).isEqualTo(two);
    } finally {
      PrecomputedValue.STARLARK_SEMANTICS.set(injectable, semantics);
    }
  }

  @Test
  public void testManagedDirectories() throws Exception {
    PrecomputedValue precomputedValue =
        (PrecomputedValue)
            getEnv().getValue(PrecomputedValue.STARLARK_SEMANTICS.getKeyForTesting());
    StarlarkSemantics semantics =
        (StarlarkSemantics) Preconditions.checkNotNull(precomputedValue).get();
    Injectable injectable = getSkyframeExecutor().injectable();
    try {
      StarlarkSemantics semanticsWithManagedDirectories =
          StarlarkSemantics.builderWithDefaults()
              .experimentalAllowIncrementalRepositoryUpdates(true)
              .build();
      PrecomputedValue.STARLARK_SEMANTICS.set(injectable, semanticsWithManagedDirectories);

      createWorkspaceFileValueForTest();

      assertManagedDirectoriesParsingError(
          "{'@repo1': 'dir1', '@repo2': ['dir3']}",
          "managed_directories attribute value should be of the type attr.string_list_dict(),"
              + " mapping repository name to the list of managed directories.");

      assertManagedDirectoriesParsingError(
          "{'@repo1': ['dir1'], '@repo2': ['dir1']}",
          "managed_directories attribute should not contain multiple (or duplicate) repository"
              + " mappings for the same directory ('dir1').");

      assertManagedDirectoriesParsingError(
          "{'@repo1': ['']}", "Expected managed directory path to be non-empty string.");
      assertManagedDirectoriesParsingError(
          "{'@repo1': ['/abc']}",
          "Expected managed directory path ('/abc') to be relative to the workspace root.");
      assertManagedDirectoriesParsingError(
          "{'@repo1': ['../abc']}",
          "Expected managed directory path ('../abc') to be under the workspace root.");
      assertManagedDirectoriesParsingError(
          "{'@repo1': ['a/b', 'a/b']}",
          "managed_directories attribute should not contain multiple (or duplicate)"
              + " repository mappings for the same directory ('a/b').");
      assertManagedDirectoriesParsingError(
          "{'@repo1': [], '@repo1': [] }", "Duplicated key \"@repo1\" when creating dictionary");
      assertManagedDirectoriesParsingError(
          "{'@repo1': ['a/b'], '@repo2': ['a/b/c/..'] }",
          "managed_directories attribute should not contain multiple (or duplicate)"
              + " repository mappings for the same directory ('a/b/c/..').");
      assertManagedDirectoriesParsingError(
          "{'@repo1': ['a'], '@repo2': ['a/b'] }",
          "managed_directories attribute value can not contain nested mappings."
              + " 'a/b' is a descendant of 'a'.");
      assertManagedDirectoriesParsingError(
          "{'@repo1': ['a/b'], '@repo2': ['a'] }",
          "managed_directories attribute value can not contain nested mappings."
              + " 'a/b' is a descendant of 'a'.");

      assertManagedDirectoriesParsingError(
          "{'repo1': []}",
          "Cannot parse repository name 'repo1'. Repository name should start with '@'.");
    } finally {
      PrecomputedValue.STARLARK_SEMANTICS.set(injectable, semantics);
    }
  }

  private WorkspaceFileValue createWorkspaceFileValueForTest() throws Exception {
    WorkspaceFileValue workspaceFileValue =
        parseWorkspaceFileValue(
            "workspace(",
            "  name = 'rr',",
            "  managed_directories = {'@repo1': ['dir1', 'dir2'], '@repo2': ['dir3/dir1/..']}",
            ")");
    ImmutableMap<PathFragment, RepositoryName> managedDirectories =
        workspaceFileValue.getManagedDirectories();
    assertThat(managedDirectories).isNotNull();
    assertThat(managedDirectories).hasSize(3);
    assertThat(managedDirectories)
        .containsExactly(
            PathFragment.create("dir1"), RepositoryName.create("@repo1"),
            PathFragment.create("dir2"), RepositoryName.create("@repo1"),
            PathFragment.create("dir3"), RepositoryName.create("@repo2"));
    return workspaceFileValue;
  }

  private void assertManagedDirectoriesParsingError(
      String managedDirectoriesValue, String expectedError) throws Exception {
    parseWorkspaceFileValueWithError(
        expectedError,
        "workspace(",
        "  name = 'rr',",
        "  managed_directories = " + managedDirectoriesValue,
        ")");
  }

  private WorkspaceFileValue parseWorkspaceFileValue(String... lines) throws Exception {
    WorkspaceFileValue workspaceFileValue = parseWorkspaceFileValueImpl(lines);
    Package pkg = workspaceFileValue.getPackage();
    if (pkg.containsErrors()) {
      throw new RuntimeException(
          Preconditions.checkNotNull(Iterables.getFirst(pkg.getEvents(), null)).getMessage());
    }
    return workspaceFileValue;
  }

  private void parseWorkspaceFileValueWithError(String expectedError, String... lines)
      throws Exception {
    WorkspaceFileValue workspaceFileValue = parseWorkspaceFileValueImpl(lines);
    Package pkg = workspaceFileValue.getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), expectedError);
  }

  private WorkspaceFileValue parseWorkspaceFileValueImpl(String[] lines)
      throws IOException, InterruptedException {
    RootedPath workspaceFile = createWorkspaceFile(lines);
    WorkspaceFileKey key = WorkspaceFileValue.key(workspaceFile);
    EvaluationResult<WorkspaceFileValue> result = eval(key);
    return result.get(key);
  }

  @Test
  public void testInvalidRepo() throws Exception {
    RootedPath workspacePath = createWorkspaceFile("workspace(name = 'foo$')");
    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), "foo$ is not a legal workspace name");
  }

  @Test
  public void testBindFunction() throws Exception {
    String[] lines = {"bind(name = 'foo/bar',", "actual = '//foo:bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key(workspacePath);
    PackageValue value = (PackageValue) externalSkyFunc.compute(key, getEnv());
    Package pkg = value.getPackage();
    assertThat(getLabelMapping(pkg, "foo/bar"))
        .isEqualTo(Label.parseAbsolute("//foo:bar", ImmutableMap.of()));
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testBindArgsReversed() throws Exception {
    String[] lines = {"bind(actual = '//foo:bar', name = 'foo/bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key(workspacePath);
    PackageValue value = (PackageValue) externalSkyFunc.compute(key, getEnv());
    Package pkg = value.getPackage();
    assertThat(getLabelMapping(pkg, "foo/bar"))
        .isEqualTo(Label.parseAbsolute("//foo:bar", ImmutableMap.of()));
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testNonExternalBinding() throws Exception {
    // name must be a valid label name.
    String[] lines = {"bind(name = 'foo:bar', actual = '//bar/baz')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), "target names may not contain ':'");
  }

  @Test
  public void testWorkspaceFileParsingError() throws Exception {
    // //external:bar:baz is not a legal package.
    String[] lines = {"bind(name = 'foo/bar', actual = '//external:bar:baz')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertThat(pkg.containsErrors()).isTrue();
    MoreAsserts.assertContainsEvent(pkg.getEvents(), "target names may not contain ':'");
  }

  @Test
  public void testNoWorkspaceFile() throws Exception {
    // Even though the WORKSPACE exists, Skyframe thinks it doesn't, so it doesn't.
    String[] lines = {"bind(name = 'foo/bar', actual = '//foo:bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);
    fakeWorkspaceFileValue.setExists(false);

    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertThat(pkg.containsErrors()).isFalse();
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testListBindFunction() throws Exception {
    String[] lines = {
      "L = ['foo', 'bar']", "bind(name = '%s/%s' % (L[0], L[1]),", "actual = '//foo:bar')"
    };
    RootedPath workspacePath = createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key(workspacePath);
    PackageValue value = (PackageValue) externalSkyFunc.compute(key, getEnv());
    Package pkg = value.getPackage();
    assertThat(getLabelMapping(pkg, "foo/bar"))
        .isEqualTo(Label.parseAbsolute("//foo:bar", ImmutableMap.of()));
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testWorkspaceFileValueListener() throws Exception {
    // Normally, syscalls cache is reset in the sync() method of the SkyframeExecutor, before
    // diffing.
    // But here we are calling only actual diffing part, exposed for testing:
    // handleDiffsForTesting(), so we better turn off the syscalls cache.
    skyframeExecutor.turnOffSyscallCacheForTesting();

    createWorkspaceFile("workspace(name = 'old')");
    skyframeExecutor.handleDiffsForTesting(NullEventHandler.INSTANCE);
    assertThat(testManagedDirectoriesKnowledge.getLastWorkspaceName()).isEqualTo("old");
    assertThat(testManagedDirectoriesKnowledge.getCnt()).isEqualTo(1);

    createWorkspaceFile("workspace(name = 'changed')");
    skyframeExecutor.handleDiffsForTesting(NullEventHandler.INSTANCE);
    assertThat(testManagedDirectoriesKnowledge.getLastWorkspaceName()).isEqualTo("changed");
    assertThat(testManagedDirectoriesKnowledge.getCnt()).isEqualTo(2);
  }

  @Test
  public void testDoNotSymlinkInExecroot() throws Exception {
    PrecomputedValue precomputedValue =
        (PrecomputedValue)
            getEnv().getValue(PrecomputedValue.STARLARK_SEMANTICS.getKeyForTesting());
    StarlarkSemantics semantics =
        (StarlarkSemantics) Preconditions.checkNotNull(precomputedValue).get();
    Injectable injectable = getSkyframeExecutor().injectable();

    try {
      StarlarkSemantics semanticsWithNinjaActions =
          StarlarkSemantics.builderWithDefaults().experimentalNinjaActions(true).build();
      PrecomputedValue.STARLARK_SEMANTICS.set(injectable, semanticsWithNinjaActions);

      assertThat(
              parseWorkspaceFileValue("toplevel_output_directories(paths = [\"out\"])")
                  .getDoNotSymlinkInExecrootPaths())
          .containsExactly("out");
      assertThat(
              parseWorkspaceFileValue(
                      "toplevel_output_directories(paths = [\"out\", \"one more with"
                          + " space\"])")
                  .getDoNotSymlinkInExecrootPaths())
          .containsExactly("out", "one more with space");
      // Empty sequence is allowed.
      assertThat(
              parseWorkspaceFileValue("toplevel_output_directories(paths = [])")
                  .getDoNotSymlinkInExecrootPaths())
          .isEmpty();

      parseWorkspaceFileValueWithError(
          "toplevel_output_directories should not "
              + "contain duplicate values: \"out\" is specified more then once.",
          "toplevel_output_directories(paths = [\"out\", \"out\"])");
      parseWorkspaceFileValueWithError(
          "toplevel_output_directories can only accept "
              + "top level directories under workspace, \"out/subdir\" "
              + "can not be specified as an attribute.",
          "toplevel_output_directories(paths = [\"out/subdir\"])");
      parseWorkspaceFileValueWithError(
          "Empty path can not be passed to " + "toplevel_output_directories.",
          "toplevel_output_directories(paths = [\"\"])");
      parseWorkspaceFileValueWithError(
          "toplevel_output_directories can only "
              + "accept top level directories under workspace, \"/usr/local/bin\" "
              + "can not be specified as an attribute.",
          "toplevel_output_directories(paths = [\"/usr/local/bin\"])");
    } finally {
      PrecomputedValue.STARLARK_SEMANTICS.set(injectable, semantics);
    }
  }

  private static class TestManagedDirectoriesKnowledge implements ManagedDirectoriesKnowledge {
    private String lastWorkspaceName;
    private int cnt = 0;

    @Nullable
    @Override
    public RepositoryName getOwnerRepository(PathFragment relativePathFragment) {
      return null;
    }

    @Override
    public ImmutableSet<PathFragment> getManagedDirectories(RepositoryName repositoryName) {
      return null;
    }

    @Override
    public boolean workspaceHeaderReloaded(
        @Nullable WorkspaceFileValue oldValue, @Nullable WorkspaceFileValue newValue) {
      if (Objects.equals(oldValue, newValue)) {
        return false;
      }
      ++cnt;
      lastWorkspaceName = newValue != null ? newValue.getPackage().getWorkspaceName() : null;
      return true;
    }

    private String getLastWorkspaceName() {
      return lastWorkspaceName;
    }

    private int getCnt() {
      return cnt;
    }
  }

  private static class TestManagedDirectoriesListener implements ManagedDirectoriesListener {
    @Nullable private Set<RepositoryName> repositoryNames;

    @Override
    public void onManagedDirectoriesRefreshed(Set<RepositoryName> repositoryNames) {
      this.repositoryNames = repositoryNames;
    }

    @Nullable
    public Set<RepositoryName> getRepositoryNames() {
      return repositoryNames;
    }

    public void reset() {
      repositoryNames = null;
    }
  }
}
