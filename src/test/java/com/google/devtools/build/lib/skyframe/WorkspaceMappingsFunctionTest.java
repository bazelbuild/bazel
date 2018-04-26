package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

@RunWith(JUnit4.class)
public class WorkspaceMappingsFunctionTest extends BuildViewTestCase {

  private EvaluationResult<WorkspaceMappingsValue> eval(SkyKey key) throws InterruptedException {
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            ModifiedFileSet.builder().modify(PathFragment.create("WORKSPACE")).build(),
            Root.fromPath(rootDirectory));
    return SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
  }

  @Test
  public void testSimpleMappings() throws Exception {
    scratch.overwriteFile("WORKSPACE",
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    assignments = {'@a' : '@b'},",
        ")");

    RepositoryName name = RepositoryName.create("@a_remote_repo");
    SkyKey skyKey = WorkspaceMappingsValue.key(name);
    EvaluationResult<WorkspaceMappingsValue> result = eval(skyKey);
    assertThat(result.hasError()).isFalse();
    assertThatEvaluationResult(result)
        .hasEntryThat(skyKey)
        .isEqualTo(
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@a"), RepositoryName.create("@b"))));
  }

  @Test
  public void testMultipleRepositoriesWithMappings() throws Exception {
    scratch.overwriteFile("WORKSPACE",
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    assignments = {'@a' : '@b'},",
        ")",
        "local_repository(",
        "    name = 'other_remote_repo',",
        "    path = '/other_remote_repo',",
        "    assignments = {'@x' : '@y'},",
        ")");

    RepositoryName name1 = RepositoryName.create("@a_remote_repo");
    SkyKey skyKey1 = WorkspaceMappingsValue.key(name1);
    assertThatEvaluationResult(eval(skyKey1))
        .hasEntryThat(skyKey1)
        .isEqualTo(
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@a"), RepositoryName.create("@b"))));

    RepositoryName name2 = RepositoryName.create("@other_remote_repo");
    SkyKey skyKey2 = WorkspaceMappingsValue.key(name2);
    assertThatEvaluationResult(eval(skyKey2))
        .hasEntryThat(skyKey2)
        .isEqualTo(
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@x"), RepositoryName.create("@y"))));
  }

  @Test
  public void testRepositoryWithMultipleMappings() throws Exception {
    scratch.overwriteFile("WORKSPACE",
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    assignments = {'@a' : '@b', '@x' : '@y'},",
        ")");

    RepositoryName name = RepositoryName.create("@a_remote_repo");
    SkyKey skyKey = WorkspaceMappingsValue.key(name);
    assertThatEvaluationResult(eval(skyKey))
        .hasEntryThat(skyKey)
        .isEqualTo(
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(
                    RepositoryName.create("@a"), RepositoryName.create("@b"),
                    RepositoryName.create("@x"), RepositoryName.create("@y"))));
  }

  @Test
  public void testErrorWithMappings() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.overwriteFile("WORKSPACE",
        "workspace(name = 'good')",
        "local_repository(",
        "    name = 'a_remote_repo',",
        "    path = '/a_remote_repo',",
        "    assignments = {'x' : '@b'},",
        ")");
    RepositoryName name = RepositoryName.create("@a_remote_repo");
    SkyKey skyKey = WorkspaceMappingsValue.key(name);
    assertThatEvaluationResult(eval(skyKey))
        .hasErrorEntryForKeyThat(skyKey)
        .hasExceptionThat()
        .isInstanceOf(NoSuchPackageException.class);
    assertContainsEvent("invalid repository name 'x': workspace names must start with '@'");
  }

  @Test
  public void testEqualsAndHashCode()throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@foo"), RepositoryName.create("@bar"))),
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@foo"), RepositoryName.create("@bar"))))
        .addEqualityGroup(
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@fizz"), RepositoryName.create("@buzz"))),
            WorkspaceMappingsValue.withMappings(
                ImmutableMap.of(RepositoryName.create("@fizz"), RepositoryName.create("@buzz"))));
  }

}
