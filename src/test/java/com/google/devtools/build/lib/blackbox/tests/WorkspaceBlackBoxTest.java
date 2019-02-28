package com.google.devtools.build.lib.blackbox.tests;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.Test;

/**
 * End to end test of workspace-related functionality.
 */
public class WorkspaceBlackBoxTest extends AbstractBlackBoxTest {
  @Test
  public void testWorkspaceChanges() throws Exception {
    Path repoA = HelperStarlarkTexts
        .setupRepositoryWithRuleWritingTextToFile(context().getTmpDir(), "a", "hi").toAbsolutePath();
    Path repoB = HelperStarlarkTexts
        .setupRepositoryWithRuleWritingTextToFile(context().getTmpDir(), "b", "bye").toAbsolutePath();

    context().write("WORKSPACE",
        String.format("local_repository(name = \"x\", path = \"%s\",)", pathToString(repoA)));
    context().bazel().build("@x//:x");

    Path xPath = context().resolveBinPath(context().bazel(), "external/x/out");
    assertThat(Files.exists(xPath)).isTrue();
    List<String> lines = PathUtils.readFile(xPath);
    assertThat(lines.size()).isEqualTo(1);
    assertThat(lines.get(0)).isEqualTo("hi");

    context().write("WORKSPACE",
        String.format("local_repository(name = \"x\", path = \"%s\",)", pathToString(repoB)));
    context().bazel().build("@x//:x");

    assertThat(Files.exists(xPath)).isTrue();
    lines = PathUtils.readFile(xPath);
    assertThat(lines.size()).isEqualTo(1);
    assertThat(lines.get(0)).isEqualTo("bye");
  }

  @Test
  public void testPathWithSpace() throws Exception {
    context().write("a b/WORKSPACE");
    context().bazel().info();
    context().bazel().help();
  }

  // TODO(ichern) move other tests from workspace_test.sh here.

}
