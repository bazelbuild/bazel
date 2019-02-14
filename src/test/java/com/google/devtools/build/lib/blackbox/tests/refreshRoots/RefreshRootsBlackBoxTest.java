package com.google.devtools.build.lib.blackbox.tests.refreshRoots;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.After;
import org.junit.Test;

public class RefreshRootsBlackBoxTest extends AbstractBlackBoxTest {
  private final static List<String> FILES = Lists.newArrayList(
      "BUILD.test",
      "WORKSPACE.test",
      ".bazelignore",
      "package.json",
      "test_rule.bzl",
      "use_node_modules.bzl"
  );

  @Test
  public void testBuildProject() throws Exception {
    generateProject();
    context().bazel().build("//...");

    checkProjectFiles();
  }

  @Test
  public void testBuildProjectFetchNotRecalled() throws Exception {
    generateProject();
    ProcessResult firstBuildResult = context().bazel().build("//...");
    assertThat(shouldBeUpdated(firstBuildResult)).isTrue();

    checkProjectFiles();

    ProcessResult secondBuildResult = context().bazel().build("//...");
    assertThat(shouldNotBeUpdated(secondBuildResult)).isTrue();
  }
  @Test
  public void testAddRefreshLater() throws Exception {
    generateProject();
    // now we remove the refresh roots
    List<String> properWorkspaceText = context().read("WORKSPACE");

    context().write("WORKSPACE", "workspace(name = \"fine_grained_user_modules\")",
        "load(\":use_node_modules.bzl\", \"generate_fine_grained_node_modules\")",
        "generate_fine_grained_node_modules(name = \"generated_node_modules\",",
        "package_json = \"//:package.json\",)");

    // for the first time, this still works
    ProcessResult firstBuildResult = context().bazel().build("//...");
    assertThat(buildSucceeded(firstBuildResult)).isTrue();
    assertThat(shouldBeUpdated(firstBuildResult)).isTrue();
    checkProjectFiles();

    // now we delete node_modules and the build fails
    Path nodeModules = context().getWorkDir().resolve("node_modules");
    PathUtils.deleteTree(nodeModules);

    ProcessResult secondBuildResult = context().bazel().withErrorCode(1).build("//...");
    // an attempt to update still works
    assertThat(shouldBeUpdated(secondBuildResult)).isTrue();
    // but not the actual files state for Bazel
    assertThat(buildFailed(secondBuildResult)).isTrue();

    // now change the WORKSPACE file back (and delete node_modules again)
    PathUtils.deleteTree(nodeModules);
    context().write("WORKSPACE", properWorkspaceText.toArray(new String[0]));

    ProcessResult thirdBuildResult = context().bazel().build("//...");
    assertThat(buildSucceeded(thirdBuildResult)).isTrue();
    assertThat(shouldBeUpdated(thirdBuildResult)).isTrue();
    checkProjectFiles();
  }

  @Test
  public void testBuildDeleteNodeModulesBuild() throws Exception {
    generateProject();
    ProcessResult firstBuildResult = context().bazel().build("//...");
    assertThat(shouldBeUpdated(firstBuildResult)).isTrue();
    checkProjectFiles();

    Path nodeModules = context().getWorkDir().resolve("node_modules");
    PathUtils.deleteTree(nodeModules);

    assertThat(nodeModules.toFile().exists()).isFalse();
    ProcessResult secondBuildResult = context().bazel().build("//...");
    assertThat(shouldBeUpdated(secondBuildResult)).isTrue();
    checkProjectFiles();

    ProcessResult thirdBuildResult = context().bazel().build("//...");
    assertThat(shouldNotBeUpdated(thirdBuildResult)).isTrue();
  }

  @Test
  public void testBuildDeleteExampleModuleBuild() throws Exception {
    generateProject();
    ProcessResult firstBuildResult = context().bazel().build("//...");
    assertThat(shouldBeUpdated(firstBuildResult)).isTrue();
    checkProjectFiles();

    Path exampleModule = context().getWorkDir().resolve("node_modules/example-module");
    PathUtils.deleteTree(exampleModule);

    assertThat(exampleModule.toFile().exists()).isFalse();
    ProcessResult secondBuildResult = context().bazel().build("//...");
    assertThat(shouldBeUpdated(secondBuildResult)).isTrue();
    checkProjectFiles();

    ProcessResult thirdBuildResult = context().bazel().build("//...");
    assertThat(shouldNotBeUpdated(thirdBuildResult)).isTrue();
  }

  private void generateProject() throws IOException {
    for (String fileName : FILES) {
      String text = ResourceFileLoader
          .loadResource(RefreshRootsBlackBoxTest.class, fileName);
      assertThat(text).isNotNull();
      assertThat(text).isNotEmpty();
      fileName = fileName.endsWith(".test") ?
          fileName.substring(0, fileName.length() - 5) :
          fileName;
      context().write(fileName, text);
    }
  }

  private void checkProjectFiles() throws IOException {
    Path nodeModules = context().getWorkDir().resolve("node_modules");
    assertThat(nodeModules.toFile().exists()).isTrue();
    assertThat(nodeModules.toFile().isDirectory()).isTrue();

    Path exampleModule = nodeModules.resolve("example-module");
    assertThat(exampleModule.toFile().exists()).isTrue();
    assertThat(exampleModule.toFile().isDirectory()).isTrue();

    Path packageJson = exampleModule.resolve("package.json");
    assertThat(packageJson.toFile().exists()).isTrue();
    assertThat(packageJson.toFile().isDirectory()).isFalse();

    List<String> text = PathUtils.readFile(packageJson);
    assertThat(text.stream().anyMatch(s -> s.trim().equals("\"name\": \"example-module\",")))
        .isTrue();
    assertThat(text.stream().anyMatch(s -> s.trim().equals("\"version\": \"0.2.0\""))).isTrue();
  }

  private boolean shouldBeUpdated(ProcessResult result) {
    return findPattern(result, "Should be updated");
  }

  private boolean shouldNotBeUpdated(ProcessResult result) {
    return findPattern(result, "Should not be updated");
  }

  private boolean buildSucceeded(ProcessResult result) {
    return findPattern(result, "INFO: Build completed successfully");
  }

  private boolean buildFailed(ProcessResult result) {
    return findPattern(result, "FAILED: Build did NOT complete successfully");
  }

  private boolean findPattern(ProcessResult result, String pattern) {
    String[] lines = result.errString().split("\n");
    return Arrays.stream(lines).anyMatch(s -> s.contains(pattern));
  }

  @Override
  @After
  public void tearDown() throws Exception {
    super.tearDown();
  }
}
