package com.google.devtools.build.lib.blackbox.tests.refreshRoots;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
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
    return shouldXXXBeUpdated(result, true);
  }

  private boolean shouldNotBeUpdated(ProcessResult result) {
    return shouldXXXBeUpdated(result, false);
  }

  private boolean shouldXXXBeUpdated(ProcessResult result, boolean value) {
    String[] lines = result.errString().split("\n");
    String pattern = value ? "Should be updated" : "Should not be updated";
    return Arrays.stream(lines).
        filter(s -> s.startsWith("DEBUG:")).
        anyMatch(s -> s.endsWith(pattern));
  }

  @Override
  @After
  public void tearDown() throws Exception {
    super.tearDown();
  }
}
