// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.tests.refreshroots;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Tests for refresh roots.
 */
public class RefreshRootsBlackBoxTest extends AbstractBlackBoxTest {
  private final static List<String> FILES = Lists.newArrayList(
      "BUILD.test",
      "WORKSPACE.test",
      ".bazelignore",
      "package.json",
      "test_rule.bzl",
      "use_node_modules.bzl"
  );
  private Random random;
  private Integer currentDebugId;

  @Override
  @Before
  public void setUp() throws Exception {
    random = new Random(17);
    super.setUp();
  }

  @Test
  public void testBuildProject() throws Exception {
    generateProject();
    buildExpectUpdated();
    checkProjectFiles();
  }

  @Test
  public void testBuildProjectFetchNotRecalled() throws Exception {
    generateProject();
    buildExpectUpdated();
    checkProjectFiles();
    buildExpectNotUpdated();
    checkProjectFiles();
  }

  private BuilderRunner bazel() {
    currentDebugId = random.nextInt();
    return context().bazel().withEnv("DEBUG_ID", String.valueOf(currentDebugId));
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
    buildExpectUpdated();
    checkProjectFiles();

    // now we delete node_modules and the build fails
    Path nodeModules = context().getWorkDir().resolve("node_modules");
    PathUtils.deleteTree(nodeModules);

    ProcessResult secondBuildResult = bazel().withErrorCode(1).build("//...");
    // without the refresh roots, some of the inputs are missing and the build failed
    buildFailed(secondBuildResult);
    assertThat(nodeModules.toFile().exists()).isFalse();

    // now change the WORKSPACE file back
    context().write("WORKSPACE", properWorkspaceText.toArray(new String[0]));

    buildExpectUpdated();
    checkProjectFiles();
  }

  @Test
  public void testBuildDeleteNodeModulesBuild() throws Exception {
    generateProject();
    buildExpectUpdated();
    checkProjectFiles();

    Path nodeModules = context().getWorkDir().resolve("node_modules");
    PathUtils.deleteTree(nodeModules);

    assertThat(nodeModules.toFile().exists()).isFalse();
    buildExpectUpdated();
    checkProjectFiles();

    buildExpectNotUpdated();
    checkProjectFiles();
  }

  @Test
  public void testChangeOfFileTextUnderNodeModules() throws Exception {
    generateProject();
    buildExpectUpdated();
    checkProjectFiles();

    Path nodeModules = context().getWorkDir().resolve("node_modules");
    Path modulePackageJson = nodeModules.resolve("example-module/package.json");
    assertThat(modulePackageJson.toFile().exists()).isTrue();

    // assert that non-structural changes are not detected
    PathUtils.append(modulePackageJson, "# comment");

    buildExpectNotUpdated();
    checkProjectFiles();
  }

  @Test
  public void testBuildDeleteExampleModuleBuild() throws Exception {
    generateProject();
    buildExpectUpdated();
    checkProjectFiles();

    Path exampleModule = context().getWorkDir().resolve("node_modules/example-module");
    PathUtils.deleteTree(exampleModule);

    assertThat(exampleModule.toFile().exists()).isFalse();
    buildExpectUpdated();
    checkProjectFiles();

    buildExpectNotUpdated();
    checkProjectFiles();
  }

  @Test
  public void testLoadIsNotCalledForRefreshRoots() throws Exception {
    generateProject();
    Path workspaceFile = context().getWorkDir().resolve(WORKSPACE);
    PathUtils.append(workspaceFile, "load('@non_existing//:target.bzl', 'some_symbol')");

    // test that there is error when loading, but no cycles detected
    ProcessResult result = bazel().shouldFail().build("//...");
    assertThat(findPattern(result, "ERROR: Failed to load Starlark extension")).isTrue();
  }

  @Test
  public void testWithBazelTools() throws Exception {
    generateProject();
    Path workspaceFile = context().getWorkDir().resolve(WORKSPACE);
    PathUtils.append(workspaceFile,
        "load(\"@bazel_tools//tools/build_defs/repo:http.bzl\", \"http_archive\", \"http_file\")");
    buildExpectUpdated();
    checkProjectFiles();
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

  private String getDebugId(BuilderRunner bazel) throws Exception {
    Path path = context().resolveExecRootPath(bazel, "external/generated_node_modules/debug_id");
    List<String> lines = PathUtils.readFile(path);
    assertThat(lines.size()).isEqualTo(1);
    return lines.get(0);
  }

  private ProcessResult buildExpectUpdated() throws Exception {
    return buildExpectUpdated(false);
  }

  private ProcessResult buildExpectUpdated(boolean debug) throws Exception {
    BuilderRunner bazel = bazel();
    if (debug) {
      bazel.enableDebug();
    }
    ProcessResult result = bazel.build("//...");
    buildSucceeded(result);
    shouldBeUpdated(bazel);
    return result;
  }

  private ProcessResult buildExpectNotUpdated() throws Exception {
    return buildExpectNotUpdated(false);
  }

  private ProcessResult buildExpectNotUpdated(boolean debug) throws Exception {
    BuilderRunner bazel = bazel();
    if (debug) {
      bazel.enableDebug();
    }
    ProcessResult result = bazel.build("//...");
    buildSucceeded(result);
    shouldNotBeUpdated(bazel);
    return result;
  }

  private void shouldBeUpdated(BuilderRunner bazel) throws Exception {
    assertThat(getDebugId(bazel)).isEqualTo(String.valueOf(currentDebugId));
  }

  private void shouldNotBeUpdated(BuilderRunner bazel) throws Exception {
    assertThat(getDebugId(bazel)).isNotEqualTo(String.valueOf(currentDebugId));
  }

  private void buildSucceeded(ProcessResult result) {
    assertThat(findPattern(result, "INFO: Build completed successfully")).isTrue();
  }

  private void buildFailed(ProcessResult result) {
    assertThat(findPattern(result, "FAILED: Build did NOT complete successfully")).isTrue();
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
