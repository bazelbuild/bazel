// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.blackbox.tests.manageddirs;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Before;
import org.junit.Test;

/** Tests for managed directories. */
public class ManagedDirectoriesBlackBoxTest extends AbstractBlackBoxTest {
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
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();
  }

  @Test
  public void testNodeModulesDeleted() throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();

    Path nodeModules = context().getWorkDir().resolve("node_modules");
    assertThat(nodeModules.toFile().isDirectory()).isTrue();
    PathUtils.deleteTree(nodeModules);

    buildExpectRepositoryRuleCalled();
    checkProjectFiles();
  }

  @Test
  public void testNodeModulesDeletedAndRecreated() throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();

    Path nodeModules = context().getWorkDir().resolve("node_modules");
    assertThat(nodeModules.toFile().isDirectory()).isTrue();

    Path nodeModulesBackup = context().getWorkDir().resolve("node_modules_backup");
    PathUtils.copyTree(nodeModules, nodeModulesBackup);
    PathUtils.deleteTree(nodeModules);

    PathUtils.copyTree(nodeModulesBackup, nodeModules);

    buildExpectRepositoryRuleNotCalled();
    checkProjectFiles();
  }

  @Test
  public void testBuildProjectFetchNotRecalled() throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();
    buildExpectRepositoryRuleNotCalled();
    checkProjectFiles();
  }

  @Test
  public void testWithoutFlag() throws Exception {
    generateProject();
    ProcessResult result = context().bazel().shouldFail().build("//...");
    assertThat(result.errString())
        .contains(
            "parameter 'managed_directories' is experimental and thus unavailable"
                + " with the current flags.");
  }

  private BuilderRunner bazel() {
    return bazel(false);
  }

  private BuilderRunner bazel(boolean watchFs) {
    currentDebugId = random.nextInt();
    String[] flags =
        watchFs
            ? new String[] {"--experimental_allow_incremental_repository_updates", "--watchfs=true"}
            : new String[] {"--experimental_allow_incremental_repository_updates"};
    return context().bazel().withFlags(flags).withEnv("DEBUG_ID", String.valueOf(currentDebugId));
  }

  @Test
  public void testChangeOfFileTextUnderNodeModules() throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();

    Path nodeModules = context().getWorkDir().resolve("node_modules");
    Path modulePackageJson = nodeModules.resolve("example-module/package.json");
    assertThat(modulePackageJson.toFile().exists()).isTrue();

    // Assert that non-structural changes are not detected.
    PathUtils.append(modulePackageJson, "# comment");

    buildExpectRepositoryRuleNotCalled();
    checkProjectFiles();
  }

  @Test
  public void testLoadIsNotCalledForManagedDirectories() throws Exception {
    generateProject();
    Path workspaceFile = context().getWorkDir().resolve(WORKSPACE);
    PathUtils.append(workspaceFile, "load('@non_existing//:target.bzl', 'some_symbol')");

    // Test that there is error when loading, so we parsed managed directories successfully.
    ProcessResult result = bazel().shouldFail().build("//...");
    assertThat(findPattern(result, "ERROR: Failed to load Starlark extension")).isTrue();
  }

  @Test
  public void testWithBazelTools() throws Exception {
    generateProject();
    Path workspaceFile = context().getWorkDir().resolve(WORKSPACE);
    PathUtils.append(
        workspaceFile,
        "load(\"@bazel_tools//tools/build_defs/repo:http.bzl\", \"http_archive\", \"http_file\")");
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();
  }

  @Test
  public void testAddManagedDirectoriesLater() throws Exception {
    // Start the server, have things cached.
    context().write("BUILD", "");
    bazel().build("//...");

    // Now that we generate the project and have managed directories updated, we are also testing,
    // that managed directories are re-read correctly from the changed file.
    generateProject();
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();

    // Test everything got cached.
    buildExpectRepositoryRuleNotCalled();
    checkProjectFiles();
  }

  @Test
  public void testFilesUnderChangedManagedDirectoriesRefreshed() throws Exception {
    doTestFilesUnderManagedDirectoriesRefreshed(false);
  }

  @Test
  public void testFilesUnderChangedManagedDirectoriesRefreshedWatchFs() throws Exception {
    doTestFilesUnderManagedDirectoriesRefreshed(true);
  }

  private void doTestFilesUnderManagedDirectoriesRefreshed(boolean watchFs) throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled(false, watchFs);
    checkProjectFiles();

    // Now remove the ManagedDirectories, and change the package version - it should still work.
    List<String> properWorkspaceText = context().read("WORKSPACE");

    context()
        .write(
            "WORKSPACE",
            "workspace(name = \"fine_grained_user_modules\")",
            "load(\":use_node_modules.bzl\", \"generate_fine_grained_node_modules\")",
            "generate_fine_grained_node_modules(name = \"generated_node_modules\",",
            "package_json = \"//:package.json\",)");
    Path packageJson =
        PathUtils.resolve(context().getWorkDir(), "node_modules", "example-module", "package.json");
    assertThat(packageJson.toFile().exists()).isTrue();

    // Now we are building it without managed directories, both managed directories and
    // RepositoryDirectoryValue will be dirty - we expect repository rule to be called again.
    buildExpectRepositoryRuleCalled(false, watchFs);
    checkProjectFiles();

    // Now change files directly in generated area, and build.
    List<String> oldPackageJson = PathUtils.readFile(packageJson);
    PathUtils.writeFile(
        packageJson,
        "{",
        "  \"license\": \"MIT\",",
        "  \"main\": \"example-module.js\",",
        "  \"name\": \"example-module\",",
        "  \"repository\": {",
        "    \"type\": \"git\",",
        "    \"url\": \"aaa\",",
        "  },",
        "  \"version\": \"7.7.7\"",
        "}");
    Path build = context().getWorkDir().resolve("BUILD");
    List<String> oldBuild = PathUtils.readFile(build);
    PathUtils.writeFile(
        build,
        "load(\":test_rule.bzl\", \"test_rule\")",
        "test_rule(",
        "    name = \"test_generated_deps\",",
        "    module_source = \"@generated_node_modules//:example-module\",",
        "    version = \"7.7.7\"",
        ")");

    // Test rule inputs has changed, so the build is not cached; however, the repository rule
    // is not rerun, since it's inputs (including managed directories settings) were not changed,
    // so debug_id is the same.
    buildExpectRepositoryRuleNotCalled();
    checkProjectFiles("7.7.7");

    // And is cached.
    buildExpectRepositoryRuleNotCalled();

    // Now change just the managed directories and see the generated version comes up.
    PathUtils.writeFile(
        context().getWorkDir().resolve(WORKSPACE), properWorkspaceText.toArray(new String[0]));
    PathUtils.writeFile(build, oldBuild.toArray(new String[0]));
    buildExpectRepositoryRuleCalled(false, watchFs);
    checkProjectFiles("0.2.0");
  }

  @Test
  public void testManagedDirectoriesSettingsAndManagedDirectoriesFilesChangeSimultaneously()
      throws Exception {
    doTestManagedDirectoriesSettingsAndManagedDirectoriesFilesChangeSimultaneously(false);
  }

  @Test
  public void testManagedDirectoriesSettingsAndManagedDirectoriesFilesChangeSimultaneouslyWatchFs()
      throws Exception {
    doTestManagedDirectoriesSettingsAndManagedDirectoriesFilesChangeSimultaneously(true);
  }

  private void doTestManagedDirectoriesSettingsAndManagedDirectoriesFilesChangeSimultaneously(
      boolean watchFs) throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled(false, watchFs);
    checkProjectFiles();

    // Modify managed directories somehow.
    context()
        .write(
            "WORKSPACE",
            "workspace(name = \"fine_grained_user_modules\",",
            "managed_directories = {'@generated_node_modules': ['node_modules', 'something']})",
            "load(\":use_node_modules.bzl\", \"generate_fine_grained_node_modules\")",
            "generate_fine_grained_node_modules(name = \"generated_node_modules\",",
            "package_json = \"//:package.json\",)");
    Path packageJson =
        PathUtils.resolve(context().getWorkDir(), "node_modules", "example-module", "package.json");
    assertThat(packageJson.toFile().exists()).isTrue();

    // Modify generated package.json under the managed directory.
    List<String> oldPackageJson = PathUtils.readFile(packageJson);
    PathUtils.writeFile(
        packageJson,
        "{",
        "  \"license\": \"MIT\",",
        "  \"main\": \"example-module.js\",",
        "  \"name\": \"example-module\",",
        "  \"repository\": {",
        "    \"type\": \"git\",",
        "    \"url\": \"aaa\",",
        "  },",
        "  \"version\": \"7.7.7\"",
        "}");
    // Expect files under managed directories be regenerated
    // and changes under managed directories be lost.
    buildExpectRepositoryRuleCalled(false, watchFs);
    checkProjectFiles();
  }

  @Test
  public void testRepositoryOverrideWithManagedDirectories() throws Exception {
    generateProject();

    Path override = context().getTmpDir().resolve("override");
    PathUtils.writeFile(override.resolve(WORKSPACE));
    // Just define some similar target.
    PathUtils.writeFile(
        override.resolve("BUILD"),
        "genrule(",
        "    name = \"example-module\",",
        "    srcs = [],",
        "    cmd = \"touch $(location package.json)\",",
        "    outs = [\"package.json\"],",
        "    visibility = ['//visibility:public'],",
        ")");

    BuilderRunner bazel =
        bazel().withFlags("--override_repository=generated_node_modules=" + override.toString());
    ProcessResult result = bazel.shouldFail().build("@generated_node_modules//:example-module");
    assertThat(result.errString())
        .contains(
            "ERROR: Overriding repositories is not allowed"
                + " for the repositories with managed directories."
                + "\nThe following overridden external repositories"
                + " have managed directories: @generated_node_modules");

    // Assert the result stays the same even when managed directories has not changed.
    result = bazel.shouldFail().build("@generated_node_modules//:example-module");
    assertThat(result.errString())
        .contains(
            "ERROR: Overriding repositories is not allowed"
                + " for the repositories with managed directories."
                + "\nThe following overridden external repositories"
                + " have managed directories: @generated_node_modules");
  }

  @Test
  public void testRepositoryOverrideChangeToConflictWithManagedDirectories() throws Exception {
    generateProject();
    buildExpectRepositoryRuleCalled();
    checkProjectFiles();

    Path override = context().getTmpDir().resolve("override");
    PathUtils.writeFile(override.resolve(WORKSPACE));
    // Just define some similar target.
    PathUtils.writeFile(
        override.resolve("BUILD"),
        "genrule(",
        "    name = \"example-module\",",
        "    srcs = [],",
        "    cmd = \"touch $(location package.json)\",",
        "    outs = [\"package.json\"],",
        "    visibility = ['//visibility:public'],",
        ")");

    // Now the overrides change.
    BuilderRunner bazel =
        bazel().withFlags("--override_repository=generated_node_modules=" + override.toString());
    ProcessResult result = bazel.shouldFail().build("@generated_node_modules//:example-module");
    assertThat(result.errString())
        .contains(
            "ERROR: Overriding repositories is not allowed"
                + " for the repositories with managed directories."
                + "\nThe following overridden external repositories"
                + " have managed directories: @generated_node_modules");
  }

  /**
   * The test to verify that WORKSPACE file can not be a symlink when managed directories are used.
   *
   * <p>The test of the case, when WORKSPACE file is a symlink, but not managed directories are
   * used, is in {@link WorkspaceBlackBoxTest#testWorkspaceFileIsSymlink()}
   */
  @Test
  public void testWorkspaceSymlinkThrowsWithManagedDirectories() throws Exception {
    generateProject();

    Path workspaceFile = context().getWorkDir().resolve(WORKSPACE);
    assertThat(workspaceFile.toFile().delete()).isTrue();

    Path tempWorkspace = Files.createTempFile(context().getTmpDir(), WORKSPACE, "");
    PathUtils.writeFile(
        tempWorkspace,
        "workspace(name = \"fine_grained_user_modules\",",
        "managed_directories = {'@generated_node_modules': ['node_modules']})",
        "",
        "load(\":use_node_modules.bzl\", \"generate_fine_grained_node_modules\")",
        "",
        "generate_fine_grained_node_modules(",
        "    name = \"generated_node_modules\",",
        "    package_json = \"//:package.json\",",
        ")");
    Files.createSymbolicLink(workspaceFile, tempWorkspace);

    ProcessResult result = bazel().shouldFail().build("//...");
    assertThat(
            findPattern(
                result,
                "WORKSPACE file can not be a symlink if incrementally updated directories are"
                    + " used."))
        .isTrue();
  }

  private void generateProject() throws IOException {
    writeProjectFile("BUILD.test", "BUILD");
    writeProjectFile("WORKSPACE.test", "WORKSPACE");
    writeProjectFile("bazelignore.test", ".bazelignore");
    writeProjectFile("package.json", "package.json");
    writeProjectFile("test_rule.bzl", "test_rule.bzl");
    writeProjectFile("use_node_modules.bzl", "use_node_modules.bzl");
  }

  private void writeProjectFile(String oldName, String newName) throws IOException {
    String text = ResourceFileLoader.loadResource(ManagedDirectoriesBlackBoxTest.class, oldName);
    assertThat(text).isNotNull();
    assertThat(text).isNotEmpty();
    context().write(newName, text);
  }

  private void checkProjectFiles() throws IOException {
    checkProjectFiles("0.2.0");
  }

  private void checkProjectFiles(String version) throws IOException {
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
    String versionString = String.format("\"version\": \"%s\"", version);
    assertThat(text.stream().anyMatch(s -> s.trim().equals(versionString))).isTrue();
  }

  private String getDebugId(BuilderRunner bazel) throws Exception {
    Path path = context().resolveExecRootPath(bazel, "external/generated_node_modules/debug_id");
    List<String> lines = PathUtils.readFile(path);
    assertThat(lines.size()).isEqualTo(1);
    return lines.get(0);
  }

  private ProcessResult buildExpectRepositoryRuleCalled() throws Exception {
    return buildExpectRepositoryRuleCalled(false, false);
  }

  private ProcessResult buildExpectRepositoryRuleCalled(boolean debug, boolean watchFs)
      throws Exception {
    BuilderRunner bazel = bazel(watchFs);
    if (debug) {
      bazel.enableDebug();
    }
    ProcessResult result = bazel.build("//...");
    buildSucceeded(result);
    debugIdShouldBeUpdated(bazel);
    return result;
  }

  private ProcessResult buildExpectRepositoryRuleNotCalled() throws Exception {
    return buildExpectRepositoryRuleNotCalled(false);
  }

  private ProcessResult buildExpectRepositoryRuleNotCalled(boolean debug) throws Exception {
    BuilderRunner bazel = bazel();
    if (debug) {
      bazel.enableDebug();
    }
    ProcessResult result = bazel.build("//...");
    buildSucceeded(result);
    debugIdShouldNotBeUpdated(bazel);
    return result;
  }

  private void debugIdShouldBeUpdated(BuilderRunner bazel) throws Exception {
    assertThat(getDebugId(bazel)).isEqualTo(String.valueOf(currentDebugId));
  }

  private void debugIdShouldNotBeUpdated(BuilderRunner bazel) throws Exception {
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
}
