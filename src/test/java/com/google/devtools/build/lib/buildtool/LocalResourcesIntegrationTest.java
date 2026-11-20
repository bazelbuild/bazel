// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Integration tests for configuring local resources through a Starlark function. */
@RunWith(TestParameterInjector.class)
public final class LocalResourcesIntegrationTest extends BuildIntegrationTestCase {

  @Before
  public void writeBuildFiles() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "out",
            outs = ["out.txt"],
            cmd = "echo out > $@",
        )
        """);
  }

  private void writeFunction(String result) throws Exception {
    write("resources.bzl", "def local_resources():", "    return " + result);
    addOptions("--local_resources_function=//:resources.bzl%local_resources");
  }

  @Test
  public void functionSetsLocalResources(@TestParameter boolean mergedAnalysisExecution)
      throws Exception {
    writeFunction("{'cpu': 2, 'gpu-2': 1.5, 'gpu-memory': '16'}");
    addOptions(
        "--experimental_merged_skyframe_analysis_execution=" + mergedAnalysisExecution,
        "--local_resources=cpu=3",
        "--local_resources=gpu-2=2",
        "--local_resources=extra=7");

    buildTarget("//:out");

    ResourceSet resources =
        getCommandEnvironment().getLocalResourceManager().getAvailableResources();
    assertThat(resources).isNotNull();
    assertThat(resources.get("cpu")).isEqualTo(2.0);
    assertThat(resources.get("gpu-2")).isEqualTo(1.5);
    assertThat(resources.get("gpu-memory")).isEqualTo(16.0);
    assertThat(resources.get("extra")).isEqualTo(7.0);
    assertThat(resources.get("memory"))
        .isEqualTo(.67 * LocalHostCapacity.getLocalHostCapacity().getMemoryMb());
  }

  @Test
  public void emptyDictionaryPreservesCommandLineResources() throws Exception {
    writeFunction("{}");
    addOptions("--local_resources=cpu=3", "--local_resources=gpu=2");

    buildTarget("//:out");

    ResourceSet resources =
        getCommandEnvironment().getLocalResourceManager().getAvailableResources();
    assertThat(resources.get("cpu")).isEqualTo(3.0);
    assertThat(resources.get("gpu")).isEqualTo(2.0);
  }

  @Test
  public void functionLoadedFromGeneratedRepository(
      @TestParameter({"@resources", "@@+resources_ext+generated"}) String repo) throws Exception {
    write(
        "repo.bzl",
        """
        def _repo_impl(ctx):
            ctx.file("BUILD.bazel", "")
            ctx.file("values.bzl", "GPU_MEMORY = 24")
            ctx.file("resources.bzl", '''
        load(":values.bzl", "GPU_MEMORY")
        def local_resources():
            return {"gpu-memory": GPU_MEMORY}
        ''')

        resources_repo = repository_rule(implementation = _repo_impl)

        def _extension_impl(ctx):
            resources_repo(name = "generated")

        resources_ext = module_extension(implementation = _extension_impl)
        """);
    FileSystemUtils.appendIsoLatin1(
        getWorkspace().getRelative("MODULE.bazel"),
        "ext = use_extension('//:repo.bzl', 'resources_ext')",
        "use_repo(ext, resources = 'generated')");
    addOptions("--local_resources_function=" + repo + "//:resources.bzl%local_resources");

    buildTarget("//:out");

    assertThat(
            getCommandEnvironment()
                .getLocalResourceManager()
                .getAvailableResources()
                .get("gpu-memory"))
        .isEqualTo(24.0);
  }

  @Test
  public void changedTransitiveLoadUpdatesResources() throws Exception {
    write("values.bzl", "GPU_MEMORY = 16");
    write(
        "resources.bzl",
        "load(':values.bzl', 'GPU_MEMORY')",
        "def local_resources():",
        "    return {'gpu-memory': GPU_MEMORY}");
    addOptions("--local_resources_function=//:resources.bzl%local_resources");
    buildTarget("//:out");

    write("values.bzl", "GPU_MEMORY = 32");
    buildTarget("//:out");

    assertThat(
            getCommandEnvironment()
                .getLocalResourceManager()
                .getAvailableResources()
                .get("gpu-memory"))
        .isEqualTo(32.0);
  }

  @Test
  public void clearingFunctionRemovesPreviousResources() throws Exception {
    writeFunction("{'gpu': 2}");
    addOptions("--local_resources=cpu=3");
    buildTarget("//:out");

    addOptions("--local_resources_function=");
    buildTarget("//:out");

    ResourceSet resources =
        getCommandEnvironment().getLocalResourceManager().getAvailableResources();
    assertThat(resources.get("cpu")).isEqualTo(3.0);
    assertThat(resources.getResources()).doesNotContainKey("gpu");
  }

  private void assertInvalidFunction(String message) {
    assertThat(assertThrows(InvalidConfigurationException.class, () -> buildTarget("//:out")))
        .hasMessageThat()
        .contains(message);
  }

  @Test
  public void malformedReference(
      @TestParameter({"//:resources.bzl", "//:resources.bzl%", "%fn", "//:resources.bzl%fn%fn"})
          String reference) {
    addOptions("--local_resources_function=" + reference);
    assertInvalidFunction("expected //pkg:file.bzl%function");
  }

  @Test
  public void missingFile() {
    addOptions("--local_resources_function=//:missing.bzl%local_resources");
    assertInvalidFunction("failed to load //:missing.bzl");
  }

  @Test
  public void missingFunction() throws Exception {
    write("resources.bzl", "other = 1");
    addOptions("--local_resources_function=//:resources.bzl%local_resources");
    assertInvalidFunction("function 'local_resources' not found");
  }

  @Test
  public void symbolMustBeFunction() throws Exception {
    write("resources.bzl", "local_resources = {'gpu': 1}");
    addOptions("--local_resources_function=//:resources.bzl%local_resources");
    assertInvalidFunction("must be a Starlark function, got dict");
  }

  @Test
  public void functionMustTakeNoArguments() throws Exception {
    write("resources.bzl", "def local_resources(ctx):", "    return {}");
    addOptions("--local_resources_function=//:resources.bzl%local_resources");
    assertInvalidFunction("missing 1 required positional argument: ctx");
  }

  @Test
  public void functionFailureReportsStack() throws Exception {
    write("resources.bzl", "def local_resources():", "    fail('cannot detect resources')");
    addOptions("--local_resources_function=//:resources.bzl%local_resources");
    assertInvalidFunction("cannot detect resources");
  }

  @Test
  public void returnValueMustBeDictionary(@TestParameter({"None", "[]", "1"}) String value)
      throws Exception {
    writeFunction(value);
    assertInvalidFunction("local resources");
  }

  @Test
  public void resourceNamesMustBeStrings() throws Exception {
    writeFunction("{1: 2}");
    assertInvalidFunction("local resources");
  }

  @Test
  public void invalidResourceValue(
      @TestParameter({"True", "[]", "'HOST_CPUS*-1'"}) String value)
      throws Exception {
    writeFunction("{'gpu': " + value + "}");
    assertInvalidFunction("local resource 'gpu'");
  }
}
