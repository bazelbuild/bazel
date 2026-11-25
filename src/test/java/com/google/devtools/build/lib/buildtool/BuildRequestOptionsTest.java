// Copyright 2007 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.worker.WorkerModule;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code BuildRequestOptions} */
@RunWith(JUnit4.class)
public class BuildRequestOptionsTest extends BuildIntegrationTestCase {
  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    BlazeModule module = new WorkerModule();
    return super.getRuntimeBuilder().addBlazeModule(module);
  }

  @Override
  protected ImmutableSet<EventKind> additionalEventsToCollect() {
    return ImmutableSet.of(EventKind.PROGRESS, EventKind.START);
  }

  /** Creates a minimal request using currently-installed request options. */
  private BuildRequest createMinimalBuildRequestWithCurrentOptions() throws Exception {
    CommandEnvironment env = runtimeWrapper.newCommand();
    for (BlazeModule module : getRuntime().getBlazeModules()) {
      module.beforeCommand(env);
    }
    for (BlazeModule module : getRuntime().getBlazeModules()) {
      env.getSkyframeExecutor().injectExtraPrecomputedValues(module.getPrecomputedValues());
    }
    env.syncPackageLoading(env.getOptions());

    return BuildRequest.builder().setCommandName("build").setOptions(env.getOptions()).build();
  }

  @Test
  public void testNoBuild() throws Exception {
    write(
        "nobuild/BUILD",
        """
        genrule(
            name = "good",
            srcs = ["BUILD"],
            outs = ["out.good"],
            cmd = "/bin/cp $(location BUILD) $(location out.good)",
        )

        genrule(
            name = "bad",
            srcs = ["//missing"],
            outs = ["out.bad"],
            cmd = ":",
        )
        """);

    // Strategy has to be specified here because the default strategy init is not executed
    // in non-integration tests.
    addOptions("--spawn_strategy=local");
    buildTarget("//nobuild:good"); // all three phases OK.

    addOptions("--nobuild");
    buildTarget("//nobuild:good"); // loading + init phases OK

    addOptions("--nobuild");
    try {
      buildTarget("//nobuild:bad"); // init phase fails
      fail();
    } catch (LoadingFailedException | ViewCreationFailedException e) {
      /* ok */
    }
  }

  /** Checks an invalid build option from perspective of a BuildTool object. */
  @Test
  public void testInvalidBuildRequest() throws Exception {
    // Specifying an option from Blaze analyze for a build request.
    addOptions("--jobs=10", "--local_test_jobs=20");
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();

    new BuildTool(getCommandEnvironment()).validateOptions(request);

    assertContainsEvent("High value for --local_test_jobs");
  }

  /** Checks a valid build option for validity. */
  @Test
  public void testValidBuildOption() throws Exception {
    // Specifying a valid option for a build request.
    addOptions("--verbose_failures");
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();

    // Validating the build command for invalid options.
    List<String> issues = request.validateOptions();
    String testFailMsg = "No warning for build options should have been issued.";
    assertWithMessage(testFailMsg).that(issues).isNotNull();
    assertWithMessage(testFailMsg).that(issues).isEmpty();
  }

  private abstract static class NonExistentOptions extends OptionsBase {}

  @Test
  public void testNonExistentOption() throws Exception {
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();
    assertThat(request.getOptions(NonExistentOptions.class)).isNull();
  }

  /** ** Local resources tests *** */
  @Test
  public void setExplicitCpu() throws Exception {
    addOptions("--local_resources=cpu=5");
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();
    ResourceManager resourceManager = new ResourceManager();
    ExecutionTool.configureResourceManager(resourceManager, request);
    assertThat(resourceManager.availableResources.getCpuUsage()).isEqualTo(5.0);
  }

  @Test
  public void setDefaultCpu() throws Exception {
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();
    ResourceManager resourceManager = new ResourceManager();
    ExecutionTool.configureResourceManager(resourceManager, request);
    assertThat(resourceManager.availableResources.getCpuUsage())
        .isEqualTo(LocalHostCapacity.getLocalHostCapacity().getCpuUsage());
  }

  @Test
  public void setRelativeCpu() throws Exception {
    addOptions("--local_resources=cpu=HOST_CPUS*2");
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();
    ResourceManager resourceManager = new ResourceManager();
    ExecutionTool.configureResourceManager(resourceManager, request);
    assertThat(resourceManager.availableResources.getCpuUsage())
        .isEqualTo(LocalHostCapacity.getLocalHostCapacity().getCpuUsage() * 2);
  }

  @Test
  public void setDefaultRam() throws Exception {
    BuildRequest request = createMinimalBuildRequestWithCurrentOptions();
    ResourceManager resourceManager = new ResourceManager();
    ExecutionTool.configureResourceManager(resourceManager, request);
    double localRam = LocalHostCapacity.getLocalHostCapacity().getMemoryMb();
    assertThat(resourceManager.availableResources.getMemoryMb())
        .isWithin(localRam * .0001)
        .of(LocalHostCapacity.getLocalHostCapacity().getMemoryMb() * .67);
  }

  /**** GPlatform options tests ****/

  private void writeGplatformBuildFile() throws Exception {
    write(
        "echo_compilation_mode/BUILD",
        """
        genrule(
            name = "echo_compilation_mode",
            outs = ["out"],
            cmd = "/bin/echo $(COMPILATION_MODE) > $(location out)",
        )
        """);

    // Strategy has to be specified here because the default strategy init is not executed
    // in non-integration tests.
    addOptions("--spawn_strategy=local");
  }

  @Test
  public void testGplatformDefault() throws Exception {
    writeGplatformBuildFile();
    addOptions("--compilation_mode=fastbuild");
    buildTarget("//echo_compilation_mode");
    Path out = getExecutableLocation("//echo_compilation_mode:out");
    assertThat(FileSystemUtils.readContentAsLatin1(out)).isEqualTo("fastbuild\n");
  }

  @Test
  public void testGplatformOverrideCompilationMode() throws Exception {
    writeGplatformBuildFile();
    addOptions("--compilation_mode=opt");
    buildTarget("//echo_compilation_mode");
    Path out = getExecutableLocation("//echo_compilation_mode:out");
    assertThat(FileSystemUtils.readContentAsLatin1(out)).isEqualTo("opt\n");
  }

  @Test
  public void bustActionCaches() throws Exception {
    write(
        "foo/BUILD",
        "genrule(name = 'top', srcs = [':mid'], outs = ['top.out'], cmd = 'cp $< $@')",
        "genrule(name = 'mid', srcs = [':bottom'], outs = ['mid.out'], cmd = 'cp $< $@')",
        "genrule(name = 'bottom', outs = ['bottom.out'], cmd = 'echo bottom > $@')");

    buildTarget("//foo:top");
    assertContainsEvent("Executing genrule //foo:top");
    assertContainsEvent("Executing genrule //foo:mid");
    assertContainsEvent("Executing genrule //foo:bottom");

    addOptions("--bust_action_caches=//foo:mid");
    buildTarget("//foo:top");
    assertDoesNotContainEvent("Executing genrule //foo:top");
    assertContainsEvent("Executing genrule //foo:mid");
    assertDoesNotContainEvent("Executing genrule //foo:bottom");
  }

  @Test
  public void bustActionCaches_commaSeparated() throws Exception {
    write(
        "foo/BUILD",
        "genrule(name = 'top', srcs = [':mid1', ':mid2', ':untouched'], outs = ['top.out'], cmd ="
            + " 'cat $(SRCS) > $@')",
        "genrule(name = 'mid1', srcs = [':bottom'], outs = ['mid1.out'], cmd = 'cp $< $@')",
        "genrule(name = 'mid2', srcs = [':bottom'], outs = ['mid2.out'], cmd = 'cp $< $@')",
        "genrule(name = 'untouched', srcs = [':bottom'], outs = ['untouched.out'], cmd = 'cp $<"
            + " $@')",
        "genrule(name = 'bottom', outs = ['bottom.out'], cmd = 'echo bottom > $@')");

    buildTarget("//foo:top");
    assertContainsEvent("Executing genrule //foo:top");
    assertContainsEvent("Executing genrule //foo:mid1");
    assertContainsEvent("Executing genrule //foo:mid2");
    assertContainsEvent("Executing genrule //foo:untouched");
    assertContainsEvent("Executing genrule //foo:bottom");

    addOptions("--bust_action_caches=//foo:mid1,//foo:mid2");
    buildTarget("//foo:top");
    assertDoesNotContainEvent("Executing genrule //foo:top");
    assertContainsEvent("Executing genrule //foo:mid1");
    assertContainsEvent("Executing genrule //foo:mid2");
    assertDoesNotContainEvent("Executing genrule //foo:untouched");
    assertDoesNotContainEvent("Executing genrule //foo:bottom");
  }

  @Test
  public void bustActionCaches_lastFlagOverrides() throws Exception {
    write(
        "foo/BUILD",
        "genrule(name = 'top', srcs = [':mid1', ':mid2', ':untouched'], outs = ['top.out'], cmd ="
            + " 'cat $(SRCS) > $@')",
        "genrule(name = 'mid1', srcs = [':bottom'], outs = ['mid1.out'], cmd = 'cp $< $@')",
        "genrule(name = 'mid2', srcs = [':bottom'], outs = ['mid2.out'], cmd = 'cp $< $@')",
        "genrule(name = 'untouched', srcs = [':bottom'], outs = ['untouched.out'], cmd = 'cp $<"
            + " $@')",
        "genrule(name = 'bottom', outs = ['bottom.out'], cmd = 'echo bottom > $@')");

    buildTarget("//foo:top");
    assertContainsEvent("Executing genrule //foo:top");
    assertContainsEvent("Executing genrule //foo:mid1");
    assertContainsEvent("Executing genrule //foo:mid2");
    assertContainsEvent("Executing genrule //foo:untouched");
    assertContainsEvent("Executing genrule //foo:bottom");

    addOptions("--bust_action_caches=//foo:mid1", "--bust_action_caches=//foo:mid2");
    buildTarget("//foo:top");
    assertDoesNotContainEvent("Executing genrule //foo:top");
    assertDoesNotContainEvent("Executing genrule //foo:mid1");
    assertContainsEvent("Executing genrule //foo:mid2");
    assertDoesNotContainEvent("Executing genrule //foo:untouched");
    assertDoesNotContainEvent("Executing genrule //foo:bottom");

    addOptions("--bust_action_caches=//foo:mid1", "--bust_action_caches=");
    buildTarget("//foo:top");
    assertDoesNotContainEvent("Executing genrule //foo:top");
    assertDoesNotContainEvent("Executing genrule //foo:mid1");
    assertDoesNotContainEvent("Executing genrule //foo:mid2");
    assertDoesNotContainEvent("Executing genrule //foo:untouched");
    assertDoesNotContainEvent("Executing genrule //foo:bottom");
  }
}
