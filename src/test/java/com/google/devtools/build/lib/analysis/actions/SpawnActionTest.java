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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.extra.EnvironmentVariable;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SpawnAction}.
 */
@RunWith(JUnit4.class)
public class SpawnActionTest extends BuildViewTestCase {
  private Artifact welcomeArtifact;
  private Artifact destinationArtifact;
  private Artifact jarArtifact;
  private AnalysisTestUtil.CollectingAnalysisEnvironment collectingAnalysisEnvironment;

  private SpawnAction.Builder builder() {
    return new SpawnAction.Builder();
  }

  @Before
  public final void createArtifacts() throws Exception {
    collectingAnalysisEnvironment = new AnalysisTestUtil.CollectingAnalysisEnvironment(
        getTestAnalysisEnvironment());
    welcomeArtifact = getSourceArtifact("pkg/welcome.txt");
    jarArtifact = getSourceArtifact("pkg/exe.jar");
    destinationArtifact = getBinArtifactWithNoOwner("dir/destination.txt");
  }

  private SpawnAction createCopyFromWelcomeToDestination(Map<String, String> environmentVariables) {
    PathFragment cp = new PathFragment("/bin/cp");
    List<String> arguments = asList(welcomeArtifact.getExecPath().getPathString(),
        destinationArtifact.getExecPath().getPathString());

    Action[] actions = builder()
        .addInput(welcomeArtifact)
        .addOutput(destinationArtifact)
        .setExecutionInfo(ImmutableMap.<String, String>of("local", ""))
        .setExecutable(cp)
        .addArguments(arguments)
        .setProgressMessage("hi, mom!")
        .setMnemonic("Dummy")
        .setEnvironment(environmentVariables)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    return (SpawnAction) actions[0];
  }

  @Test
  public void testWelcomeArtifactIsInput() {
    SpawnAction copyFromWelcomeToDestination =
        createCopyFromWelcomeToDestination(ImmutableMap.<String, String>of());
    Iterable<Artifact> inputs = copyFromWelcomeToDestination.getInputs();
    assertEquals(Sets.newHashSet(welcomeArtifact), Sets.newHashSet(inputs));
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    SpawnAction copyFromWelcomeToDestination =
        createCopyFromWelcomeToDestination(ImmutableMap.<String, String>of());
    Collection<Artifact> outputs = copyFromWelcomeToDestination.getOutputs();
    assertEquals(Sets.newHashSet(destinationArtifact), Sets.newHashSet(outputs));
  }

  @Test
  public void testBuilder() throws Exception {
    Artifact input = getSourceArtifact("input");
    Artifact output = getBinArtifactWithNoOwner("output");
    Action[] actions = builder()
        .addInput(input)
        .addOutput(output)
        .setExecutable(scratch.file("/bin/xxx").asFragment())
        .setProgressMessage("Test")
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertEquals(ActionsTestUtil.NULL_ACTION_OWNER.getLabel(),
        action.getOwner().getLabel());
    assertThat(action.getInputs()).containsExactlyElementsIn(asList(input));
    assertThat(action.getOutputs()).containsExactlyElementsIn(asList(output));
    assertEquals(AbstractAction.DEFAULT_RESOURCE_SET, action.getSpawn().getLocalResources());
    assertThat(action.getArguments()).containsExactlyElementsIn(asList("/bin/xxx"));
    assertEquals("Test", action.getProgressMessage());
  }

  @Test
  public void testBuilderWithExecutable() throws Exception {
    Action[] actions = builder()
        .setExecutable(welcomeArtifact)
        .addOutput(destinationArtifact)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertThat(action.getArguments())
        .containsExactlyElementsIn(asList(welcomeArtifact.getExecPath().getPathString()));
  }

  @Test
  public void testBuilderWithJavaExecutable() throws Exception {
    Action[] actions = builder()
        .addOutput(destinationArtifact)
        .setJavaExecutable(scratch.file("/bin/java").asFragment(),
            jarArtifact, "MyMainClass", asList("-jvmarg"))
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertEquals(asList("/bin/java", "-Xverify:none", "-jvmarg", "-cp",
        "pkg/exe.jar", "MyMainClass"), action.getArguments());
  }

  @Test
  public void testBuilderWithJavaExecutableAndParameterFile() throws Exception {
    useConfiguration("--min_param_file_size=0");
    collectingAnalysisEnvironment = new AnalysisTestUtil.CollectingAnalysisEnvironment(
        getTestAnalysisEnvironment());
    Artifact output = getBinArtifactWithNoOwner("output");
    Artifact paramFile = getBinArtifactWithNoOwner("output-2.params");
    Action[] actions = builder()
        .addOutput(output)
        .setJavaExecutable(
            scratch.file("/bin/java").asFragment(), jarArtifact, "MyMainClass", asList("-jvmarg"))
        .addArgument("-X")
        .useParameterFile(ParameterFileType.UNQUOTED)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    if (getMutableActionGraph() != null) {
      // Otherwise, CachingAnalysisEnvironment.registerAction() registers the action. We cannot
      // use STUB_ANALYSIS_ENVIRONMENT here because we also need a BuildConfiguration.
      collectingAnalysisEnvironment.registerWith(getMutableActionGraph());
    }
    assertEquals(asList("/bin/java", "-Xverify:none", "-jvmarg", "-cp",
        "pkg/exe.jar", "MyMainClass", "@" + paramFile.getExecPathString()),
        action.getArguments());
    assertThat(
        ImmutableList.copyOf(
            ((ParameterFileWriteAction) getGeneratingAction(paramFile)).getContents()))
        .containsExactly("-X");
    MoreAsserts.assertContainsSublist(actionInputsToPaths(action.getSpawn().getInputFiles()),
        "pkg/exe.jar");
  }

  @Test
  public void testBuilderWithJavaExecutableAndParameterFileAndParameterFileFlag() throws Exception {
    useConfiguration("--min_param_file_size=0");
    collectingAnalysisEnvironment = new AnalysisTestUtil.CollectingAnalysisEnvironment(
        getTestAnalysisEnvironment());

    Artifact output = getBinArtifactWithNoOwner("output");
    Artifact paramFile = getBinArtifactWithNoOwner("output-2.params");
    Action[] actions = builder()
        .addOutput(output)
        .setJavaExecutable(
            scratch.file("/bin/java").asFragment(), jarArtifact, "MyMainClass", asList("-jvmarg"))
        .addArgument("-X")
        .useParameterFile(ParameterFileType.UNQUOTED, ISO_8859_1, "--flagfile=")
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    if (getMutableActionGraph() != null) {
      // Otherwise, CachingAnalysisEnvironment.registerAction() registers the action. We cannot
      // use STUB_ANALYSIS_ENVIRONMENT here because we also need a BuildConfiguration.
      collectingAnalysisEnvironment.registerWith(getMutableActionGraph());
    }
    assertEquals(asList("/bin/java", "-Xverify:none", "-jvmarg", "-cp",
        "pkg/exe.jar", "MyMainClass", "--flagfile=" + paramFile.getExecPathString()),
        ImmutableList.copyOf(action.getArguments()));
    assertEquals(Arrays.asList("-X"),
        ImmutableList.copyOf(
            ((ParameterFileWriteAction) getGeneratingAction(paramFile)).getContents()));
    MoreAsserts.assertContainsSublist(actionInputsToPaths(action.getSpawn().getInputFiles()),
        "pkg/exe.jar");
  }

  @Test
  public void testBuilderWithExtraExecutableArguments() throws Exception {
    Action[] actions = builder()
        .addOutput(destinationArtifact)
        .setJavaExecutable(
            scratch.file("/bin/java").asFragment(), jarArtifact, "MyMainClass", asList("-jvmarg"))
        .addExecutableArguments("execArg1", "execArg2")
        .addArguments("arg1")
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertEquals(asList("/bin/java", "-Xverify:none", "-jvmarg", "-cp",
        "pkg/exe.jar", "MyMainClass", "execArg1", "execArg2", "arg1"),
        action.getArguments());
  }

  @Test
  public void testBuilderWithExtraExecutableArgumentsAndParameterFile() throws Exception {
    useConfiguration("--min_param_file_size=0");
    collectingAnalysisEnvironment = new AnalysisTestUtil.CollectingAnalysisEnvironment(
        getTestAnalysisEnvironment());
    Artifact output = getBinArtifactWithNoOwner("output");
    Artifact paramFile = getBinArtifactWithNoOwner("output-2.params");
    Action[] actions = builder()
        .addOutput(output)
        .setJavaExecutable(
            scratch.file("/bin/java").asFragment(), jarArtifact, "MyMainClass", asList("-jvmarg"))
        .addExecutableArguments("execArg1", "execArg2")
        .addArguments("arg1", "arg2", "arg3")
        .useParameterFile(ParameterFileType.UNQUOTED)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    if (getMutableActionGraph() != null) {
      // Otherwise, CachingAnalysisEnvironment.registerAction() registers the action. We cannot
      // use STUB_ANALYSIS_ENVIRONMENT here because we also need a BuildConfiguration.
      collectingAnalysisEnvironment.registerWith(getMutableActionGraph());
    }
    assertEquals(asList("/bin/java", "-Xverify:none", "-jvmarg", "-cp",
        "pkg/exe.jar", "MyMainClass", "execArg1", "execArg2",
        "@" + paramFile.getExecPathString()), action.getSpawn().getArguments());
    assertEquals(asList("/bin/java", "-Xverify:none", "-jvmarg", "-cp",
        "pkg/exe.jar", "MyMainClass", "execArg1", "execArg2",
        "@" + paramFile.getExecPathString()), ImmutableList.copyOf(action.getArguments()));
    assertEquals(Arrays.asList("arg1", "arg2", "arg3"),
        ImmutableList.copyOf(
            ((ParameterFileWriteAction) getGeneratingAction(paramFile)).getContents()));
  }

  @Test
  public void testParameterFiles() throws Exception {
    Artifact output1 = getBinArtifactWithNoOwner("output1");
    Artifact output2 = getBinArtifactWithNoOwner("output2");
    Artifact paramFile = getBinArtifactWithNoOwner("output1-2.params");
    PathFragment executable = new PathFragment("/bin/executable");

    useConfiguration("--min_param_file_size=500");

    String longOption = Strings.repeat("x", 1000);
    SpawnAction spawnAction = ((SpawnAction) builder()
        .addOutput(output1)
        .setExecutable(executable)
        .useParameterFile(ParameterFileType.UNQUOTED)
        .addArgument(longOption)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig)[0]);
    assertThat(spawnAction.getRemainingArguments()).containsExactly(
        "@" + paramFile.getExecPathString()).inOrder();

    useConfiguration("--min_param_file_size=1500");
    spawnAction = ((SpawnAction) builder()
        .addOutput(output2)
        .setExecutable(executable)
        .useParameterFile(ParameterFileType.UNQUOTED)
        .addArgument(longOption)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig)[0]);
    assertThat(spawnAction.getRemainingArguments()).containsExactly(longOption).inOrder();
  }

  @Test
  public void testExtraActionInfo() throws Exception {
    SpawnAction action = createCopyFromWelcomeToDestination(ImmutableMap.<String, String>of());
    ExtraActionInfo info = action.getExtraActionInfo().build();
    assertEquals("Dummy", info.getMnemonic());

    SpawnInfo spawnInfo = info.getExtension(SpawnInfo.spawnInfo);
    assertNotNull(spawnInfo);

    assertThat(spawnInfo.getArgumentList())
        .containsExactlyElementsIn(action.getArguments());

    Iterable<String> inputPaths = Artifact.toExecPaths(
        action.getInputs());
    Iterable<String> outputPaths = Artifact.toExecPaths(
        action.getOutputs());

    assertThat(spawnInfo.getInputFileList()).containsExactlyElementsIn(inputPaths);
    assertThat(spawnInfo.getOutputFileList()).containsExactlyElementsIn(outputPaths);
    Map<String, String> environment = action.getEnvironment();
    assertEquals(environment.size(), spawnInfo.getVariableCount());

    for (EnvironmentVariable variable : spawnInfo.getVariableList()) {
      assertThat(environment).containsEntry(variable.getName(), variable.getValue());
    }
  }

  /**
   * Test that environment variables are not escaped or quoted.
   */
  @Test
  public void testExtraActionInfoEnvironmentVariables() throws Exception {
    Map<String, String> env = ImmutableMap.of(
        "P1", "simple",
        "P2", "spaces are not escaped",
        "P3", ":",
        "P4", "",
        "NONSENSE VARIABLE", "value"
    );

    SpawnInfo spawnInfo = createCopyFromWelcomeToDestination(env).getExtraActionInfo().build()
        .getExtension(SpawnInfo.spawnInfo);
    assertThat(env).hasSize(spawnInfo.getVariableCount());
    for (EnvironmentVariable variable : spawnInfo.getVariableList()) {
      assertThat(env).containsEntry(variable.getName(), variable.getValue());
    }
  }

  @Test
  public void testInputManifestsRemovedIfSupplied() throws Exception {
    Artifact manifest = getSourceArtifact("MANIFEST");
    Action[] actions = builder()
        .addInput(manifest)
        .addRunfilesSupplier(
            new RunfilesSupplierImpl(new PathFragment("destination"), Runfiles.EMPTY, manifest))
        .addOutput(getBinArtifactWithNoOwner("output"))
        .setExecutable(scratch.file("/bin/xxx").asFragment())
        .setProgressMessage("Test")
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    List<String> inputFiles = actionInputsToPaths(action.getSpawn().getInputFiles());
    assertThat(inputFiles).isEmpty();
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact artifactA = getSourceArtifact("a");
    final Artifact artifactB = getSourceArtifact("b");

    ActionTester.runTest(64, new ActionCombinationFactory() {
      @Override
      public Action generate(int i) {
        SpawnAction.Builder builder = builder();
        builder.addOutput(destinationArtifact);

        PathFragment executable = (i & 1) == 0 ? artifactA.getExecPath() : artifactB.getExecPath();
        if ((i & 2) == 0) {
          builder.setExecutable(executable);
        } else {
          builder.setJavaExecutable(executable, jarArtifact, "Main", ImmutableList.<String>of());
        }

        builder.setMnemonic((i & 4) == 0 ? "a" : "b");

        if ((i & 8) == 0) {
          builder.addRunfilesSupplier(runfilesSupplier(artifactA, new PathFragment("a")));
        } else {
          builder.addRunfilesSupplier(runfilesSupplier(artifactB, new PathFragment("a")));
        }

        if ((i & 16) == 0) {
          builder.addRunfilesSupplier(runfilesSupplier(artifactA, new PathFragment("aa")));
        } else {
          builder.addRunfilesSupplier(runfilesSupplier(artifactA, new PathFragment("ab")));
        }

        Map<String, String> env = new HashMap<>();
        if ((i & 32) == 0) {
          env.put("foo", "bar");
        }
        builder.setEnvironment(env);

        Action[] actions = builder.build(ActionsTestUtil.NULL_ACTION_OWNER,
            collectingAnalysisEnvironment, targetConfig);
        collectingAnalysisEnvironment.registerAction(actions);
        return actions[0];
      }
    });
  }

  @Test
  public void testMnemonicMustNotContainSpaces() {
    SpawnAction.Builder builder = builder();
    try {
      builder.setMnemonic("contains space");
      fail("Expected exception");
    } catch (IllegalArgumentException expected) {}
    try {
      builder.setMnemonic("contains\nnewline");
      fail("Expected exception");
    } catch (IllegalArgumentException expected) {}
    try {
      builder.setMnemonic("contains/slash");
      fail("Expected exception");
    } catch (IllegalArgumentException expected) {}
  }

  /**
   * Tests that the ExtraActionInfo proto that's generated from an action, contains Aspect-related
   * information.
   */
  @Test
  public void testGetExtraActionInfoOnAspects() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//a:def.bzl', 'testrule')",
        "testrule(name='a', deps=[':b'])",
        "testrule(name='b')");
    scratch.file(
        "a/def.bzl",
        "def _aspect_impl(target, ctx):",
        "  f = ctx.new_file('foo.txt')",
        "  ctx.action(outputs = [f], command = 'echo foo > \"$1\"')",
        "  return struct(output=f)",
        "def _rule_impl(ctx):",
        "  return struct(files=depset([artifact.output for artifact in ctx.attr.deps]))",
        "aspect1 = aspect(_aspect_impl, attr_aspects=['deps'], ",
        "    attrs = {'parameter': attr.string(values = ['param_value'])})",
        "testrule = rule(_rule_impl, attrs = { ",
        "    'deps' : attr.label_list(aspects = [aspect1]), ",
        "    'parameter': attr.string(default='param_value') })");

    update(
        ImmutableList.of("//a:a"),
        false /* keepGoing */,
        1 /* loadingPhaseThreads */,
        true /* doAnalysis */,
        new EventBus());

    Artifact artifact = getOnlyElement(getFilesToBuild(getConfiguredTarget("//a:a")));
    ExtraActionInfo.Builder extraActionInfo = getGeneratingAction(artifact).getExtraActionInfo();
    assertThat(extraActionInfo.getAspectName()).isEqualTo("//a:def.bzl%aspect1");
    assertThat(extraActionInfo.getAspectParametersMap())
        .containsExactly(
            "parameter", ExtraActionInfo.StringList.newBuilder().addValue("param_value").build());
  }

  private static RunfilesSupplier runfilesSupplier(Artifact manifest, PathFragment dir) {
    return new RunfilesSupplierImpl(dir, Runfiles.EMPTY, manifest);
  }
}
