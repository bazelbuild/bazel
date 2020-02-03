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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.Arrays.asList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
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
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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
    PathFragment cp = PathFragment.create("/bin/cp");
    List<String> arguments = asList(welcomeArtifact.getExecPath().getPathString(),
        destinationArtifact.getExecPath().getPathString());

    Action[] actions =
        builder()
            .addInput(welcomeArtifact)
            .addOutput(destinationArtifact)
            .setExecutionInfo(ImmutableMap.<String, String>of("local", ""))
            .setExecutable(cp)
            .setProgressMessage("hi, mom!")
            .setMnemonic("Dummy")
            .setEnvironment(environmentVariables)
            .addCommandLine(CommandLine.of(arguments))
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    return (SpawnAction) actions[0];
  }

  @Test
  public void testWelcomeArtifactIsInput() {
    SpawnAction copyFromWelcomeToDestination =
        createCopyFromWelcomeToDestination(ImmutableMap.<String, String>of());
    Iterable<Artifact> inputs = copyFromWelcomeToDestination.getInputs().toList();
    assertThat(inputs).containsExactly(welcomeArtifact);
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    SpawnAction copyFromWelcomeToDestination =
        createCopyFromWelcomeToDestination(ImmutableMap.<String, String>of());
    Collection<Artifact> outputs = copyFromWelcomeToDestination.getOutputs();
    assertThat(outputs).containsExactly(destinationArtifact);
  }

  @Test
  public void testExecutionInfoCopied() {
    SpawnAction copyFromWelcomeToDestination =
        createCopyFromWelcomeToDestination(ImmutableMap.of());
    Map<String, String> executionInfo = copyFromWelcomeToDestination.getExecutionInfo();
    assertThat(executionInfo).containsExactly("local", "");
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
        .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertThat(action.getOwner().getLabel())
        .isEqualTo(ActionsTestUtil.NULL_ACTION_OWNER.getLabel());
    assertThat(action.getInputs().toList()).containsExactly(input);
    assertThat(action.getOutputs()).containsExactly(output);
    assertThat(action.getSpawn().getLocalResources())
        .isEqualTo(AbstractAction.DEFAULT_RESOURCE_SET);
    assertThat(action.getArguments()).containsExactly("/bin/xxx");
    assertThat(action.getProgressMessage()).isEqualTo("Test");
  }

  @Test
  public void testBuilderWithExecutable() throws Exception {
    Action[] actions = builder()
        .setExecutable(welcomeArtifact)
        .addOutput(destinationArtifact)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertThat(action.getArguments())
        .containsExactly(welcomeArtifact.getExecPath().getPathString());
  }

  @Test
  public void testBuilderWithJavaExecutable() throws Exception {
    Action[] actions = builder()
        .addOutput(destinationArtifact)
        .setJavaExecutable(scratch.file("/bin/java").asFragment(),
            jarArtifact, "MyMainClass", asList("-jvmarg"))
        .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertThat(action.getArguments())
        .containsExactly(
            "/bin/java", "-Xverify:none", "-jvmarg", "-cp", "pkg/exe.jar", "MyMainClass")
        .inOrder();
  }

  @Test
  public void testBuilderWithJavaExecutableAndParameterFile2() throws Exception {
    useConfiguration("--min_param_file_size=0", "--defer_param_files");
    collectingAnalysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(getTestAnalysisEnvironment());
    Artifact output = getBinArtifactWithNoOwner("output");
    Action[] actions =
        builder()
            .addOutput(output)
            .setJavaExecutable(
                scratch.file("/bin/java").asFragment(),
                jarArtifact,
                "MyMainClass",
                asList("-jvmarg"))
            .addCommandLine(
                CustomCommandLine.builder().add("-X").build(),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    SpawnAction action = (SpawnAction) actions[0];

    // The action reports all arguments, including those inside the param file
    assertThat(action.getArguments())
        .containsExactly(
            "/bin/java", "-Xverify:none", "-jvmarg", "-cp", "pkg/exe.jar", "MyMainClass", "-X")
        .inOrder();

    Spawn spawn =
        action.getSpawn(
            (artifact, outputs) -> outputs.add(artifact), ImmutableMap.of(), ImmutableMap.of());
    String paramFileName = output.getExecPathString() + "-0.params";
    // The spawn's primary arguments should reference the param file
    assertThat(spawn.getArguments())
        .containsExactly(
            "/bin/java",
            "-Xverify:none",
            "-jvmarg",
            "-cp",
            "pkg/exe.jar",
            "MyMainClass",
            "@" + paramFileName)
        .inOrder();

    // Asserts that the inputs contain the param file virtual input
    Optional<? extends ActionInput> input =
        spawn.getInputFiles().toList().stream()
            .filter(i -> i instanceof VirtualActionInput)
            .findFirst();
    assertThat(input.isPresent()).isTrue();
    VirtualActionInput paramFile = (VirtualActionInput) input.get();
    assertThat(paramFile.getBytes().toString(ISO_8859_1).trim()).isEqualTo("-X");
  }

  @Test
  public void testBuilderWithExtraExecutableArguments() throws Exception {
    Action[] actions =
        builder()
            .addOutput(destinationArtifact)
            .setJavaExecutable(
                scratch.file("/bin/java").asFragment(),
                jarArtifact,
                "MyMainClass",
                asList("-jvmarg"))
            .addExecutableArguments("execArg1", "execArg2")
            .addCommandLine(CustomCommandLine.builder().add("arg1").build())
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertThat(action.getArguments())
        .containsExactly(
            "/bin/java",
            "-Xverify:none",
            "-jvmarg",
            "-cp",
            "pkg/exe.jar",
            "MyMainClass",
            "execArg1",
            "execArg2",
            "arg1");
  }

  @Test
  public void testMultipleCommandLines() throws Exception {
    Artifact input = getSourceArtifact("input");
    Artifact output = getBinArtifactWithNoOwner("output");
    Action[] actions =
        builder()
            .addInput(input)
            .addOutput(output)
            .setExecutable(scratch.file("/bin/xxx").asFragment())
            .addCommandLine(CommandLine.of(ImmutableList.of("arg1")))
            .addCommandLine(CommandLine.of(ImmutableList.of("arg2")))
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    SpawnAction action = (SpawnAction) actions[0];
    assertThat(action.getArguments()).containsExactly("/bin/xxx", "arg1", "arg2");
  }

  @Test
  public void testGetArgumentsWithParameterFiles() throws Exception {
    useConfiguration("--min_param_file_size=0", "--nodefer_param_files");
    Artifact input = getSourceArtifact("input");
    Artifact output = getBinArtifactWithNoOwner("output");
    Action[] actions =
        builder()
            .addInput(input)
            .addOutput(output)
            .setExecutable(scratch.file("/bin/xxx").asFragment())
            .addCommandLine(
                CommandLine.of(ImmutableList.of("arg1")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
            .addCommandLine(
                CommandLine.of(ImmutableList.of("arg2")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    SpawnAction action = (SpawnAction) actions[0];
    // getArguments returns all arguments, regardless whether some go in parameter files or not
    assertThat(action.getArguments()).containsExactly("/bin/xxx", "arg1", "arg2");
  }

  @Test
  public void testExtraActionInfo() throws Exception {
    SpawnAction action = createCopyFromWelcomeToDestination(ImmutableMap.<String, String>of());
    ExtraActionInfo info = action.getExtraActionInfo(actionKeyContext).build();
    assertThat(info.getMnemonic()).isEqualTo("Dummy");

    SpawnInfo spawnInfo = info.getExtension(SpawnInfo.spawnInfo);
    assertThat(info.hasExtension(SpawnInfo.spawnInfo)).isTrue();

    assertThat(spawnInfo.getArgumentList())
        .containsExactlyElementsIn(action.getArguments());

    Iterable<String> inputPaths = Artifact.asExecPaths(action.getInputs());
    Iterable<String> outputPaths = Artifact.asExecPaths(action.getOutputs());

    assertThat(spawnInfo.getInputFileList()).containsExactlyElementsIn(inputPaths);
    assertThat(spawnInfo.getOutputFileList()).containsExactlyElementsIn(outputPaths);
    Map<String, String> environment = action.getIncompleteEnvironmentForTesting();
    assertThat(spawnInfo.getVariableCount()).isEqualTo(environment.size());

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

    SpawnInfo spawnInfo =
        createCopyFromWelcomeToDestination(env)
            .getExtraActionInfo(actionKeyContext)
            .build()
            .getExtension(SpawnInfo.spawnInfo);
    assertThat(env).hasSize(spawnInfo.getVariableCount());
    for (EnvironmentVariable variable : spawnInfo.getVariableList()) {
      assertThat(env).containsEntry(variable.getName(), variable.getValue());
    }
  }

  @Test
  public void testInputManifestsRemovedIfSupplied() throws Exception {
    Artifact manifest = getSourceArtifact("MANIFEST");
    Action[] actions =
        builder()
            .addInput(manifest)
            .addRunfilesSupplier(
                new RunfilesSupplierImpl(
                    PathFragment.create("destination"),
                    Runfiles.EMPTY,
                    manifest,
                    /* buildRunfileLinks= */ false,
                    /* runfileLinksEnabled= */ false))
            .addOutput(getBinArtifactWithNoOwner("output"))
            .setExecutable(scratch.file("/bin/xxx").asFragment())
            .setProgressMessage("Test")
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    List<String> inputFiles = actionInputsToPaths(action.getSpawn().getInputFiles());
    assertThat(inputFiles).isEmpty();
  }

  private enum KeyAttributes {
    EXECUTABLE_PATH,
    EXECUTABLE,
    MNEMONIC,
    RUNFILES_SUPPLIER,
    RUNFILES_SUPPLIER_PATH,
    ENVIRONMENT
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact artifactA = getSourceArtifact("a");
    final Artifact artifactB = getSourceArtifact("b");

    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) {
            SpawnAction.Builder builder = builder();
            builder.addOutput(destinationArtifact);

            PathFragment executable =
                attributesToFlip.contains(KeyAttributes.EXECUTABLE_PATH)
                    ? artifactA.getExecPath()
                    : artifactB.getExecPath();
            if (attributesToFlip.contains(KeyAttributes.EXECUTABLE)) {
              builder.setExecutable(executable);
            } else {
              builder.setJavaExecutable(
                  executable, jarArtifact, "Main", ImmutableList.<String>of());
            }

            builder.setMnemonic(attributesToFlip.contains(KeyAttributes.MNEMONIC) ? "a" : "b");

            if (attributesToFlip.contains(KeyAttributes.RUNFILES_SUPPLIER)) {
              builder.addRunfilesSupplier(runfilesSupplier(artifactA, PathFragment.create("a")));
            } else {
              builder.addRunfilesSupplier(runfilesSupplier(artifactB, PathFragment.create("a")));
            }

            if (attributesToFlip.contains(KeyAttributes.RUNFILES_SUPPLIER_PATH)) {
              builder.addRunfilesSupplier(runfilesSupplier(artifactA, PathFragment.create("aa")));
            } else {
              builder.addRunfilesSupplier(runfilesSupplier(artifactA, PathFragment.create("ab")));
            }

            Map<String, String> env = new HashMap<>();
            if (attributesToFlip.contains(KeyAttributes.ENVIRONMENT)) {
              env.put("foo", "bar");
            }
            builder.setEnvironment(env);

            Action[] actions =
                builder.build(
                    ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
            collectingAnalysisEnvironment.registerAction(actions);
            return actions[0];
          }
        },
        actionKeyContext);
  }

  @Test
  public void testMnemonicMustNotContainSpaces() {
    SpawnAction.Builder builder = builder();
    assertThrows(IllegalArgumentException.class, () -> builder.setMnemonic("contains space"));
    assertThrows(IllegalArgumentException.class, () -> builder.setMnemonic("contains\nnewline"));
    assertThrows(IllegalArgumentException.class, () -> builder.setMnemonic("contains/slash"));
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
        "MyInfo = provider()",
        "def _aspect_impl(target, ctx):",
        "  f = ctx.actions.declare_file('foo.txt')",
        "  ctx.actions.run_shell(outputs = [f], command = 'echo foo > \"$1\"')",
        "  return MyInfo(output=f)",
        "def _rule_impl(ctx):",
        "  return DefaultInfo(",
        "      files=depset([artifact[MyInfo].output for artifact in ctx.attr.deps]))",
        "aspect1 = aspect(_aspect_impl, attr_aspects=['deps'], ",
        "    attrs = {'parameter': attr.string(values = ['param_value'])})",
        "testrule = rule(_rule_impl, attrs = { ",
        "    'deps' : attr.label_list(aspects = [aspect1]), ",
        "    'parameter': attr.string(default='param_value') })");

    update(
        ImmutableList.of("//a:a"),
        /* keepGoing= */ false,
        /* loadingPhaseThreads= */ 1,
        /* doAnalysis= */ true,
        new EventBus());

    Artifact artifact = getFilesToBuild(getConfiguredTarget("//a:a")).getSingleton();
    ExtraActionInfo.Builder extraActionInfo =
        getGeneratingAction(artifact).getExtraActionInfo(actionKeyContext);
    assertThat(extraActionInfo.getAspectName()).isEqualTo("//a:def.bzl%aspect1");
    assertThat(extraActionInfo.getAspectParametersMap())
        .containsExactly(
            "parameter", ExtraActionInfo.StringList.newBuilder().addValue("param_value").build());
  }

  private SpawnAction createWorkerSupportSpawn(Map<String, String> executionInfoVariables)
      throws Exception {
    Artifact input = getSourceArtifact("input");
    Artifact output = getBinArtifactWithNoOwner("output");
    Action[] actions =
        builder()
            .addInput(input)
            .addOutput(output)
            .setExecutionInfo(executionInfoVariables)
            .setExecutable(scratch.file("/bin/xxx").asFragment())
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    return (SpawnAction) actions[0];
  }

  @Test
  public void testWorkerSupport() throws Exception {
    SpawnAction workerSupportSpawn =
        createWorkerSupportSpawn(ImmutableMap.<String, String>of("supports-workers", "1"));
    assertThat(Spawns.supportsWorkers(workerSupportSpawn.getSpawn())).isEqualTo(true);
  }

  @Test
  public void testMultiplexWorkerSupport() throws Exception {
    SpawnAction multiplexWorkerSupportSpawn =
        createWorkerSupportSpawn(
            ImmutableMap.<String, String>of("supports-multiplex-workers", "1"));
    assertThat(Spawns.supportsMultiplexWorkers(multiplexWorkerSupportSpawn.getSpawn()))
        .isEqualTo(true);
  }

  private static RunfilesSupplier runfilesSupplier(Artifact manifest, PathFragment dir) {
    return new RunfilesSupplierImpl(
        dir,
        Runfiles.EMPTY,
        manifest,
        /* buildRunfileLinks= */ false,
        /* runfileLinksEnabled= */ false);
  }
}
