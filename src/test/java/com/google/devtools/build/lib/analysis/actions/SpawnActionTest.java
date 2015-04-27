// Copyright 2015 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertSameContents;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.Arrays.asList;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.extra.EnvironmentVariable;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tests {@link SpawnAction}.
 */
public class SpawnActionTest extends BuildViewTestCase {
  private Artifact welcomeArtifact;
  private Artifact destinationArtifact;
  private Artifact jarArtifact;
  private AnalysisTestUtil.CollectingAnalysisEnvironment collectingAnalysisEnvironment;

  private SpawnAction.Builder builder() {
    return new SpawnAction.Builder();
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();

    collectingAnalysisEnvironment = new AnalysisTestUtil.CollectingAnalysisEnvironment(
        getTestAnalysisEnvironment());
    welcomeArtifact = getSourceArtifact("pkg/welcome.txt");
    jarArtifact = getSourceArtifact("pkg/exe.jar");
    destinationArtifact = getBinArtifactWithNoOwner("dir/destination.txt");
  }

  private SpawnAction createCopyFromWelcomeToDestination() {
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
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    return (SpawnAction) actions[0];
  }

  public void testWelcomeArtifactIsInput() {
    SpawnAction copyFromWelcomeToDestination = createCopyFromWelcomeToDestination();
    Iterable<Artifact> inputs = copyFromWelcomeToDestination.getInputs();
    assertEquals(Sets.newHashSet(welcomeArtifact), Sets.newHashSet(inputs));
  }

  public void testDestinationArtifactIsOutput() {
    SpawnAction copyFromWelcomeToDestination = createCopyFromWelcomeToDestination();
    Collection<Artifact> outputs = copyFromWelcomeToDestination.getOutputs();
    assertEquals(Sets.newHashSet(destinationArtifact), Sets.newHashSet(outputs));
  }

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
    assertSameContents(asList(input), action.getInputs());
    assertSameContents(asList(output), action.getOutputs());
    assertEquals(AbstractAction.DEFAULT_RESOURCE_SET, action.getSpawn().getLocalResources());
    assertSameContents(asList("/bin/xxx"), action.getArguments());
    assertEquals("Test", action.getProgressMessage());
  }

  public void testBuilderWithExecutable() throws Exception {
    Action[] actions = builder()
        .setExecutable(welcomeArtifact)
        .addOutput(destinationArtifact)
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    assertSameContents(asList(welcomeArtifact.getExecPath().getPathString()),
        action.getArguments());
  }

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
    assertContainsSublist(actionInputsToPaths(action.getSpawn().getInputFiles()),
        "pkg/exe.jar");
  }

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
    assertContainsSublist(actionInputsToPaths(action.getSpawn().getInputFiles()),
        "pkg/exe.jar");
  }

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

  public void testExtraActionInfo() throws Exception {
    SpawnAction copyFromWelcomeToDestination = createCopyFromWelcomeToDestination();
    ExtraActionInfo.Builder builder = copyFromWelcomeToDestination.getExtraActionInfo();
    ExtraActionInfo info = builder.build();
    assertEquals("Dummy", info.getMnemonic());

    SpawnInfo spawnInfo = info.getExtension(SpawnInfo.spawnInfo);
    assertNotNull(spawnInfo);

    assertSameContents(copyFromWelcomeToDestination.getArguments(), spawnInfo.getArgumentList());

    Iterable<String> inputPaths = Artifact.toExecPaths(
        copyFromWelcomeToDestination.getInputs());
    Iterable<String> outputPaths = Artifact.toExecPaths(
        copyFromWelcomeToDestination.getOutputs());

    assertSameContents(inputPaths, spawnInfo.getInputFileList());
    assertSameContents(outputPaths, spawnInfo.getOutputFileList());
    Map<String, String> environment = copyFromWelcomeToDestination.getEnvironment();
    assertEquals(environment.size(), spawnInfo.getVariableCount());

    for (EnvironmentVariable variable : spawnInfo.getVariableList()) {
      assertThat(environment).containsEntry(variable.getName(), variable.getValue());
    }
  }

  public void testInputManifest() throws Exception {
    Artifact manifest = getSourceArtifact("MANIFEST");
    Action[] actions = builder()
        .addInput(manifest)
        .addInputManifest(manifest, new PathFragment("/destination/"))
        .addOutput(getBinArtifactWithNoOwner("output"))
        .setExecutable(scratch.file("/bin/xxx").asFragment())
        .setProgressMessage("Test")
        .build(ActionsTestUtil.NULL_ACTION_OWNER, collectingAnalysisEnvironment, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    SpawnAction action = (SpawnAction) actions[0];
    List<String> inputFiles = actionInputsToPaths(action.getSpawn().getInputFiles());
    assertThat(inputFiles).isEmpty();
  }

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
          builder.addInputManifest(artifactA, new PathFragment("a"));
        } else {
          builder.addInputManifest(artifactB, new PathFragment("a"));
        }

        if ((i & 16) == 0) {
          builder.addInputManifest(artifactA, new PathFragment("aa"));
        } else {
          builder.addInputManifest(artifactA, new PathFragment("ab"));
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
}
