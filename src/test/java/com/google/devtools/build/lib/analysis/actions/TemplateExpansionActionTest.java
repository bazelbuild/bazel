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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Tests {@link TemplateExpansionAction}.
 */
@RunWith(JUnit4.class)
public class TemplateExpansionActionTest extends FoundationTestCase {

  private static final String TEMPLATE = Joiner.on('\n').join("key=%key%", "value=%value%");
  private static final String SPECIAL_CHARS = "Š©±½_strøget";

  private Root outputRoot;
  private Artifact inputArtifact;
  private Artifact outputArtifact;
  private Path output;
  private List<Substitution> substitutions;
  private BlazeDirectories directories;
  private BinTools binTools;

  @Before
  public final void createDirectoriesAndTools() throws Exception {
    createArtifacts(TEMPLATE);

    substitutions = Lists.newArrayList();
    substitutions.add(Substitution.of("%key%", "foo"));
    substitutions.add(Substitution.of("%value%", "bar"));
    directories = new BlazeDirectories(
        scratch.resolve("/install"),
        scratch.resolve("/base"),
        scratch.resolve("/workspace"),
        TestConstants.PRODUCT_NAME);
    binTools = BinTools.empty(directories);
  }

  private void createArtifacts(String template) throws Exception {
    Root workspace = Root.asSourceRoot(scratch.dir("/workspace"));
    outputRoot = Root.asDerivedRoot(scratch.dir("/workspace"), scratch.dir("/workspace/out"));
    Path input = scratch.overwriteFile("/workspace/input.txt", StandardCharsets.UTF_8, template);
    inputArtifact = new Artifact(input, workspace);
    output = scratch.resolve("/workspace/out/destination.txt");
    outputArtifact = new Artifact(output, outputRoot);
  }

  private TemplateExpansionAction create() {
    TemplateExpansionAction result = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact, Template.forString(TEMPLATE), substitutions, false);
    return result;
  }

  @Test
  public void testInputsIsEmpty() {
    assertThat(create().getInputs()).isEmpty();
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    assertEquals(ImmutableSet.of(outputArtifact), create().getOutputs());
  }

  @Test
  public void testExpansion() throws Exception {
    Executor executor = new TestExecutorBuilder(directories, binTools).build();
    create().execute(createContext(executor));
    String content = new String(FileSystemUtils.readContentAsLatin1(output));
    String expected = Joiner.on('\n').join("key=foo", "value=bar");
    assertEquals(expected, content);
  }

  @Test
  public void testKeySameIfSame() throws Exception {
    Artifact outputArtifact2 = new Artifact(scratch.resolve("/workspace/out/destination.txt"),
        outputRoot);
    TemplateExpansionAction a = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo")), false);
    TemplateExpansionAction b = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact2, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo")), false);
    assertEquals(a.computeKey(), b.computeKey());
  }

  @Test
  public void testKeyDiffersForSubstitution() throws Exception {
    Artifact outputArtifact2 = new Artifact(scratch.resolve("/workspace/out/destination.txt"),
        outputRoot);
    TemplateExpansionAction a = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo")), false);
    TemplateExpansionAction b = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact2, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo2")), false);
    assertFalse(a.computeKey().equals(b.computeKey()));
  }

  @Test
  public void testKeyDiffersForExecutable() throws Exception {
    Artifact outputArtifact2 = new Artifact(scratch.resolve("/workspace/out/destination.txt"),
        outputRoot);
    TemplateExpansionAction a = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo")), false);
    TemplateExpansionAction b = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact2, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo")), true);
    assertFalse(a.computeKey().equals(b.computeKey()));
  }

  @Test
  public void testKeyDiffersForTemplates() throws Exception {
    Artifact outputArtifact2 = new Artifact(scratch.resolve("/workspace/out/destination.txt"),
        outputRoot);
    TemplateExpansionAction a = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact, Template.forString(TEMPLATE),
         ImmutableList.of(Substitution.of("%key%", "foo")), false);
    TemplateExpansionAction b = new TemplateExpansionAction(NULL_ACTION_OWNER,
         outputArtifact2, Template.forString(TEMPLATE + " "),
         ImmutableList.of(Substitution.of("%key%", "foo")), false);
    assertFalse(a.computeKey().equals(b.computeKey()));
  }

  private TemplateExpansionAction createWithArtifact() {
    return createWithArtifact(substitutions);
  }

  private TemplateExpansionAction createWithArtifact(List<Substitution> substitutions) {
    TemplateExpansionAction result = new TemplateExpansionAction(
        NULL_ACTION_OWNER, inputArtifact, outputArtifact, substitutions, false);
    return result;
  }

  private ActionExecutionContext createContext(Executor executor) {
    return new ActionExecutionContext(executor, null, null, new FileOutErr(), null);
  }

  private void executeTemplateExpansion(String expected) throws Exception {
    executeTemplateExpansion(expected, substitutions);
  }

  private void executeTemplateExpansion(String expected, List<Substitution> substitutions)
      throws Exception {
    Executor executor = new TestExecutorBuilder(directories, binTools).build();
    createWithArtifact(substitutions).execute(createContext(executor));
    String actual = FileSystemUtils.readContent(output, StandardCharsets.UTF_8);
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  public void testArtifactTemplateHasInput() {
    assertEquals(ImmutableList.of(inputArtifact), createWithArtifact().getInputs());
  }

  @Test
  public void testArtifactTemplateHasOutput() {
    assertEquals(ImmutableSet.of(outputArtifact), createWithArtifact().getOutputs());
  }

  @Test
  public void testArtifactTemplateExpansion() throws Exception {
    // The trailing "" is needed because scratch.overwriteFile implicitly appends "\n".
    String expected = Joiner.on('\n').join("key=foo", "value=bar", "");
    executeTemplateExpansion(expected);
  }

  @Test
  public void testWithSpecialCharacters() throws Exception {
    // We have to overwrite the artifacts since we need our template in "inputs"
    createArtifacts(SPECIAL_CHARS + "%key%");

    // scratch.overwriteFile appends a newline, so we need an additional \n here
    String expected = String.format("%s%s\n", SPECIAL_CHARS, SPECIAL_CHARS);

    executeTemplateExpansion(expected, ImmutableList.of(Substitution.of("%key%", SPECIAL_CHARS)));
  }
}
