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
package com.google.devtools.build.lib.skyframe;

import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Collection;
import java.util.Set;

/**
 * Tests that the data passed from the application to the Builder is passed
 * down to each Action executed.
 */
@RunWith(JUnit4.class)
public class ActionDataTest extends TimestampBuilderTestCase {

  @Test
  public void testArgumentToBuildArtifactsIsPassedDownToAction() throws Exception {

    class MyAction extends AbstractAction {

      Object executor = null;

      public MyAction(Collection<Artifact> outputs) {
        super(ActionsTestUtil.NULL_ACTION_OWNER, ImmutableList.<Artifact>of(), outputs);
      }

      @Override
      public void execute(ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException {
        this.executor = actionExecutionContext.getExecutor();
        try {
          FileSystemUtils.createEmptyFile(getPrimaryOutput().getPath());
        } catch (IOException e) {
          throw new ActionExecutionException("failed: ", e, this, false);
        }
      }

      @Override
      public ResourceSet estimateResourceConsumption(Executor executor) {
        return ResourceSet.ZERO;
      }

      @Override
      protected String computeKey() {
        return "MyAction";
      }

      @Override
      public String getMnemonic() {
        return "MyAction";
      }
    }

    Artifact output = createDerivedArtifact("foo");
    Set<Artifact> outputs = Sets.newHashSet(output);

    MyAction action = new MyAction(outputs);
    registerAction(action);

    Executor executor = new DummyExecutor(scratch.dir("/"));
    amnesiacBuilder()
        .buildArtifacts(
            reporter, outputs, null, null, null, null, executor, null, /*explain=*/ false, null);
    assertSame(executor, action.executor);

    executor = new DummyExecutor(scratch.dir("/"));
    amnesiacBuilder()
        .buildArtifacts(
            reporter, outputs, null, null, null, null, executor, null, /*explain=*/ false, null);
    assertSame(executor, action.executor);
  }

  private static class InputDiscoveringAction extends AbstractAction {
    private final Collection<Artifact> discoveredInputs;

    public InputDiscoveringAction(Artifact output, Collection<Artifact> discoveredInputs) {
      super(
          ActionsTestUtil.NULL_ACTION_OWNER,
          ImmutableList.<Artifact>of(),
          ImmutableList.of(output));
      this.discoveredInputs = discoveredInputs;
    }

    @Override
    public boolean discoversInputs() {
      return true;
    }

    @Override
    public boolean inputsKnown() {
      return true;
    }

    @Override
    public Iterable<Artifact> getMandatoryInputs() {
      return ImmutableList.of();
    }

    @Override
    public Iterable<Artifact> getInputs() {
      return discoveredInputs;
    }

    @Override
    public void execute(ActionExecutionContext actionExecutionContext) {
      throw new IllegalStateException();
    }

    @Override
    public String getMnemonic() {
      return "InputDiscovering";
    }

    @Override
    protected String computeKey() {
      return "";
    }

    @Override
    public ResourceSet estimateResourceConsumption(Executor executor) {
      return ResourceSet.ZERO;
    }
  }

  @Test
  public void testActionSharabilityAndDiscoveredInputs() throws Exception {
    Artifact output =
        new Artifact(
            scratch.file("/out/output"), Root.asDerivedRoot(scratch.dir("/"), scratch.dir("/out")));
    Artifact discovered =
        new Artifact(
            scratch.file("/bin/discovered"),
            Root.asDerivedRoot(scratch.dir("/"), scratch.dir("/bin")));

    Action a = new InputDiscoveringAction(output, ImmutableList.of(discovered));
    Action b = new InputDiscoveringAction(output, ImmutableList.<Artifact>of());

    assertTrue(Actions.canBeShared(a, b));
  }
}
