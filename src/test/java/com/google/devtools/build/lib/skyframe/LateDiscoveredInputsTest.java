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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Timestamp builder tests for inputs that are only discovered after an action was executed. */
@RunWith(JUnit4.class)
public class LateDiscoveredInputsTest extends TimestampBuilderTestCase {

  /**
   * The action file system reads the {@link ActionInputMap} that backs the {@link
   * InputMetadataProvider} handed to the action, and it outlives the action: build events
   * referencing it are uploaded asynchronously. Since the map is only thread-compatible, inputs
   * discovered after execution must not be added to it. See
   * https://github.com/bazelbuild/bazel/issues/30683.
   */
  @Test
  public void inputsDiscoveredAfterExecution_notAddedToInputMetadataProviderOfAction()
      throws Exception {
    Artifact hello = createSourceArtifact("hello");
    hello.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");
    Artifact late = createSourceArtifact("late");
    FileSystemUtils.writeContentAsLatin1(late.getPath(), "late1");
    Artifact goodbye = createDerivedArtifact("goodbye");

    Button button = new Button();
    AtomicReference<InputMetadataProvider> inputMetadataProvider = new AtomicReference<>();
    registerAction(
        new LateInputDiscoveringAction(
            button,
            NestedSetBuilder.create(Order.STABLE_ORDER, hello),
            ImmutableSet.of(goodbye),
            late,
            inputMetadataProvider));

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // built

    assertThat(inputMetadataProvider.get().getInput(late.getExecPath())).isNull();

    // The late-discovered input is still tracked by the action cache.
    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt

    FileSystemUtils.writeContentAsLatin1(late.getPath(), "late2");

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // rebuilt
  }

  /** A {@link TestAction} that only learns about {@code late} once it has been executed. */
  private static final class LateInputDiscoveringAction extends TestAction {
    private final Artifact late;
    private final AtomicReference<InputMetadataProvider> inputMetadataProvider;

    LateInputDiscoveringAction(
        Runnable effect,
        NestedSet<Artifact> inputs,
        ImmutableSet<Artifact> outputs,
        Artifact late,
        AtomicReference<InputMetadataProvider> inputMetadataProvider) {
      super(effect, inputs, outputs);
      this.late = late;
      this.inputMetadataProvider = inputMetadataProvider;
    }

    @Override
    public boolean discoversInputs() {
      return true;
    }

    @Override
    public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException, InterruptedException {
      inputMetadataProvider.set(actionExecutionContext.getInputMetadataProvider());
      ActionResult result = super.execute(actionExecutionContext);
      updateInputs(
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(getMandatoryInputs())
              .add(late)
              .build());
      return result;
    }
  }
}
