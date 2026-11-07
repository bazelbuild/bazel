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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputService.RewoundActionSynchronizer;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Timestamp builder tests for the inputs that {@link SkyframeActionExecutor} passes to the {@link
 * RewoundActionSynchronizer}.
 */
@RunWith(JUnit4.class)
public class RewoundActionSynchronizationTest extends TimestampBuilderTestCase {

  /**
   * Actions such as {@code CppCompileAction} read the inputs they discover before execution, but
   * only add them to {@link Action#getInputs} afterwards. The synchronizer must still learn about
   * them so that their producers aren't rewound while the action reads them. See
   * https://github.com/bazelbuild/bazel/issues/32488.
   */
  @Test
  public void actionExecution_receivesInputsDiscoveredBeforeExecution() throws Exception {
    Artifact mandatory = createSourceArtifact("mandatory");
    mandatory.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(mandatory.getPath(), "mandatory");
    Artifact discovered = createSourceArtifact("discovered");
    FileSystemUtils.writeContentAsLatin1(discovered.getPath(), "discovered");
    Artifact output = createDerivedArtifact("output");

    Button button = new Button();
    registerAction(
        new InputDiscoveringAction(
            button,
            NestedSetBuilder.create(Order.STABLE_ORDER, mandatory),
            ImmutableSet.of(output),
            discovered));
    AtomicReference<ImmutableList<Artifact>> inputsOnEntry = new AtomicReference<>();
    AtomicReference<ImmutableList<Artifact>> discoveredInputsOnEntry = new AtomicReference<>();
    rewoundActionSynchronizer =
        new RewoundActionSynchronizer() {
          @Override
          public SilentCloseable enterActionPreparation(Action action, boolean wasRewound) {
            return () -> {};
          }

          @Override
          public SilentCloseable enterActionExecution(
              Action action,
              boolean wasRewound,
              NestedSet<Artifact> discoveredInputs,
              InputMetadataProvider metadataProvider) {
            inputsOnEntry.set(action.getInputs().toList());
            discoveredInputsOnEntry.set(discoveredInputs.toList());
            return () -> {};
          }

          @Override
          public SilentCloseable enterInputDiscovery(
              Action action,
              NestedSet<Artifact> evaluatedInputs,
              InputMetadataProvider metadataProvider) {
            return () -> {};
          }
        };

    buildArtifacts(cachingBuilder(), output);

    assertThat(button.pressed).isTrue();
    assertThat(inputsOnEntry.get()).containsExactly(mandatory);
    assertThat(discoveredInputsOnEntry.get()).containsExactly(discovered);
  }

  /**
   * A {@link TestAction} that, just like {@code CppCompileAction}, discovers {@code discovered}
   * before execution, but only adds it to its inputs after execution.
   */
  private static final class InputDiscoveringAction extends TestAction {
    private final Artifact discovered;

    InputDiscoveringAction(
        Runnable effect,
        NestedSet<Artifact> inputs,
        ImmutableSet<Artifact> outputs,
        Artifact discovered) {
      super(effect, inputs, outputs);
      this.discovered = discovered;
    }

    @Override
    public boolean discoversInputs() {
      return true;
    }

    @Override
    public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
      return NestedSetBuilder.create(Order.STABLE_ORDER, discovered);
    }

    @Override
    public ActionResult execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException, InterruptedException {
      ActionResult result = super.execute(actionExecutionContext);
      updateInputs(
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(getMandatoryInputs())
              .add(discovered)
              .build());
      return result;
    }
  }
}
