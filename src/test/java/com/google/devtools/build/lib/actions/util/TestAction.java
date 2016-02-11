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
package com.google.devtools.build.lib.actions.util;

import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileSystemUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;

/**
 * A dummy action for testing.  Its execution runs the specified
 * Runnable or Callable, which is defined by the test case,
 * and touches all the output files.
 */
public class TestAction extends AbstractAction {

  public static final Runnable NO_EFFECT = new Runnable() { @Override public void run() {} };

  private static final ResourceSet RESOURCES =
      ResourceSet.createWithRamCpuIo(/*memoryMb=*/1.0, /*cpu=*/0.1, /*io=*/0.0);

  protected final Callable<Void> effect;

  /** Use this constructor if the effect can't throw exceptions. */
  public TestAction(Runnable effect,
             Collection<Artifact> inputs,
             Collection<Artifact> outputs) {
    super(NULL_ACTION_OWNER, inputs, outputs);
    this.effect = Executors.callable(effect, null);
  }

  /**
   * Use this constructor if the effect can throw exceptions.
   * Any checked exception thrown will be repackaged as an
   * ActionExecutionException.
   */
  public TestAction(Callable<Void> effect,
             Collection<Artifact> inputs,
             Collection<Artifact> outputs) {
    super(NULL_ACTION_OWNER, inputs, outputs);
    this.effect = effect;
  }

  @Override
  public Collection<Artifact> getMandatoryInputs() {
    List<Artifact> mandatoryInputs = new ArrayList<>();
    for (Artifact input : getInputs()) {
      if (!input.getExecPath().getBaseName().endsWith(".optional")) {
        mandatoryInputs.add(input);
      }
    }
    return mandatoryInputs;
  }

  @Override
  public boolean discoversInputs() {
    for (Artifact input : getInputs()) {
      if (!input.getExecPath().getBaseName().endsWith(".optional")) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Collection<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
    Preconditions.checkState(discoversInputs(), this);
    return ImmutableList.of();
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    for (Artifact artifact : getInputs()) {
      // Do not check *.optional artifacts - artifacts with such extension are
      // used by tests to specify artifacts that may or may not be missing.
      // This is used, e.g., to test Blaze behavior when action has missing
      // input artifacts but still is successfully executed.
      if (!artifact.getPath().exists() &&
          !artifact.getExecPath().getBaseName().endsWith(".optional")) {
        throw new IllegalStateException("action's input file does not exist: "
            + artifact.getPath());
      }
    }

    try {
      effect.call();
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Exception e) {
      throw new ActionExecutionException("TestAction failed due to exception",
                                         e, this, false);
    }

    try {
      for (Artifact artifact: getOutputs()) {
        FileSystemUtils.touchFile(artifact.getPath());
      }
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  @Override
  protected String computeKey() {
    List<String> outputsList = new ArrayList<>();
    for (Artifact output : getOutputs()) {
      outputsList.add(output.getPath().getPathString());
    }
    // This could use a functional iterable and avoid creating a list
    return "test " + StringUtilities.combineKeys(outputsList);
  }

  @Override
  public String getMnemonic() { return "Test"; }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return RESOURCES;
  }


  /** No-op action that has exactly one output, and can be a middleman action. */
  public static class DummyAction extends TestAction {
    private final MiddlemanType type;

    public DummyAction(Collection<Artifact> inputs, Artifact output, MiddlemanType type) {
      super(NO_EFFECT, inputs, ImmutableList.of(output));
      this.type = type;
    }

    public DummyAction(Collection<Artifact> inputs, Artifact output) {
      this(inputs, output, MiddlemanType.NORMAL);
    }

    @Override
    public MiddlemanType getActionType() {
      return type;
    }
  }
}
