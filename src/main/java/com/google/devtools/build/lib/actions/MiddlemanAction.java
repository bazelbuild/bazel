// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.util.Fingerprint;

/**
 * An action that depends on a set of inputs and creates a single output file whenever it runs. This
 * is useful for bundling up a bunch of dependencies that are shared between individual targets in
 * the action graph; for example generated header files.
 */
@Immutable
@AutoCodec
public final class MiddlemanAction extends AbstractAction {
  public static final String MIDDLEMAN_MNEMONIC = "Middleman";
  private final String description;
  private final MiddlemanType middlemanType;

  @VisibleForSerialization
  @AutoCodec.Instantiator
  MiddlemanAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      ImmutableSet<Artifact> outputs,
      String description,
      MiddlemanType middlemanType) {
    super(owner, inputs, outputs);
    Preconditions.checkNotNull(middlemanType);
    Preconditions.checkArgument(Iterables.getOnlyElement(outputs).isMiddlemanArtifact(), outputs);
    Preconditions.checkNotNull(description);
    this.description = description;
    this.middlemanType = middlemanType;
  }

  @Override
  public final ActionResult execute(ActionExecutionContext actionExecutionContext) {
    throw new IllegalStateException("MiddlemanAction should never be executed");
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    // TODO(bazel-team): Need to take middlemanType into account here.
    // Only the set of inputs matters, and the dependency checker is
    // responsible for considering those.
  }

  /**
   * Returns the type of the middleman.
   */
  @Override
  public MiddlemanType getActionType() {
    return middlemanType;
  }

  @Override
  protected String getRawProgressMessage() {
    return null; // users don't really want to know about Middlemen.
  }

  @Override
  public String prettyPrint() {
    return description + " for " + Label.print(getOwner().getLabel());
  }

  @Override
  public String getMnemonic() {
    return MIDDLEMAN_MNEMONIC;
  }

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }

  /** Creates a new middleman action. */
  public static Action create(
      ActionRegistry env,
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Artifact stampFile,
      String purpose,
      MiddlemanType middlemanType) {
    MiddlemanAction action =
        new MiddlemanAction(owner, inputs, ImmutableSet.of(stampFile), purpose, middlemanType);
    env.registerAction(action);
    return action;
  }
}
