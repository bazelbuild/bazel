// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.actions;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MiddlemanAction;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SuppressNoBuildAttemptError;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.TargetCompleteEvent;

/**
 * This action depends on all of a target's build artifacts and reports the
 * target complete in the master log when it is built. It therefore fires as
 * soon as all artifacts in a target's subgraph have been completed, and allows
 * for reporting of target completion in real-time.
 *
 * <p>Consequently, you can depend on the output artifact of this action to
 * execute something after a target completes.
 */
public final class TargetCompletionMiddlemanAction extends MiddlemanAction
    implements SuppressNoBuildAttemptError, NotifyOnActionCacheHit {

  private static final String GUID = "6633f0ab-9685-406e-be8a-f19dc46498e6";

  // Target to report complete.
  private final ConfiguredTarget target;

  /**
   * Constructs a target completion middleman.
   *
   * @param target the configured target to report built when the action executes.
   * @param owner the owner of the action (usually derived from {@code target})
   * @param inputs input artifacts upon which this action depends.
   * @param output output artifact which this action "produces",
   *        or satisfies dependencies of.
   */
  public TargetCompletionMiddlemanAction(ConfiguredTarget target, ActionOwner owner,
                                         Iterable<Artifact> inputs,
                                         Artifact output) {
    super(owner, inputs, output, "target_completion", MiddlemanType.TARGET_COMPLETION_MIDDLEMAN);
    this.target = target;
  }

  @Override
  public String describeStrategy(Executor executor) {
    return "local";
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext) {
    actionExecutionContext.getExecutor().getEventBus().post(
        new TargetCompleteEvent(target, null, null, /* isCached */ false));
  }

  @Override
  public String getMnemonic() {
    return "TargetCompletionMiddleman";
  }

  @Override
  public String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(target.getLabel().toString());
    return f.hexDigest();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return ResourceSet.ZERO;
  }

  @Override
  protected String getRawProgressMessage() {
    return null;
  }

  @Override
  public void actionCacheHit(Executor executor) {
    // Similar to reporting of a cached test result:
    // This action is cached is if all of its inputs are cached.
    // Since the inputs are the target runfiles, that means the target is already built.
    executor.getEventBus().post(new TargetCompleteEvent(target, null, null, /* isCached */ true));
  }

  @Override
  public String toString() {
    return String.format("TargetCompletionMiddlemanAction(target: %s)", target);
  }

}
