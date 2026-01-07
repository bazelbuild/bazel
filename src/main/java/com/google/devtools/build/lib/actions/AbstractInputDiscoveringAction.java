package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import javax.annotation.Nullable;

@ThreadSafe
public abstract class AbstractInputDiscoveringAction extends AbstractAction {
  private volatile NestedSet<Artifact> discoveredInputs = null;

  protected AbstractInputDiscoveringAction(
      ActionOwner owner,
      NestedSet<Artifact> analysisTimeInputs,
      Iterable<? extends Artifact> outputs) {
    super(owner, analysisTimeInputs, outputs);
  }

  @VisibleForSerialization
  protected AbstractInputDiscoveringAction(
      ActionOwner owner, NestedSet<Artifact> analysisTimeInputs, Object rawOutputs) {
    super(owner, analysisTimeInputs, rawOutputs);
  }

  @Override
  public final boolean discoversInputs() {
    return true;
  }

  @Override
  public void updateDiscoveredInputs(NestedSet<Artifact> discoveredInputs) {
    this.discoveredInputs = discoveredInputs;
  }

  @Nullable
  @Override
  public NestedSet<Artifact> getDiscoveredInputs() {
    return discoveredInputs;
  }

  @Override
  public NestedSet<Artifact> getSchedulingDependencies() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }
}
