// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.collect.ImmutableSet;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Convenience class for {@link NodeEntry} implementations that delegate many operations. */
public abstract class DelegatingNodeEntry implements NodeEntry {
  protected abstract NodeEntry getDelegate();

  @Override
  public SkyValue getValue() throws InterruptedException {
    return getDelegate().getValue();
  }

  @Override
  public SkyValue getValueMaybeWithMetadata() throws InterruptedException {
    return getDelegate().getValueMaybeWithMetadata();
  }

  @Override
  public SkyValue toValue() throws InterruptedException {
    return getDelegate().toValue();
  }

  @Nullable
  @Override
  public ErrorInfo getErrorInfo() throws InterruptedException {
    return getDelegate().getErrorInfo();
  }

  @Override
  public Set<SkyKey> getInProgressReverseDeps() {
    return getDelegate().getInProgressReverseDeps();
  }

  @Override
  public Set<SkyKey> setValue(
      SkyValue value, Version graphVersion, @Nullable Version maxTransitiveSourceVersion)
      throws InterruptedException {
    return getDelegate().setValue(value, graphVersion, maxTransitiveSourceVersion);
  }

  @Override
  public DependencyState addReverseDepAndCheckIfDone(@Nullable SkyKey reverseDep)
      throws InterruptedException {
    return getDelegate().addReverseDepAndCheckIfDone(reverseDep);
  }

  @Override
  public DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep)
      throws InterruptedException {
    return getDelegate().checkIfDoneForDirtyReverseDep(reverseDep);
  }

  @Override
  public boolean signalDep(Version childVersion, @Nullable SkyKey childForDebugging) {
    return getDelegate().signalDep(childVersion, childForDebugging);
  }

  @Override
  public NodeValueAndRdepsToSignal markClean() throws InterruptedException {
    return getDelegate().markClean();
  }

  @Override
  public void forceRebuild() {
    getDelegate().forceRebuild();
  }

  @Override
  public Version getVersion() {
    return getDelegate().getVersion();
  }

  @Override
  public Version getMaxTransitiveSourceVersion() {
    return getDelegate().getMaxTransitiveSourceVersion();
  }

  @Override
  public DirtyState getDirtyState() {
    return getDelegate().getDirtyState();
  }

  @Override
  public List<SkyKey> getNextDirtyDirectDeps() throws InterruptedException {
    return getDelegate().getNextDirtyDirectDeps();
  }

  @Override
  public Iterable<SkyKey> getAllDirectDepsForIncompleteNode() throws InterruptedException {
    return getDelegate().getAllDirectDepsForIncompleteNode();
  }

  @Override
  public ImmutableSet<SkyKey> getAllRemainingDirtyDirectDeps() throws InterruptedException {
    return getDelegate().getAllRemainingDirtyDirectDeps();
  }

  @Override
  public Collection<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    return getDelegate().getAllReverseDepsForNodeBeingDeleted();
  }

  @Override
  public void markRebuilding() {
    getDelegate().markRebuilding();
  }

  @Override
  public GroupedDeps getTemporaryDirectDeps() {
    return getDelegate().getTemporaryDirectDeps();
  }

  @Override
  public boolean noDepsLastBuild() {
    return getDelegate().noDepsLastBuild();
  }

  @Override
  public void removeUnfinishedDeps(Set<SkyKey> unfinishedDeps) {
    getDelegate().removeUnfinishedDeps(unfinishedDeps);
  }

  @Override
  public void resetForRestartFromScratch() {
    getDelegate().resetForRestartFromScratch();
  }

  @Override
  public void addSingletonTemporaryDirectDep(SkyKey dep) {
    getDelegate().addSingletonTemporaryDirectDep(dep);
  }

  @Override
  public void addTemporaryDirectDepGroup(List<SkyKey> group) {
    getDelegate().addTemporaryDirectDepGroup(group);
  }

  @Override
  public void addTemporaryDirectDepsInGroups(Set<SkyKey> deps, List<Integer> groupSizes) {
    getDelegate().addTemporaryDirectDepsInGroups(deps, groupSizes);
  }

  @Override
  public boolean isReadyToEvaluate() {
    return getDelegate().isReadyToEvaluate();
  }

  @Override
  public boolean hasUnsignaledDeps() {
    return getDelegate().hasUnsignaledDeps();
  }

  @Override
  public boolean isDone() {
    return getDelegate().isDone();
  }

  @Override
  public Iterable<SkyKey> getDirectDeps() throws InterruptedException {
    return getDelegate().getDirectDeps();
  }

  @Override
  public boolean hasAtLeastOneDep() throws InterruptedException {
    return getDelegate().hasAtLeastOneDep();
  }

  @Override
  public void removeReverseDep(SkyKey reverseDep) throws InterruptedException {
    getDelegate().removeReverseDep(reverseDep);
  }

  @Override
  public void removeReverseDepsFromDoneEntryDueToDeletion(Set<SkyKey> deletedKeys) {
    getDelegate().removeReverseDepsFromDoneEntryDueToDeletion(deletedKeys);
  }

  @Override
  public void removeInProgressReverseDep(SkyKey reverseDep) {
    getDelegate().removeInProgressReverseDep(reverseDep);
  }

  @Override
  public Collection<SkyKey> getReverseDepsForDoneEntry() throws InterruptedException {
    return getDelegate().getReverseDepsForDoneEntry();
  }

  @Override
  public boolean isDirty() {
    return getDelegate().isDirty();
  }

  @Override
  public boolean isChanged() {
    return getDelegate().isChanged();
  }

  @Override
  @Nullable
  public MarkedDirtyResult markDirty(DirtyType dirtyType) throws InterruptedException {
    return getDelegate().markDirty(dirtyType);
  }

  @Override
  public void addExternalDep() {
    getDelegate().addExternalDep();
  }

  @Override
  public int getPriority() {
    return getDelegate().getPriority();
  }

  @Override
  public int depth() {
    return getDelegate().depth();
  }

  @Override
  public void updateDepthIfGreater(int proposedDepth) {
    getDelegate().updateDepthIfGreater(proposedDepth);
  }

  @Override
  public void incrementEvaluationCount() {
    getDelegate().incrementEvaluationCount();
  }
}
