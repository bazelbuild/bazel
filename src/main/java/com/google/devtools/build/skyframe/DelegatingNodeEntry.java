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
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Convenience class for {@link NodeEntry} implementations that delegate many operations. */
public abstract class DelegatingNodeEntry implements NodeEntry {
  protected abstract NodeEntry getDelegate();

  protected ThinNodeEntry getThinDelegate() {
    return getDelegate();
  }

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
  public Set<SkyKey> setValue(SkyValue value, Version version) throws InterruptedException {
    return getDelegate().setValue(value, version);
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
  public Iterable<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    return getDelegate().getAllReverseDepsForNodeBeingDeleted();
  }

  @Override
  public void markRebuilding() {
    getDelegate().markRebuilding();
  }

  @Override
  public GroupedList<SkyKey> getTemporaryDirectDeps() {
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
  public Set<SkyKey> addTemporaryDirectDeps(GroupedListHelper<SkyKey> helper) {
    return getDelegate().addTemporaryDirectDeps(helper);
  }

  @Override
  public boolean isReady() {
    return getDelegate().isReady();
  }

  @Override
  public boolean isDone() {
    return getThinDelegate().isDone();
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
  public void removeInProgressReverseDep(SkyKey reverseDep) {
    getDelegate().removeInProgressReverseDep(reverseDep);
  }

  @Override
  public Iterable<SkyKey> getReverseDepsForDoneEntry() throws InterruptedException {
    return getDelegate().getReverseDepsForDoneEntry();
  }

  @Override
  public boolean isDirty() {
    return getThinDelegate().isDirty();
  }

  @Override
  public boolean isChanged() {
    return getThinDelegate().isChanged();
  }

  @Override
  @Nullable
  public MarkedDirtyResult markDirty(DirtyType dirtyType) throws InterruptedException {
    return getThinDelegate().markDirty(dirtyType);
  }

  @Override
  public void addTemporaryDirectDepsGroupToDirtyEntry(List<SkyKey> group) {
    getDelegate().addTemporaryDirectDepsGroupToDirtyEntry(group);
  }

  @Override
  public void addExternalDep() {
    getDelegate().addExternalDep();
  }
}
