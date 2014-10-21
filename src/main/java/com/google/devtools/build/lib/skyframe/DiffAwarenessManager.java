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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;

import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/** Helper class to make it easier to correctly use the {@link DiffAwareness} interface. */
public final class DiffAwarenessManager {

  private final ImmutableSet<? extends DiffAwareness.Factory> diffAwarenessFactories;
  @Nullable
  private ImmutableList<Path> pathEntries = null;
  private Map<Path, DiffAwareness> currentDiffAwarenesses = Maps.newHashMap();
  private Map<Path, ProcessableModifiedFileSet> unprocessedDiffs = Maps.newHashMap();
  private final Reporter reporter;

  public DiffAwarenessManager(Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Reporter reporter) {
    this.diffAwarenessFactories = ImmutableSet.copyOf(diffAwarenessFactories);
    this.reporter = reporter;
  }

  /** Must be called at least once, and whenever --package_path changes. */
  public void setPathEntries(List<Path> pathEntries) {
    if (!Objects.equal(this.pathEntries, pathEntries)) {
      reset();
      this.pathEntries = ImmutableList.copyOf(pathEntries);
    }
  }

  /** Reset internal {@link DiffAwareness} state. */
  public void reset() {
    for (DiffAwareness diffAwareness : currentDiffAwarenesses.values()) {
      diffAwareness.close();
    }
    currentDiffAwarenesses.clear();
    unprocessedDiffs.clear();
  }

  /** A set of modified files that should be marked as processed. */
  public interface ProcessableModifiedFileSet {
    ModifiedFileSet getModifiedFileSet();

    /**
     * This should be called when the changes have been noted. Otherwise, the result from the next
     * call to {@link #getDiff} will conservatively contain the old unprocessed diff.
     */
    void markProcessed();
  }

  /** Gets the set of changed files since the last call with this path entry. */
  public ProcessableModifiedFileSet getDiff(Path pathEntry) {
    Preconditions.checkNotNull(pathEntries, "Must call setPathEntries before getDiff");
    DiffAwareness diffAwareness = maybeGetDiffAwareness(pathEntry);
    if (diffAwareness == null) {
      return BrokenProcessableModifiedFileSet.INSTANCE;
    }
    ModifiedFileSet oldDiff = ModifiedFileSet.NOTHING_MODIFIED;
    if (unprocessedDiffs.containsKey(pathEntry)) {
      oldDiff = unprocessedDiffs.get(pathEntry).getModifiedFileSet();
    }
    ModifiedFileSet newDiff;
    try {
      newDiff = diffAwareness.getDiff();
    } catch (BrokenDiffAwarenessException e) {
      currentDiffAwarenesses.remove(pathEntry);
      unprocessedDiffs.remove(pathEntry);
      reporter.handle(Event.warn(e.getMessage()));
      return BrokenProcessableModifiedFileSet.INSTANCE;
    }
    ModifiedFileSet diff = ModifiedFileSet.union(oldDiff, newDiff);
    ProcessableModifiedFileSet result = new ProcessableModifiedFileSetImpl(diff, pathEntry);
    unprocessedDiffs.put(pathEntry, result);
    return result;
  }

  /**
   * Returns the current diff awareness for the given path entry, or a fresh one if there is no
   * current one, or otherwise {@link null} if no factory could make a fresh one.
   */
  @Nullable
  private DiffAwareness maybeGetDiffAwareness(Path pathEntry) {
    DiffAwareness currentDiffAwareness = currentDiffAwarenesses.get(pathEntry);
    if (currentDiffAwareness != null) {
      return currentDiffAwareness;
    }
    for (DiffAwareness.Factory factory : diffAwarenessFactories) {
      DiffAwareness newDiffAwareness = factory.maybeCreate(pathEntry, pathEntries);
      if (newDiffAwareness != null) {
        currentDiffAwarenesses.put(pathEntry, newDiffAwareness);
        return newDiffAwareness;
      }
    }
    return null;
  }

  private class ProcessableModifiedFileSetImpl implements ProcessableModifiedFileSet {

    private final ModifiedFileSet modifiedFileSet;
    private final Path pathEntry;

    private ProcessableModifiedFileSetImpl(ModifiedFileSet modifiedFileSet, Path pathEntry) {
      this.modifiedFileSet = modifiedFileSet;
      this.pathEntry = pathEntry;
    }

    @Override
    public ModifiedFileSet getModifiedFileSet() {
      return modifiedFileSet;
    }

    @Override
    public void markProcessed() {
      ProcessableModifiedFileSet currentUnprocessedDiff = unprocessedDiffs.get(pathEntry);
      if (this == currentUnprocessedDiff) {
        // We need to check for equality here because of the following two scenarios:
        //
        // d1 = m.getDiff(p);
        // d2 = m.getDiff(p);
        // d1.markProcessed();
        //
        // d1 = m.getDiff(p);
        // m.reset();
        // d2 = m.getDiff(p);
        // d1.markProcessed();
        unprocessedDiffs.remove(pathEntry);
      }
    }
  }

  private static class BrokenProcessableModifiedFileSet implements ProcessableModifiedFileSet {

    private static final BrokenProcessableModifiedFileSet INSTANCE =
        new BrokenProcessableModifiedFileSet();

    @Override
    public ModifiedFileSet getModifiedFileSet() {
      return ModifiedFileSet.EVERYTHING_MODIFIED;
    }

    @Override
    public void markProcessed() {
    }
  }
}
