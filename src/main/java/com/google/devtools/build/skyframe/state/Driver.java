// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe.state;

import static com.google.common.base.MoreObjects.toStringHelper;

import com.google.common.collect.Lists;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayDeque;
import java.util.ArrayList;

/**
 * This class drives a {@link StateMachine} instance.
 *
 * <p>One recommended usage pattern for this class is to embed an instance within a top level {@link
 * StateMachine} implementation and from there, re-export the {@link #drive} method. Then the
 * results from the {@link StateMachine} will be readily retrievable from the {@link SkyFunction}
 * state.
 */
// TODO(shahan); this is incompatible with partial re-evaluation, which causes the assumption that
// an unavailable previously requested dependency implies an error to no longer be true. This can be
// fixed by integrating with the partial re-evaluation mailbox.
public final class Driver {
  private final ArrayDeque<TaskTreeNode> ready = new ArrayDeque<>();

  /** A Skyframe lookup has not yet been made for the key. */
  private final ArrayList<Lookup> newlyAdded = new ArrayList<>();

  /** A Skyframe lookup has already been made for the key, but it was not available. */
  private final ArrayList<Lookup> pending = new ArrayList<>();

  public Driver(StateMachine root) {
    ready.addLast(new TaskTreeNode(this, /* parent= */ null, root));
  }

  /**
   * Drives the machine as far as it can go without a Skyframe restart.
   *
   * @return true if execution is complete, false if a restart is needed.
   */
  public boolean drive(LookupEnvironment env) throws InterruptedException {
    if (!pending.isEmpty()) {
      // If pending is non-empty, it means there was a Skyframe restart. Either everything that was
      // pending is available now or we are in error bubbling. In the latter case, this method
      // returns early when it either observes an error or missing value.
      //
      // NB: this assumption does not hold under partial re-evaluation and likewise the inference
      // below about unavailable values being errors.
      SkyframeLookupResult result = env.getLookupHandleForPreviouslyRequestedDeps();
      boolean hasExceptionOrMissingValue = false;
      for (var lookup : pending) {
        if (!result.queryDep(lookup.key(), lookup)) {
          // Since the key was previously requested, unavailability here could be an unhandled
          // exception or a missing value during error bubbling. It's not possible to determine
          // which here. Requests the key to ensure that if it is an error, the environment
          // instance knows that the failure is due to child error.
          var unusedNull = env.getValue(lookup.key());
          // Failing fast here would make behavior dependent on element ordering and possibly miss
          // errors in error bubbling, so instead, flags the exception and fails after all lookups
          // have been processed.
          hasExceptionOrMissingValue = true;
        }
      }
      if (hasExceptionOrMissingValue) {
        return false;
      }
      pending.clear();
    }

    while (true) {
      // Runs all ready tasks, including ones that may be added during execution.
      TaskTreeNode next;
      while ((next = ready.poll()) != null) {
        next.run();
      }

      // No more tasks are ready. If there are no newly added lookups, it isn't possible to drive
      // this machine any further.
      if (newlyAdded.isEmpty()) {
        return pending.isEmpty(); // If there are no pending lookups, the machine is done.
      }

      // Performs lookups for any newly added keys.
      if (newlyAdded.size() == 1) { // Uses a lower overhead lookup for the unary case.
        var onlyLookup = newlyAdded.get(0);
        if (!onlyLookup.doLookup(env)) {
          pending.add(onlyLookup);
        }
      } else {
        SkyframeLookupResult result =
            env.getValuesAndExceptions(Lists.transform(newlyAdded, Lookup::key));
        for (var lookup : newlyAdded) {
          if (!result.queryDep(lookup.key(), lookup)) {
            pending.add(lookup); // Unhandled exceptions also end up here.
          }
        }
      }
      newlyAdded.clear(); // Every entry is either done or has moved to pending.
    }
  }

  void addReady(TaskTreeNode task) {
    ready.addLast(task);
  }

  /**
   * Adds a dependency to look up.
   *
   * <p>The callback could be deferred until the next Skyframe restart if the queried key is not
   * immediately available.
   */
  void addLookup(Lookup lookup) {
    newlyAdded.add(lookup);
  }

  @Override
  public String toString() {
    return toStringHelper(this)
        .add("ready", ready)
        .add("newlyAdded", newlyAdded)
        .add("pending", pending)
        .toString();
  }
}
