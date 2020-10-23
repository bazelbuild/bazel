// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.SpawnResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.concurrent.ThreadSafe;

/** Collects results from SpawnResult. */
@ThreadSafe
public class SpawnStats {
  private static final ImmutableList<String> REPORT_FIRST = ImmutableList.of("remote cache hit");

  private final ConcurrentHashMultiset<String> runners = ConcurrentHashMultiset.create();
  private final AtomicLong totalWallTimeMillis = new AtomicLong();
  private final AtomicInteger totalNumberOfActions = new AtomicInteger();

  public void countActionResult(ActionResult actionResult) {
    for (SpawnResult r : actionResult.spawnResults()) {
      countRunnerName(r.getRunnerName());
      totalWallTimeMillis.addAndGet(r.getMetrics().executionWallTime().toMillis());
    }
  }

  public void countRunnerName(String runner) {
    runners.add(runner);
  }

  public void incrementActionCount() {
    totalNumberOfActions.incrementAndGet();
  }

  public long getTotalWallTimeMillis() {
    return totalWallTimeMillis.get();
  }

  /*
   * Returns a human-readable summary of spawns counted.
   */
  public String getSummary() {
    return getSummary(REPORT_FIRST);
  }

  /*
   * Returns a human-readable summary of spawns counted.
   */
  public String getSummary(ImmutableList<String> reportFirst) {
    ResultString result = new ResultString();
    int numActionsWithoutInternal = runners.size();
    int numActionsTotal = totalNumberOfActions.get();

    // First report cache results.
    for (String s : reportFirst) {
      int count = runners.setCount(s, 0);
      if (count > 0) {
        result.add(s, count);
      }
    }

    // Account for internal actions such as SymlinkTree.
    if (numActionsWithoutInternal < numActionsTotal) {
      result.add("internal", numActionsTotal - numActionsWithoutInternal);
    }

    // Sort the rest alphabetically
    ArrayList<Multiset.Entry<String>> list = new ArrayList<>(runners.entrySet());
    Collections.sort(list, Comparator.comparing(e -> e.getElement()));

    for (Multiset.Entry<String> e : list) {
      result.add(e.getElement(), e.getCount());
    }

    return numActionsTotal + " process" + (numActionsTotal == 1 ? "" : "es") + result + ".";
  }

  private static class ResultString {
    StringBuilder result = new StringBuilder();
    int runnersNum = 0;

    public void add(String name, int count) {
      runnersNum += 1;

      if (result.length() > 0) {
        result.append(", ");
      }
      result.append(count);
      result.append(" ");
      result.append(name);
    }

    @Override
    public String toString() {
      if (runnersNum == 0) {
        return "";
      }
      return ": " + result;
    }
  }
}
