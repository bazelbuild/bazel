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
package com.google.devtools.build.lib.profiler;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;

import java.util.List;

/**
 * A stat recorder that is able to look at the kind of object added and delegate to the appropriate
 * {@link StatRecorder} based on a predicate.
 *
 * <p> Note that the predicates are evaluated in order and delegated only to the first one. That
 * means that the most specific and cheapest predicates should be passed first.
 */
public class PredicateBasedStatRecorder implements StatRecorder {

  private final Predicate[] predicates;
  private final StatRecorder[] recorders;

  public PredicateBasedStatRecorder(List<RecorderAndPredicate> stats) {
    predicates = new Predicate[stats.size()];
    recorders = new StatRecorder[stats.size()];
    for (int i = 0; i < stats.size(); i++) {
      RecorderAndPredicate stat = stats.get(i);
      predicates[i] = stat.predicate;
      recorders[i] = stat.recorder;
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public void addStat(int duration, Object obj) {
    String description = obj.toString();
    for (int i = 0; i < predicates.length; i++) {
      if (predicates[i].apply(description)) {
        recorders[i].addStat(duration, obj);
        return;
      }
    }
  }

  @Override
  public boolean isEmpty() {
    for (StatRecorder recorder : recorders) {
      if (!recorder.isEmpty()) {
        return false;
      }
    }
    return true;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (StatRecorder recorder : recorders) {
      if (recorder.isEmpty()) {
        continue;
      }
      sb.append(recorder);
      sb.append("\n");
    }
    return sb.toString();
  }

  /**
   * A Wrapper of a {@code StatRecorder} and a {@code Predicate}. Objects that matches the predicate
   * will be delegated to the StatRecorder.
   */
  public static final class RecorderAndPredicate {

    private final StatRecorder recorder;
    private final Predicate<? super String> predicate;

    public RecorderAndPredicate(StatRecorder recorder, Predicate<? super String> predicate) {
      this.recorder = recorder;
      this.predicate = predicate;
    }
  }

  /** Returns all the delegate stat recorders. */
  public ImmutableList<StatRecorder> getRecorders() {
    return ImmutableList.<StatRecorder>builder().add(recorders).build();
  }
}
