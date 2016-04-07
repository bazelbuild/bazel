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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Supplier;
import com.google.common.collect.Sets;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Set;

/**
 * An {@link EvaluationProgressReceiver} used during loading phase
 *
 * It receives updates about progress in the loading phase via the
 * {@link EvaluationProgressReceiver} interface. These updates get aggregated
 * and a summary of the current state of the loading phase can be obtained via
 * and provides a summary about the current state via the {@link #progressState}
 * method.
 */
public class LoadingProgressReceiver implements EvaluationProgressReceiver {

  private final Set<SkyKey> enqueuedPackages = Sets.newConcurrentHashSet();
  private final Set<SkyKey> completedPackages = Sets.newConcurrentHashSet();
  private final Deque<SkyKey> pending = new ArrayDeque<>();

  @Override
  public void invalidated(SkyKey skyKey, InvalidationState state) {}

  @Override
  public synchronized void enqueueing(SkyKey skyKey) {
    if (skyKey.functionName().equals(SkyFunctions.PACKAGE)) {
      enqueuedPackages.add(skyKey);
      pending.addLast(skyKey);
    }
  }

  @Override
  public void computed(SkyKey skyKey, long elapsedTimeNanos) {}

  @Override
  public synchronized void evaluated(
      SkyKey skyKey, Supplier<SkyValue> valueSupplier, EvaluationState state) {
    if (skyKey.functionName().equals(SkyFunctions.PACKAGE)) {
      completedPackages.add(skyKey);
      pending.remove(skyKey);
    }
  }

  public synchronized String progressState() {
    long completed = completedPackages.size();
    long enqueued = enqueuedPackages.size();
    String answer = "" + completed + " / " + enqueued;
    if (enqueued > completed) {
      answer += " " + pending.peekFirst().toString();
      if (enqueued > completed + 1) {
        answer += " ...";
      }
    }
    return answer;
  }
}
