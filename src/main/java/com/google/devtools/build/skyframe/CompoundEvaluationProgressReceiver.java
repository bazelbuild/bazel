// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;

/**
 * An {@link EvaluationProgressReceiver} that delegates to a bunch of other
 * {@link EvaluationProgressReceiver}s.
 */
public class CompoundEvaluationProgressReceiver implements EvaluationProgressReceiver {
  protected final ImmutableList<? extends EvaluationProgressReceiver> receivers;

  protected CompoundEvaluationProgressReceiver(
      ImmutableList<? extends EvaluationProgressReceiver> receivers) {
    this.receivers = receivers;
  }

  public static EvaluationProgressReceiver of(EvaluationProgressReceiver... receivers) {
    return new CompoundEvaluationProgressReceiver(ImmutableList.copyOf(receivers));
  }

  @Override
  public void invalidated(SkyKey skyKey, InvalidationState state) {
    for (EvaluationProgressReceiver receiver : receivers) {
      receiver.invalidated(skyKey, state);
    }
  }

  @Override
  public void enqueueing(SkyKey skyKey) {
    for (EvaluationProgressReceiver receiver : receivers) {
      receiver.enqueueing(skyKey);
    }
  }

  @Override
  public void computing(SkyKey skyKey) {
    for (EvaluationProgressReceiver receiver : receivers) {
      receiver.computing(skyKey);
    }
  }

  @Override
  public void computed(SkyKey skyKey, long elapsedTimeNanos) {
    for (EvaluationProgressReceiver receiver : receivers) {
      receiver.computed(skyKey, elapsedTimeNanos);
    }
  }

  @Override
  public void evaluated(SkyKey skyKey, Supplier<SkyValue> valueSupplier, EvaluationState state) {
    for (EvaluationProgressReceiver receiver : receivers) {
      receiver.evaluated(skyKey, valueSupplier, state);
    }
  }
}
