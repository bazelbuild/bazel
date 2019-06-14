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

import com.google.common.collect.ImmutableList;

/**
 * An {@link EvaluationProgressReceiver} that delegates to a bunch of other {@link
 * EvaluationProgressReceiver}s.
 */
public final class CompoundEvaluationProgressReceiver
    extends CompoundEvaluationProgressReceiverBase {
  private CompoundEvaluationProgressReceiver(
      ImmutableList<? extends EvaluationProgressReceiver> receivers) {
    super(receivers);
  }

  public static EvaluationProgressReceiver of(EvaluationProgressReceiver... receivers) {
    return new CompoundEvaluationProgressReceiver(ImmutableList.copyOf(receivers));
  }
}
