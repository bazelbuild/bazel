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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.common.options.OptionsClassProvider;

import javax.annotation.Nullable;

/**
 * A driver for auto-updating graphs which operate over monotonically increasing integer versions.
 */
public class SequentialBuildDriver implements BuildDriver {
  private final MemoizingEvaluator memoizingEvaluator;
  private IntVersion curVersion;

  public SequentialBuildDriver(MemoizingEvaluator evaluator) {
    this.memoizingEvaluator = Preconditions.checkNotNull(evaluator);
    this.curVersion = IntVersion.of(0);
  }

  @Override
  public <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<SkyKey> roots, boolean keepGoing, int numThreads, EventHandler reporter)
      throws InterruptedException {
    try {
      return memoizingEvaluator.evaluate(roots, curVersion, keepGoing, numThreads, reporter);
    } finally {
      curVersion = curVersion.next();
    }
  }

 @Override
 public String meta(Iterable<SkyKey> of, OptionsClassProvider options) {
   return "";
 }

 @Override
  public MemoizingEvaluator getGraphForTesting() {
    return memoizingEvaluator;
  }

  @Nullable
  @Override
  public SkyValue getExistingValueForTesting(SkyKey key) {
    return memoizingEvaluator.getExistingValueForTesting(key);
  }

  @Nullable
  @Override
  public ErrorInfo getExistingErrorForTesting(SkyKey key) {
    return memoizingEvaluator.getExistingErrorForTesting(key);
  }
}
