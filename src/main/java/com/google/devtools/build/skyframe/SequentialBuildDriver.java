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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.common.options.OptionsProvider;
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
      Iterable<? extends SkyKey> roots, EvaluationContext evaluationContext)
      throws InterruptedException {
    try {
      return memoizingEvaluator.evaluate(
          roots,
          curVersion,
          evaluationContext.getExecutorService() == null
              ? EvaluationContext.newBuilder()
                  .copyFrom(evaluationContext)
                  .setExecutorServiceSupplier(
                      () ->
                          AbstractQueueVisitor.createExecutorService(
                              evaluationContext.getNumThreads(), "skyframe-evaluator"))
                  .build()
              : evaluationContext);
    } finally {
      curVersion = curVersion.next();
    }
  }

  @Override
  public String meta(Iterable<SkyKey> of, OptionsProvider options) {
    return "";
  }

  @Override
  public MemoizingEvaluator getGraphForTesting() {
    return memoizingEvaluator;
  }

  @Nullable
  @Override
  public SkyValue getExistingValueForTesting(SkyKey key) throws InterruptedException {
    return memoizingEvaluator.getExistingValue(key);
  }

  @Nullable
  @Override
  public ErrorInfo getExistingErrorForTesting(SkyKey key) throws InterruptedException {
    return memoizingEvaluator.getExistingErrorForTesting(key);
  }

  @Nullable
  @Override
  public NodeEntry getEntryForTesting(SkyKey key) throws InterruptedException {
    return memoizingEvaluator.getExistingEntryAtLatestVersion(key);
  }
}
