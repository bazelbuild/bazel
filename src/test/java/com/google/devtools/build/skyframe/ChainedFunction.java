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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.GraphTester.ValueComputer;
import com.google.devtools.build.skyframe.ParallelEvaluator.SkyFunctionEnvironment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

import java.util.concurrent.CountDownLatch;

import javax.annotation.Nullable;

/**
 * {@link ValueComputer} that can be chained together with others of its type to synchronize the
 * order in which builders finish.
 */
public final class ChainedFunction implements SkyFunction {
  @Nullable private final SkyValue value;
  @Nullable private final CountDownLatch notifyStart;
  @Nullable private final CountDownLatch waitToFinish;
  @Nullable private final CountDownLatch notifyFinish;
  private final boolean waitForException;
  private final Iterable<SkyKey> deps;

  /** Do not use! Use {@link Builder} instead. */
  ChainedFunction(
      @Nullable CountDownLatch notifyStart,
      @Nullable CountDownLatch waitToFinish,
      @Nullable CountDownLatch notifyFinish,
      boolean waitForException,
      @Nullable SkyValue value,
      Iterable<SkyKey> deps) {
    this.notifyStart = notifyStart;
    this.waitToFinish = waitToFinish;
    this.notifyFinish = notifyFinish;
    this.waitForException = waitForException;
    Preconditions.checkState(this.waitToFinish != null || !this.waitForException, value);
    this.value = value;
    this.deps = deps;
  }

  @Override
  public SkyValue compute(SkyKey key, SkyFunction.Environment env) throws GenericFunctionException,
      InterruptedException {
    try {
      if (notifyStart != null) {
        notifyStart.countDown();
      }
      if (waitToFinish != null) {
        TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
            waitToFinish, key + " timed out waiting to finish");
        if (waitForException) {
          SkyFunctionEnvironment skyEnv = (SkyFunctionEnvironment) env;
          TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
              skyEnv.getExceptionLatchForTesting(), key + " timed out waiting for exception");
        }
      }
      for (SkyKey dep : deps) {
        env.getValue(dep);
      }
      if (value == null) {
        throw new GenericFunctionException(new SomeErrorException("oops"),
            Transience.PERSISTENT);
      }
      if (env.valuesMissing()) {
        return null;
      }
      return value;
    } finally {
      if (notifyFinish != null) {
        notifyFinish.countDown();
      }
    }
  }

  /** Builder for {@link ChainedFunction} objects. */
  public static class Builder {
    @Nullable private SkyValue value;
    @Nullable private CountDownLatch notifyStart;
    @Nullable private CountDownLatch waitToFinish;
    @Nullable private CountDownLatch notifyFinish;
    private boolean waitForException;
    private Iterable<SkyKey> deps = ImmutableList.of();

    public Builder setValue(SkyValue value) {
      this.value = value;
      return this;
    }

    public Builder setNotifyStart(CountDownLatch notifyStart) {
      this.notifyStart = notifyStart;
      return this;
    }

    public Builder setWaitToFinish(CountDownLatch waitToFinish) {
      this.waitToFinish = waitToFinish;
      return this;
    }

    public Builder setNotifyFinish(CountDownLatch notifyFinish) {
      this.notifyFinish = notifyFinish;
      return this;
    }

    public Builder setWaitForException(boolean waitForException) {
      this.waitForException = waitForException;
      return this;
    }

    public Builder setDeps(Iterable<SkyKey> deps) {
      this.deps = Preconditions.checkNotNull(deps);
      return this;
    }

    public SkyFunction build() {
      return new ChainedFunction(
          notifyStart, waitToFinish, notifyFinish, waitForException, value, deps);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    throw new UnsupportedOperationException();
  }
}
