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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.skyframe.GraphTester.NodeComputer;

import java.util.concurrent.CountDownLatch;

import javax.annotation.Nullable;

/**
 * {@link NodeComputer} that can be chained together with others of its type to synchronize the
 * order in which builders finish.
 */
final class ChainedNodeBuilder implements NodeBuilder {
  @Nullable private final Node value;
  @Nullable private final CountDownLatch notifyStart;
  @Nullable private final CountDownLatch waitToFinish;
  @Nullable private final CountDownLatch notifyFinish;
  private final boolean waitForException;
  private final Iterable<NodeKey> deps;

  ChainedNodeBuilder(@Nullable CountDownLatch notifyStart, @Nullable CountDownLatch waitToFinish,
      @Nullable CountDownLatch notifyFinish, boolean waitForException,
      @Nullable Node value, Iterable<NodeKey> deps) {
    this.notifyStart = notifyStart;
    this.waitToFinish = waitToFinish;
    this.notifyFinish = notifyFinish;
    this.waitForException = waitForException;
    Preconditions.checkState(this.waitToFinish != null || !this.waitForException, value);
    this.value = value;
    this.deps = deps;
  }

  @Override
  public Node build(NodeKey key, NodeBuilder.Environment env) throws GenericNodeBuilderException,
      InterruptedException {
    try {
      if (notifyStart != null) {
        notifyStart.countDown();
      }
      if (waitToFinish != null) {
        TrackingAwaiter.waitAndMaybeThrowInterrupt(waitToFinish,
            key + " timed out waiting to finish");
        if (waitForException) {
          TrackingAwaiter.waitAndMaybeThrowInterrupt(env.getExceptionLatchForTesting(),
              key + " timed out waiting for exception");
        }
      }
      for (NodeKey dep : deps) {
        env.getDep(dep);
      }
      if (value == null) {
        throw new GenericNodeBuilderException(key, new Exception());
      }
      if (env.depsMissing()) {
        return null;
      }
      return value;
    } finally {
      if (notifyFinish != null) {
        notifyFinish.countDown();
      }
    }
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    throw new UnsupportedOperationException();
  }
}
