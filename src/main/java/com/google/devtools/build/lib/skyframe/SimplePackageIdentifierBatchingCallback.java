// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import javax.annotation.concurrent.GuardedBy;

/**
 * Simple implementation of {@link PackageIdentifierBatchingCallback} that naively shards a stream
 * of {@link PackageIdentifier} instances, in order, into fixed-size batches. The final batch may be
 * smaller than the others.
 */
public class SimplePackageIdentifierBatchingCallback implements PackageIdentifierBatchingCallback {
  private final SafeBatchCallback<PackageIdentifier> batchResults;
  private final int batchSize;

  @GuardedBy("this")
  private ImmutableList.Builder<PackageIdentifier> packageIdentifiers;

  @GuardedBy("this")
  private int bufferedPackageIds;

  public SimplePackageIdentifierBatchingCallback(
      SafeBatchCallback<PackageIdentifier> batchResults, int batchSize) {
    this.batchResults = batchResults;
    this.batchSize = batchSize;
    reset();
  }

  @Override
  public synchronized void process(Iterable<PackageIdentifier> partialResult)
      throws InterruptedException {
    for (PackageIdentifier path : partialResult) {
      packageIdentifiers.add(path);
      bufferedPackageIds++;
      if (bufferedPackageIds >= this.batchSize) {
        flush();
      }
    }
  }

  @Override
  public synchronized void close() throws InterruptedException {
    flush();
  }

  @GuardedBy("this")
  private void flush() throws InterruptedException {
    if (bufferedPackageIds > 0) {
      batchResults.process(packageIdentifiers.build());
      reset();
    }
  }

  @GuardedBy("this")
  private void reset() {
    packageIdentifiers = ImmutableList.builderWithExpectedSize(batchSize);
    bufferedPackageIds = 0;
  }
}
