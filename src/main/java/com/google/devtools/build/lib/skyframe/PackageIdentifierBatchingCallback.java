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
import com.google.devtools.build.lib.concurrent.ParallelVisitor.UnusedException;
import com.google.devtools.build.lib.concurrent.ThreadSafeBatchCallback;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import javax.annotation.concurrent.GuardedBy;

/**
 * A callback for {@link
 * com.google.devtools.build.lib.pkgcache.RecursivePackageProvider#streamPackagesUnderDirectory}
 * that buffers the PackageIdentifiers it receives into fixed-size batches that it delivers to a
 * supplied {@code ThreadSafeBatchCallback<PackageIdentifier, RuntimeException>}.
 *
 * <p>The final batch delivered to the delegate callback may be smaller than the fixed size; the
 * callback must be {@link #close() closed} to deliver this final batch.
 */
@ThreadSafe
public class PackageIdentifierBatchingCallback
    implements ThreadSafeBatchCallback<PackageIdentifier, UnusedException>, AutoCloseable {

  private final ThreadSafeBatchCallback<PackageIdentifier, UnusedException> batchResults;
  private final int batchSize;

  @GuardedBy("this")
  private ImmutableList.Builder<PackageIdentifier> packageIdentifiers;

  @GuardedBy("this")
  private int bufferedPackageIds;

  public PackageIdentifierBatchingCallback(
      ThreadSafeBatchCallback<PackageIdentifier, UnusedException> batchResults, int batchSize) {
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
