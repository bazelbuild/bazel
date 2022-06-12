// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * A callback for {@link
 * com.google.devtools.build.lib.pkgcache.RecursivePackageProvider#streamPackagesUnderDirectory}
 * that buffers the {@link PackageIdentifier} instances it receives into bounded-size batches that
 * it delivers to a supplied callback.
 *
 * <p>This callback must be {@link #close() closed} to deliver this final batch.
 */
@ThreadSafe
public interface PackageIdentifierBatchingCallback
    extends SafeBatchCallback<PackageIdentifier>, AutoCloseable {
  void close() throws InterruptedException;

  /** Factory for {@link PackageIdentifierBatchingCallback}. */
  interface Factory {
    PackageIdentifierBatchingCallback create(
        SafeBatchCallback<PackageIdentifier> batchResults, int maxBatchSize);
  }
}
