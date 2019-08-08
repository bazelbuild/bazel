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
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import java.util.function.Consumer;

/**
 * A callback for {@link RecursivePackageProvider#streamPackagesUnderDirectory} that buffers the
 * PackageIdentifiers it receives into batches that it delivers to a supplied {@code
 * Consumer<ImmutableList<PackageIdentifier>>}.
 */
@ThreadCompatible
public class PackageIdentifierBatchingCallback
    implements Consumer<PackageIdentifier>, AutoCloseable {

  private final Consumer<ImmutableList<PackageIdentifier>> batchResults;
  private final int batchSize;
  private ImmutableList.Builder<PackageIdentifier> packageIdentifiers;
  private int bufferedPackageIds;

  public PackageIdentifierBatchingCallback(
      Consumer<ImmutableList<PackageIdentifier>> batchResults, int batchSize) {
    this.batchResults = batchResults;
    this.batchSize = batchSize;
    reset();
  }

  @Override
  public void accept(PackageIdentifier path) {
    packageIdentifiers.add(path);
    bufferedPackageIds++;
    if (bufferedPackageIds >= this.batchSize) {
      flush();
    }
  }

  @Override
  public void close() {
    flush();
  }

  private void flush() {
    if (bufferedPackageIds > 0) {
      batchResults.accept(packageIdentifiers.build());
      reset();
    }
  }

  private void reset() {
    packageIdentifiers = ImmutableList.builderWithExpectedSize(batchSize);
    bufferedPackageIds = 0;
  }
}
