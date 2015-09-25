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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A target that can contribute profiling information to LIPO C++ compilations.
 *
 * <p>This is used in the LIPO context collector tree to collect data from the transitive
 * closure of the :lipo_context_collector target. It is eventually passed to the configured
 * targets in the target configuration through {@link LipoContextProvider}.
 */
@Immutable
public final class TransitiveLipoInfoProvider implements TransitiveInfoProvider {
  public static final TransitiveLipoInfoProvider EMPTY =
      new TransitiveLipoInfoProvider(
          NestedSetBuilder.<IncludeScannable>emptySet(Order.STABLE_ORDER));

  private final NestedSet<IncludeScannable> includeScannables;

  public TransitiveLipoInfoProvider(NestedSet<IncludeScannable> includeScannables) {
    this.includeScannables = includeScannables;
  }

  /**
   * Returns the include scannables in the transitive closure.
   *
   * <p>This is used for constructing the path fragment -> include scannable map in the
   * LIPO-enabled target configuration.
   */
  public NestedSet<IncludeScannable> getTransitiveIncludeScannables() {
    return includeScannables;
  }
}
