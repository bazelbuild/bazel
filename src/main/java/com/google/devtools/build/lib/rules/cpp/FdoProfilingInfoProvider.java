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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

/**
 * A target that can contribute profiling information to FDO C++ compilations.
 */
@Immutable
public final class FdoProfilingInfoProvider implements TransitiveInfoProvider {
  public static final FdoProfilingInfoProvider EMPTY =
      new FdoProfilingInfoProvider(NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER));

  private final NestedSet<Label> transitiveLipoLabels;

  public FdoProfilingInfoProvider(NestedSet<Label> transitiveLipoLabels) {
    this.transitiveLipoLabels = transitiveLipoLabels;
  }

  /**
   * Returns the set of LIPO labels for the targets in the transitive
   * closure that can be compiled with LIPO. These are the targets that
   * invoke the C++ compiler and are compiled into C++ binaries.
   */
  public NestedSet<Label> getTransitiveLipoLabels() {
    return transitiveLipoLabels;
  }
}
