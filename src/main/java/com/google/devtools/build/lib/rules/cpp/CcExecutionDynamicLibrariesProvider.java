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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A target that provides the execution-time dynamic libraries of a C++ rule.
 */
@Immutable
public final class CcExecutionDynamicLibrariesProvider implements TransitiveInfoProvider {
  public static final CcExecutionDynamicLibrariesProvider EMPTY =
      new CcExecutionDynamicLibrariesProvider(
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));

  private final NestedSet<Artifact> ccExecutionDynamicLibraries;

  public CcExecutionDynamicLibrariesProvider(NestedSet<Artifact> ccExecutionDynamicLibraries) {
    this.ccExecutionDynamicLibraries = ccExecutionDynamicLibraries;
  }

  /**
   * Returns the execution-time dynamic libraries.
   *
   *  <p>This normally returns the dynamic library created by the rule itself. However, if the rule
   * does not create any dynamic libraries, then it returns the combined results of calling
   * getExecutionDynamicLibraryArtifacts on all the rule's deps. This behaviour is so that this
   * method is useful for a cc_library with deps but no srcs.
   */
  public NestedSet<Artifact> getExecutionDynamicLibraryArtifacts() {
    return ccExecutionDynamicLibraries;
  }
}
