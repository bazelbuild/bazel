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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcDynamicLibrariesForRuntimeApi;

/** An object that contains the dynamic libraries for runtime of a C++ rule. */
@Immutable
@AutoCodec
public final class CcDynamicLibrariesForRuntime implements
    CcDynamicLibrariesForRuntimeApi {
  public static final CcDynamicLibrariesForRuntime EMPTY =
      new CcDynamicLibrariesForRuntime(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));

  private final NestedSet<Artifact> ccDynamicLibrariesForRuntime;

  public CcDynamicLibrariesForRuntime(NestedSet<Artifact> ccDynamicLibrariesForRuntime) {
    this.ccDynamicLibrariesForRuntime = ccDynamicLibrariesForRuntime;
  }

  /**
   * Returns the dynamic libraries for runtime.
   *
   * <p>This normally returns the dynamic library created by the rule itself. However, if the rule
   * does not create any dynamic libraries, then it returns the combined results of calling
   * getDynamicLibrariesForRuntimeArtifacts on all the rule's deps. This behaviour is so that this
   * method is useful for a cc_library with deps but no srcs.
   */
  public NestedSet<Artifact> getDynamicLibrariesForRuntimeArtifacts() {
    return ccDynamicLibrariesForRuntime;
  }
}
