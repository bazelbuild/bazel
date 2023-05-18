// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.StarlarkGlobals;
import com.google.devtools.build.lib.packages.StarlarkLibrary;
import com.google.devtools.build.lib.packages.StarlarkNativeModule;

/**
 * Sole implementation of {@link StarlarkGlobals}.
 *
 * <p>The reason for the class-interface split is to allow {@link BazelStarlarkEnvironment} to
 * retrieve symbols defined and aggregated in the lib/analysis/ dir, without creating a dependency
 * from lib/packages/ to lib/analysis.
 */
public final class StarlarkGlobalsImpl implements StarlarkGlobals {

  private StarlarkGlobalsImpl() {}

  public static final StarlarkGlobalsImpl INSTANCE = new StarlarkGlobalsImpl();

  @Override
  public ImmutableMap<String, Object> getFixedBuildFileToplevelsSharedWithNative() {
    return StarlarkNativeModule.BINDINGS_FOR_BUILD_FILES;
  }

  @Override
  public ImmutableMap<String, Object> getFixedBuildFileToplevelsNotInNative() {
    return StarlarkLibrary.BUILD; // e.g. select, depset
  }
}
