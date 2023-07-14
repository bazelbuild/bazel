// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import net.starlark.java.eval.StarlarkValue;

/** Provider that contains the profile used for propeller. */
@Immutable
public final class PropellerOptimizeProvider extends NativeInfo implements StarlarkValue {
  public static final BuiltinProvider<PropellerOptimizeProvider> PROVIDER =
      new BuiltinProvider<PropellerOptimizeProvider>(
          "PropellerOptimizeInfo", PropellerOptimizeProvider.class) {};

  private final PropellerOptimizeInputFile propellerOptimizeInputFile;

  public PropellerOptimizeProvider(PropellerOptimizeInputFile propellerOptimizeInputFile) {
    this.propellerOptimizeInputFile = propellerOptimizeInputFile;
  }

  @Override
  public BuiltinProvider<PropellerOptimizeProvider> getProvider() {
    return PROVIDER;
  }

  public PropellerOptimizeInputFile getInputFile() {
    return propellerOptimizeInputFile;
  }
}
