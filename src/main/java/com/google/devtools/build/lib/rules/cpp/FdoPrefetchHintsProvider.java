// Copyright 2018 The Bazel Authors. All rights reserved.
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

/** Provider that contains the profile used for prefetch hints. */
@Immutable
public final class FdoPrefetchHintsProvider extends NativeInfo implements StarlarkValue {
  public static final BuiltinProvider<FdoPrefetchHintsProvider> PROVIDER =
      new BuiltinProvider<FdoPrefetchHintsProvider>(
          "FdoPrefetchHintsInfo", FdoPrefetchHintsProvider.class) {};

  private final FdoInputFile fdoInputFile;

  public FdoPrefetchHintsProvider(FdoInputFile fdoInputFile) {
    this.fdoInputFile = fdoInputFile;
  }

  @Override
  public BuiltinProvider<FdoPrefetchHintsProvider> getProvider() {
    return PROVIDER;
  }

  public FdoInputFile getInputFile() {
    return fdoInputFile;
  }
}
