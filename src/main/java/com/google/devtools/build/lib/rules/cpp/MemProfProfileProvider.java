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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;

/** Provider that contains the memprof profile. */
@Immutable
public final class MemProfProfileProvider extends NativeInfo {
  public static final BuiltinProvider<MemProfProfileProvider> PROVIDER =
      new BuiltinProvider<MemProfProfileProvider>(
          "MemProfProfileInfo", MemProfProfileProvider.class) {};

  private final FdoInputFile memprofInputFile;

  public MemProfProfileProvider(FdoInputFile memprofInputFile) {
    this.memprofInputFile = memprofInputFile;
  }

  @Override
  public BuiltinProvider<MemProfProfileProvider> getProvider() {
    return PROVIDER;
  }

  public FdoInputFile getInputFile() {
    return memprofInputFile;
  }
}
