// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi.python;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.python.PyInfoApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/** Fake implementation of {@link PyInfoApi}. */
public class FakePyInfo implements PyInfoApi<FileApi> {

  @Override
  public Depset getTransitiveSources() {
    return null;
  }

  @Override
  public boolean getUsesSharedLibraries() {
    return false;
  }

  @Override
  public Depset getImports() {
    return null;
  }

  @Override
  public boolean getHasPy2OnlySources() {
    return false;
  }

  @Override
  public boolean getHasPy3OnlySources() {
    return false;
  }

  @Override
  public void repr(Printer printer) {}

  /** Fake implementation of {@link PyInfoProviderApi}. */
  public static class FakePyInfoProvider implements PyInfoProviderApi {

    @Override
    public PyInfoApi<?> constructor(
        Depset transitiveSources,
        boolean usesSharedLibraries,
        Object importsUncast,
        boolean hasPy2OnlySources,
        boolean hasPy3OnlySources,
        StarlarkThread thread)
        throws EvalException {
      return new FakePyInfo();
    }

    @Override
    public void repr(Printer printer) {}
  }
}
