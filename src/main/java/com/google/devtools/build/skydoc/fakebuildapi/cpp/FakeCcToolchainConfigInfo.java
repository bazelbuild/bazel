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

package com.google.devtools.build.skydoc.fakebuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainConfigInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;

/** Fake implementation of {@link CcToolchainConfigInfoApi}. */
public class FakeCcToolchainConfigInfo implements CcToolchainConfigInfoApi {

  @Override
  public String getProto() {
    return null;
  }

  @Override
  public String toProto() throws EvalException {
    return null;
  }

  @Override
  public String toJson() throws EvalException {
    return null;
  }

  @Override
  public void repr(Printer printer) {}

  /** Fake implementation of {@link CcToolchainConfigInfoApi.Provider}. */
  public static class Provider implements CcToolchainConfigInfoApi.Provider {

    @Override
    public void repr(Printer printer) {}
  }
}
