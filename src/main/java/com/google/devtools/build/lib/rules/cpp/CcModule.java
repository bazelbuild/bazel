// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/** A module that contains Skylark utilities for C++ support. */
@SkylarkModule(name = "cc_common", doc = "Utilities related to C++ support.")
public class CcModule {
  public static final CcModule INSTANCE = new CcModule();

  @SkylarkCallable(
    name = CcToolchainProvider.SKYLARK_NAME,
    doc =
        "The key used to retrieve the provider that contains information about the C++ "
            + "toolchain being used",
    structField = true
  )
  public Provider getCcToolchainProvider() {
    return ToolchainInfo.PROVIDER;
  }
}
