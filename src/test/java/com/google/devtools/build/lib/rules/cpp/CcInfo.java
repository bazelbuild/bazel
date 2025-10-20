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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import net.starlark.java.eval.EvalException;

/** Provider for C++ compilation and linking information. */
public final class CcInfo {
  public static final CcInfoProvider PROVIDER = new CcInfoProvider();

  /** A wrapper around the Starlark provider. */
  public static class CcInfoProvider extends StarlarkProviderWrapper<CcInfo> {
    public CcInfoProvider() {
      super(
          BzlLoadValue.keyForBuiltins(
              Label.parseCanonicalUnchecked("@_builtins//:common/cc/cc_info.bzl")),
          "CcInfo");
    }

    @Override
    public CcInfo wrap(Info value) {
      return new CcInfo((StarlarkInfo) value);
    }
  }

  private final StarlarkInfo starlarkInfo;

  private CcInfo(StarlarkInfo starlarkInfo) {
    this.starlarkInfo = starlarkInfo;
  }

  public static CcInfo wrap(StarlarkInfo starlarkInfo) {
    return new CcInfo(starlarkInfo);
  }

  public CcCompilationContext getCcCompilationContext() {
    try {
      return CcCompilationContext.of(
          starlarkInfo.getValue("compilation_context", StarlarkInfo.class));
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  public CcLinkingContext getCcLinkingContext() {
    try {
      return CcLinkingContext.of(starlarkInfo.getValue("linking_context", StarlarkInfo.class));
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  public StarlarkInfo getCcDebugInfoContext() {
    try {
      return starlarkInfo.getValue("_debug_context", StarlarkInfo.class);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  public NestedSet<LibraryToLink> getTransitiveCcNativeLibrariesForTests() {
    try {
      return LibraryToLink.wrap(
          starlarkInfo
              .getValue("_legacy_transitive_native_libraries", Depset.class)
              .getSet(StarlarkInfo.class));
    } catch (EvalException | TypeException e) {
      throw new IllegalStateException(e);
    }
  }
}
