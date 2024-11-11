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

package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.core.Bootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.core.ContextAndFlagGuardedValue;

/** {@link Bootstrap} for Starlark objects related to Android rules. */
public class AndroidBootstrap implements Bootstrap {
  private static final ImmutableSet<PackageIdentifier> allowedRepositories =
      ImmutableSet.of(
          PackageIdentifier.createUnchecked("_builtins", ""),
          PackageIdentifier.createUnchecked("rules_android", ""),
          PackageIdentifier.createUnchecked("", "tools/build_defs/android"));

  private final AndroidStarlarkCommonApi<?, ?, ?, ?, ?> androidCommon;

  public AndroidBootstrap(AndroidStarlarkCommonApi<?, ?, ?, ?, ?> androidCommon) {

    this.androidCommon = androidCommon;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    // TODO: Make an incompatible change flag to hide android_common behind
    // --experimental_google_legacy_api.
    // Rationale: android_common module contains commonly used functions used outside of
    // the Android Starlark migration. Let's not break them without an incompatible
    // change process.
    builder.put(
        "android_common",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            androidCommon,
            allowedRepositories));
  }
}
