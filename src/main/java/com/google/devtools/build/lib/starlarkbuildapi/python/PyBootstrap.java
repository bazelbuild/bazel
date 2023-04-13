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

package com.google.devtools.build.lib.starlarkbuildapi.python;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.core.Bootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.core.ContextAndFlagGuardedValue;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.PyWrapCcHelperApi;
import com.google.devtools.build.lib.starlarkbuildapi.python.PyInfoApi.PyInfoProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.stubs.ProviderStub;
import net.starlark.java.eval.FlagGuardedValue;

/** {@link Bootstrap} for Starlark objects related to the Python rules. */
public class PyBootstrap implements Bootstrap {
  public static final ImmutableSet<PackageIdentifier> allowedRepositories =
      ImmutableSet.of(
          PackageIdentifier.createUnchecked("_builtins", ""),
          PackageIdentifier.createUnchecked("bazel_tools", ""),
          PackageIdentifier.createUnchecked("rules_python", ""),
          PackageIdentifier.createUnchecked("", "tools/build_defs/python"));

  private final PyInfoProviderApi pyInfoProviderApi;
  private final PyStarlarkTransitionsApi pyStarlarkTransitionsApi;
  private final PyWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?> pyWrapCcHelper;

  public PyBootstrap(
      PyInfoProviderApi pyInfoProviderApi,
      PyStarlarkTransitionsApi pyStarlarkTransitionsApi,
      PyWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?> pyWrapCcHelper) {
    this.pyInfoProviderApi = pyInfoProviderApi;
    this.pyStarlarkTransitionsApi = pyStarlarkTransitionsApi;
    this.pyWrapCcHelper = pyWrapCcHelper;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put(
        "PyInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            pyInfoProviderApi,
            allowedRepositories));
    builder.put(
        "PyRuntimeInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            // Workaround for https://github.com/bazelbuild/bazel/issues/17713
            new ProviderStub(),
            allowedRepositories));

    builder.put(
        "py_transitions",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API, pyStarlarkTransitionsApi));
    builder.put(
        "py_wrap_cc_helper_do_not_use",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API, pyWrapCcHelper));
    builder.put(
        "PyWrapCcInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            // Workaround for https://github.com/bazelbuild/bazel/issues/17713
            new ProviderStub(),
            allowedRepositories));
    builder.put(
        "PyCcLinkParamsProvider",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            // Workaround for https://github.com/bazelbuild/bazel/issues/17713
            new ProviderStub(),
            allowedRepositories));
  }
}
