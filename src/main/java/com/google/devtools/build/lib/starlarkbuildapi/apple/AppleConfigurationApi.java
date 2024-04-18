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

package com.google.devtools.build.lib.starlarkbuildapi.apple;

import com.google.devtools.build.docgen.annot.DocCategory;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** An interface for a configuration type containing info for Apple platforms and tools. */
@StarlarkBuiltin(
    name = "apple",
    doc = "A configuration fragment for Apple platforms.",
    category = DocCategory.CONFIGURATION_FRAGMENT)
public interface AppleConfigurationApi<ApplePlatformTypeApiT extends ApplePlatformTypeApi>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "single_arch_cpu",
      structField = true,
      doc =
          "The single \"effective\" architecture for this configuration (e.g., <code>i386</code>"
              + " or <code>arm64</code>) in the context of rule logic that is only concerned with"
              + " a single architecture (such as <code>objc_library</code>, which registers"
              + " single-architecture compile actions).")
  String getSingleArchitecture();

  @StarlarkMethod(
      name = "single_arch_platform",
      doc =
          "The platform of the current configuration. This should only be invoked in a context "
              + "where only a single architecture may be supported; consider "
              + "<a href='#multi_arch_platform'>multi_arch_platform</a> for other cases.",
      structField = true)
  ApplePlatformApi getSingleArchPlatform();

  @StarlarkMethod(
      name = "multi_arch_platform",
      doc =
          "The platform of the current configuration for the given platform type. This should only "
              + "be invoked in a context where multiple architectures may be supported; consider "
              + "<a href='#single_arch_platform'>single_arch_platform</a> for other cases.",
      parameters = {
        @Param(
            name = "platform_type",
            positional = true,
            named = false,
            doc = "The apple platform type.")
      })
  ApplePlatformApi getMultiArchPlatform(ApplePlatformTypeApiT platformType);

  @Nullable
  @StarlarkMethod(
      name = "xcode_version_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  String getXcodeVersionFlag() throws EvalException;

  @StarlarkMethod(
      name = "ios_sdk_version_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> iosSdkVersionFlag() throws EvalException;

  @StarlarkMethod(
      name = "macos_sdk_version_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> macOsSdkVersionFlag() throws EvalException;

  @StarlarkMethod(
      name = "tvos_sdk_version_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> tvOsSdkVersionFlag() throws EvalException;

  @StarlarkMethod(
      name = "watchos_sdk_version_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> watchOsSdkVersionFlag() throws EvalException;

  @StarlarkMethod(
      name = "ios_minimum_os_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> iosMinimumOsFlag() throws EvalException;

  @StarlarkMethod(
      name = "macos_minimum_os_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> macOsMinimumOsFlag() throws EvalException;

  @StarlarkMethod(
      name = "tvos_minimum_os_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> tvOsMinimumOsFlag() throws EvalException;

  @StarlarkMethod(
      name = "watchos_minimum_os_flag",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  DottedVersionApi<?> watchOsMinimumOsFlag() throws EvalException;

  @StarlarkMethod(name = "prefer_mutual_xcode", documented = false, structField = true)
  public boolean shouldPreferMutualXcode() throws EvalException;

  @StarlarkMethod(name = "include_xcode_exec_requirements", documented = false, structField = true)
  public boolean includeXcodeExecRequirementsFlag() throws EvalException;
}
