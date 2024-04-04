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
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.StarlarkThread;

/** A provider containing information about a version of Xcode and its properties. */
@StarlarkBuiltin(
    name = "XcodeProperties",
    category = DocCategory.PROVIDER,
    doc = "A provider containing information about a version of Xcode and its properties.")
public interface XcodePropertiesApi extends StructApi {
  String NAME = "XcodeProperties";

  @StarlarkMethod(
      name = "xcode_version",
      doc = "The Xcode version, or <code>None</code> if the Xcode version is unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getXcodeVersionString();

  @StarlarkMethod(
      name = "default_ios_sdk_version",
      doc =
          "The default iOS SDK version for this version of Xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultIosSdkVersionString();

  @StarlarkMethod(
      name = "ios_sdk_minimum_os",
      doc = "The minimum OS version supported by the iOS SDK for this version of Xcode.",
      structField = true)
  String getIosSdkMinimumOsString();

  @StarlarkMethod(
      name = "default_visionos_sdk_version",
      doc =
          "The default visionOS SDK version for this version of Xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultVisionosSdkVersionString();

  @StarlarkMethod(
      name = "visionos_sdk_minimum_os",
      doc = "The minimum OS version supported by the visionOS SDK for this version of Xcode.",
      structField = true)
  String getVisionosSdkMinimumOsString();

  @StarlarkMethod(
      name = "default_watchos_sdk_version",
      doc =
          "The default watchOS SDK version for this version of Xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultWatchosSdkVersionString();

  @StarlarkMethod(
      name = "watchos_sdk_minimum_os",
      doc = "The minimum OS version supported by the watchOS SDK for this version of Xcode.",
      structField = true)
  String getWatchosSdkMinimumOsString();

  @StarlarkMethod(
      name = "default_tvos_sdk_version",
      doc =
          "The default tvOS SDK version for this version of Xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultTvosSdkVersionString();

  @StarlarkMethod(
      name = "tvos_sdk_minimum_os",
      doc = "The minimum OS version supported by the tvOS SDK for this version of Xcode.",
      structField = true)
  String getTvosSdkMinimumOsString();

  @StarlarkMethod(
      name = "default_macos_sdk_version",
      doc =
          "The default macOS SDK version for this version of Xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultMacosSdkVersionString();

  @StarlarkMethod(
      name = "macos_sdk_minimum_os",
      doc = "The minimum OS version supported by the macOS SDK for this version of Xcode.",
      structField = true)
  String getMacosSdkMinimumOsString();

  /** The provider implementing this can construct XcodeProperties objects. */
  @StarlarkBuiltin(name = "Provider", doc = "", documented = false)
  interface Provider extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        useStarlarkThread = true,
        parameters = {
          @Param(
              name = "version",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "default_ios_sdk_version",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "ios_sdk_minimum_os",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "default_visionos_sdk_version",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "visionos_sdk_minimum_os",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "default_watchos_sdk_version",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "watchos_sdk_minimum_os",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "default_tvos_sdk_version",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "tvos_sdk_minimum_os",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "default_macos_sdk_version",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
          @Param(
              name = "macos_sdk_minimum_os",
              doc = "",
              positional = false,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = String.class),
              },
              named = true,
              defaultValue = "None"),
        },
        selfCall = true,
        documented = false)
    @StarlarkConstructor
    XcodePropertiesApi createInfo(
        Object version,
        Object defaultIosSdkVersion,
        Object iosSdkMinimumOs,
        Object defaultVisionosSdkVersion,
        Object visionosSdkMinimumOs,
        Object defaultWatchosSdkVersion,
        Object watchosSdkMinimumOs,
        Object defaultTvosSdkVersion,
        Object tvosSdkMinimumOs,
        Object defaultMacosSdkVersion,
        Object macosSdkMinimumOs,
        StarlarkThread thread)
        throws EvalException;
  }
}
