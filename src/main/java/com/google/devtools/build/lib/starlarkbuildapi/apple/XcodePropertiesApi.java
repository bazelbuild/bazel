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
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** A provider containing information about a version of Xcode and its properties. */
@StarlarkBuiltin(
    name = "XcodeProperties",
    category = DocCategory.PROVIDER,
    doc = "A provider containing information about a version of Xcode and its properties.")
public interface XcodePropertiesApi extends StructApi {

  @StarlarkMethod(
      name = "xcode_version",
      doc = "The xcode version, or <code>None</code> if the xcode version is unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getXcodeVersionString();

  @StarlarkMethod(
      name = "default_ios_sdk_version",
      doc =
          "The default iOS sdk version for this version of xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultIosSdkVersionString();

  @StarlarkMethod(
      name = "default_visionos_sdk_version",
      doc =
          "The default visionOS sdk version for this version of xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultVisionosSdkVersionString();

  @StarlarkMethod(
      name = "default_watchos_sdk_version",
      doc =
          "The default watchOS sdk version for this version of xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultWatchosSdkVersionString();

  @StarlarkMethod(
      name = "default_tvos_sdk_version",
      doc =
          "The default tvOS sdk version for this version of xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultTvosSdkVersionString();

  @StarlarkMethod(
      name = "default_macos_sdk_version",
      doc =
          "The default macOS sdk version for this version of xcode, or <code>None</code> if "
              + "unknown.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getDefaultMacosSdkVersionString();
}
