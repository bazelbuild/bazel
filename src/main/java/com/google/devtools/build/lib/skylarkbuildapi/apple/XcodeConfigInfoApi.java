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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * An interface for an info type containing the set of Apple versions computed from command line
 * options and the {@code xcode_config} rule.
 */
@SkylarkModule(
    name = "XcodeVersionConfig",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "The set of Apple versions computed from command line options and the xcode_config rule.")
public interface XcodeConfigInfoApi<
        ApplePlatformApiT extends ApplePlatformApi,
        ApplePlatformTypeApiT extends ApplePlatformTypeApi>
    extends StructApi {

  @SkylarkCallable(
      name = "xcode_version",
      doc =
          "Returns the Xcode version that is being used to build.<p>"
              + "This will return <code>None</code> if no Xcode versions are available.",
      allowReturnNones = true)
  DottedVersionApi<?> getXcodeVersion();

  @SkylarkCallable(
      name = "minimum_os_for_platform_type",
      doc =
          "The minimum compatible OS version for target simulator and devices for a particular "
              + "platform type.",
      parameters = {
        @Param(
            name = "platform_type",
            positional = true,
            named = false,
            type = ApplePlatformTypeApi.class,
            doc = "The apple platform type."),
      })
  DottedVersionApi<?> getMinimumOsForPlatformType(ApplePlatformTypeApiT platformType);

  @SkylarkCallable(
      name = "sdk_version_for_platform",
      doc =
          "The version of the platform SDK that will be used to build targets for the given "
              + "platform.",
      parameters = {
        @Param(
            name = "platform",
            positional = true,
            named = false,
            type = ApplePlatformApi.class,
            doc = "The apple platform."),
      })
  DottedVersionApi<?> getSdkVersionForPlatform(ApplePlatformApiT platform);

  @SkylarkCallable(
      name = "availability",
      doc =
          "Returns the availability of this Xcode version, 'remote' if the version is only"
              + " available remotely, 'local' if the version is only available locally, 'both' if"
              + " the version is available both locally and remotely, or 'unknown' if the"
              + " availability could not be determined.")
  public String getAvailabilityString();

  /** An interface for the provider of {@link XcodeConfigInfoApi}. */
  @SkylarkModule(
      name = "Provider",
      category = SkylarkModuleCategory.PROVIDER,
      documented = false,
      doc = "")
  interface XcodeConfigProviderApi extends ProviderApi {

    @SkylarkCallable(
        name = "XcodeVersionConfig",
        doc = "Returns the Xcode info that is associated with this target",
        parameters = {
          @Param(
              name = "iosSdkVersion",
              named = true,
              positional = false,
              doc = "The ios SDK version."),
          @Param(
              name = "iosMinimumOsVersion",
              named = true,
              positional = false,
              doc = "The ios minimum os version."),
          @Param(
              name = "watchosSdkVersion",
              named = true,
              positional = false,
              doc = "The watchos SDK version."),
          @Param(
              name = "watchosMinimumOsVersion",
              named = true,
              positional = false,
              doc = "The watchos minimum os version."),
          @Param(
              name = "tvosSdkVersion",
              named = true,
              positional = false,
              doc = "The tvos SDK version."),
          @Param(
              name = "tvosMinimumOsVersion",
              named = true,
              positional = false,
              doc = "The tvos minimum os version."),
          @Param(
              name = "macosSdkVersion",
              named = true,
              positional = false,
              doc = "The macos SDK version."),
          @Param(
              name = "macosMinimumOsVersion",
              named = true,
              positional = false,
              doc = "The macos minimum os version."),
          @Param(
              name = "xcodeVersion",
              named = true,
              positional = false,
              doc = "The selected Xcode version from this config."),
        },
        selfCall = true)
    @SkylarkConstructor(
        objectType = XcodeConfigInfoApi.class,
        receiverNameForDoc = "XcodeConfigInfo")
    XcodeConfigInfoApi<?, ?> xcodeConfigInfo(
        String iosSdkVersion,
        String iosMinimumOsVersion,
        String watchosSdkVersion,
        String watchosMinimumOsVersion,
        String tvosSdkVersion,
        String tvosMinimumOsVersion,
        String macosSdkVersion,
        String macosMinimumOsVersion,
        String xcodeVersion)
        throws EvalException;
  }
}
