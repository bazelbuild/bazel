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

import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * An interface for an info type containing the set of Apple versions computed from command line
 * options and the {@code xcode_config} rule.
 */
@SkylarkModule(
    name = "XcodeVersionConfig",
    doc = "The set of Apple versions computed from command line options and the xcode_config rule.")
public interface XcodeConfigProviderApi<ApplePlatformApiT extends ApplePlatformApi,
    ApplePlatformTypeApiT extends ApplePlatformTypeApi> extends StructApi {

  @SkylarkCallable(name = "xcode_version",
      doc = "Returns the Xcode version that is being used to build.<p>"
          + "This will return <code>None</code> if no Xcode versions are available.",
      allowReturnNones = true)
  public DottedVersionApi<?> getXcodeVersion();

  @SkylarkCallable(
      name = "minimum_os_for_platform_type",
      doc = "The minimum compatible OS version for target simulator and devices for a particular "
          + "platform type.",
      parameters = {
        @Param(
            name = "platform_type",
            positional = true,
            named = false,
            type = ApplePlatformTypeApi.class,
            doc = "The apple platform type."
        ),
      }
  )
  public DottedVersionApi<?> getMinimumOsForPlatformType(ApplePlatformTypeApiT platformType);

  @SkylarkCallable(
      name = "sdk_version_for_platform",
      doc = "The version of the platform SDK that will be used to build targets for the given "
          + "platform.",
      parameters = {
        @Param(
            name = "platform",
            positional = true,
            named = false,
            type = ApplePlatformApi.class,
            doc = "The apple platform."
        ),
      })
  public DottedVersionApi<?> getSdkVersionForPlatform(ApplePlatformApiT platform);
}
