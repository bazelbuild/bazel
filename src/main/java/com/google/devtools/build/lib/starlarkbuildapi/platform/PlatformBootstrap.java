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

package com.google.devtools.build.lib.starlarkbuildapi.platform;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkbuildapi.core.Bootstrap;

/** {@link Bootstrap} for Starlark objects related to platforms. */
public class PlatformBootstrap<IncompatiblePlatformProviderApiT extends IncompatiblePlatformProviderApi> implements Bootstrap {

  private final PlatformCommonApi<IncompatiblePlatformProviderApiT> platformCommon;

  public PlatformBootstrap(PlatformCommonApi<IncompatiblePlatformProviderApiT> platformCommon) {
    this.platformCommon = platformCommon;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put("platform_common", platformCommon);
  }
}
