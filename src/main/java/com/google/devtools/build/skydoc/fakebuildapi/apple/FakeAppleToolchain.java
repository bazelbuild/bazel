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

package com.google.devtools.build.skydoc.fakebuildapi.apple;

import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleConfigurationApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ApplePlatformTypeApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleToolchainApi;

/**
 * Fake implementation of {@link AppleToolchainApi}.
 */
public class FakeAppleToolchain
    implements AppleToolchainApi<AppleConfigurationApi<ApplePlatformTypeApi>> {

  @Override
  public String sdkDirConstant() {
    return "";
  }

  @Override
  public String developerDirConstant() {
    return "";
  }

  @Override
  public String platformFrameworkDirFromConfig(
      AppleConfigurationApi<ApplePlatformTypeApi> configuration) {
    return "";
  }
}
