// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;

/** Supplies the pregenerate_oat_files_for_tests attribute of type boolean provided by
 * android_device rule.
 */
public class Dex2OatProvider implements TransitiveInfoProvider {

  private final boolean cloudDex2oatEnabled;

  public Dex2OatProvider(boolean cloudDex2oatEnabled) {
    this.cloudDex2oatEnabled = cloudDex2oatEnabled;
  }

  /**
   * Returns if the device should run cloud dex2oat.
   */
  public boolean isCloudDex2oatEnabled() {
    return cloudDex2oatEnabled;
  }
}
