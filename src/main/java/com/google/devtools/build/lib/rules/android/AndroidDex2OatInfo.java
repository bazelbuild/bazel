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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidDex2OatInfoApi;

/**
 * Supplies the pregenerate_oat_files_for_tests attribute of type boolean provided by android_device
 * rule.
 */
@Immutable
public final class AndroidDex2OatInfo extends NativeInfo implements AndroidDex2OatInfoApi {

  public static final BuiltinProvider<AndroidDex2OatInfo> PROVIDER = new Provider();

  /** Provider for {@link AndroidDex2OatInfo} objects. */
  private static class Provider extends BuiltinProvider<AndroidDex2OatInfo>
      implements AndroidDex2OatInfoApi.Provider {
    public Provider() {
      super(NAME, AndroidDex2OatInfo.class);
    }

    @Override
    public AndroidDex2OatInfo androidDex2OatInfo(Boolean enabled) {
      return new AndroidDex2OatInfo(enabled);
    }
  }

  private final boolean dex2OatEnabled;

  public AndroidDex2OatInfo(boolean dex2OatEnabled) {
    super(PROVIDER);
    this.dex2OatEnabled = dex2OatEnabled;
  }

  /** Returns if the device should run cloud dex2oat. */
  public boolean isDex2OatEnabled() {
    return dex2OatEnabled;
  }
}
