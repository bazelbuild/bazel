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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidDex2OatInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;

/**
 * Supplies the pregenerate_oat_files_for_tests attribute of type boolean provided by android_device
 * rule.
 */
@Immutable
public final class AndroidDex2OatInfo extends NativeInfo implements AndroidDex2OatInfoApi {

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 0,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly=*/ 1,
              /*starArg=*/ false,
              /*kwArg=*/ false,
              "enabled"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.of(SkylarkType.of(Boolean.class))); // instrumentation_apk
  public static final NativeProvider<AndroidDex2OatInfo> PROVIDER =
      new NativeProvider<AndroidDex2OatInfo>(AndroidDex2OatInfo.class, NAME, SIGNATURE) {
        @Override
        protected AndroidDex2OatInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) {
          return new AndroidDex2OatInfo(/*dex2OatEnabled=*/ (Boolean) args[0]);
        }
      };

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
