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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;

/** Supplies the device broker type string, passed to the Android test runtime. */
@SkylarkModule(name = "AndroidDeviceBrokerInfo", doc = "", documented = false)
@Immutable
public final class AndroidDeviceBrokerInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "AndroidDeviceBrokerInfo";
  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 0,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly=*/ 1,
              /*starArg=*/ false,
              /*kwArg=*/ false,
              "type"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.of(SkylarkType.of(String.class))); // instrumentation_apk
  public static final NativeProvider<AndroidDeviceBrokerInfo> PROVIDER =
      new NativeProvider<AndroidDeviceBrokerInfo>(
          AndroidDeviceBrokerInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        protected AndroidDeviceBrokerInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) {
          return new AndroidDeviceBrokerInfo(/*deviceBrokerType=*/ (String) args[0]);
        }
      };

  private final String deviceBrokerType;

  public AndroidDeviceBrokerInfo(String deviceBrokerType) {
    super(PROVIDER);
    this.deviceBrokerType = deviceBrokerType;
  }

  /**
   * Returns the type of device broker that is appropriate to use to interact with devices obtained
   * by this artifact.
   */
  public String getDeviceBrokerType() {
    return deviceBrokerType;
  }
}
