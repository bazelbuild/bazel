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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Supplies the device broker type string, passed to the Android test runtime. */
@SkylarkModule(
    name = "DeviceBrokerInfo",
    doc = "Information about the device broker",
    category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public final class DeviceBrokerInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "DeviceBrokerInfo";
  public static final NativeProvider<DeviceBrokerInfo> PROVIDER =
      new NativeProvider<DeviceBrokerInfo>(DeviceBrokerInfo.class, SKYLARK_NAME) {};

  private final String deviceBrokerType;

  public DeviceBrokerInfo(String deviceBrokerType) {
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
