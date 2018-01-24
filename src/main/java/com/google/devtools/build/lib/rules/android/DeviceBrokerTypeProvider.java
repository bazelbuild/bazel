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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** Supplies the device broker type string, passed to the android_test runtime. */
@Immutable
public final class DeviceBrokerTypeProvider implements TransitiveInfoProvider {

  private final String deviceBrokerType;

  public DeviceBrokerTypeProvider(String deviceBrokerType) {
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
