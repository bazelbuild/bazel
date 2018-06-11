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
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidDeviceBrokerInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** Supplies the device broker type string, passed to the Android test runtime. */
@Immutable
public final class AndroidDeviceBrokerInfo extends NativeInfo
    implements AndroidDeviceBrokerInfoApi {

  private static final String SKYLARK_NAME = "AndroidDeviceBrokerInfo";

  /**
   * Provider instance for {@link AndroidDeviceBrokerInfo}.
   */
  public static final AndroidDeviceBrokerInfoProvider PROVIDER =
      new AndroidDeviceBrokerInfoProvider();

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

  /** Provider for {@link AndroidDeviceBrokerInfo}. */
  public static class AndroidDeviceBrokerInfoProvider
      extends BuiltinProvider<AndroidDeviceBrokerInfo>
      implements AndroidDeviceBrokerInfoApiProvider {

    private AndroidDeviceBrokerInfoProvider() {
      super(SKYLARK_NAME, AndroidDeviceBrokerInfo.class);
    }

    @Override
    public AndroidDeviceBrokerInfo createInfo(String type)
        throws EvalException {
      return new AndroidDeviceBrokerInfo(type);
    }
  }
}
