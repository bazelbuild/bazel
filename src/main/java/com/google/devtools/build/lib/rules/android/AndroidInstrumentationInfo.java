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
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidInstrumentationInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * A provider for targets that create Android instrumentations. Consumed by Android testing rules.
 */
@Immutable
public class AndroidInstrumentationInfo extends NativeInfo
    implements AndroidInstrumentationInfoApi<ApkInfo> {

  private static final String STARLARK_NAME = "AndroidInstrumentationInfo";

  public static final AndroidInstrumentationInfoProvider PROVIDER =
      new AndroidInstrumentationInfoProvider();

  private final ApkInfo target;

  AndroidInstrumentationInfo(ApkInfo target) {
    super(PROVIDER);
    this.target = target;
  }

  @Override
  public ApkInfo getTarget() {
    return target;
  }

  /** Provider for {@link AndroidInstrumentationInfo}. */
  public static class AndroidInstrumentationInfoProvider
      extends BuiltinProvider<AndroidInstrumentationInfo>
      implements AndroidInstrumentationInfoApiProvider<ApkInfo> {

    private AndroidInstrumentationInfoProvider() {
      super(STARLARK_NAME, AndroidInstrumentationInfo.class);
    }

    @Override
    public AndroidInstrumentationInfoApi<ApkInfo> createInfo(ApkInfo target) throws EvalException {
      return new AndroidInstrumentationInfo(target);
    }
  }
}
