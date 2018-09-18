// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.rules.cpp.CcLinkingInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidCcLinkParamsProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A target that provides C++ libraries to be linked into Android targets. */
@Immutable
public final class AndroidCcLinkParamsProvider extends NativeInfo
    implements AndroidCcLinkParamsProviderApi<CcLinkingInfo> {
  public static final String PROVIDER_NAME = "AndroidCcLinkParamsInfo";
  public static final Provider PROVIDER = new Provider();

  private final CcLinkingInfo ccLinkingInfo;

  public AndroidCcLinkParamsProvider(CcLinkingInfo ccLinkingInfo) {
    super(PROVIDER);
    this.ccLinkingInfo = ccLinkingInfo;
  }

  @Override
  public CcLinkingInfo getLinkParams() {
    return ccLinkingInfo;
  }

  /** Provider class for {@link AndroidCcLinkParamsProvider} objects. */
  public static class Provider extends BuiltinProvider<AndroidCcLinkParamsProvider>
      implements AndroidCcLinkParamsProviderApi.Provider<CcLinkingInfo> {
    private Provider() {
      super(PROVIDER_NAME, AndroidCcLinkParamsProvider.class);
    }

    @Override
    public AndroidCcLinkParamsProviderApi<CcLinkingInfo> createInfo(CcLinkingInfo ccLinkingInfo)
        throws EvalException {
      return new AndroidCcLinkParamsProvider(ccLinkingInfo);
    }
  }
}
