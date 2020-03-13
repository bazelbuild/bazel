// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCcLinkParamsProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A target that provides C++ libraries to be linked into Java targets. */
@Immutable
public final class JavaCcLinkParamsProvider
    implements Info, JavaCcLinkParamsProviderApi<Artifact, CcInfo> {
  public static final String PROVIDER_NAME = "JavaCcLinkParamsInfo";
  public static final Provider PROVIDER = new Provider();

  private final CcInfo ccInfo;

  public JavaCcLinkParamsProvider(CcInfo ccInfo) {
    this.ccInfo = CcInfo.builder().setCcLinkingContext(ccInfo.getCcLinkingContext()).build();
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public CcInfo getCcInfo() {
    return ccInfo;
  }

  /** Provider class for {@link JavaCcLinkParamsProvider} objects. */
  public static class Provider extends BuiltinProvider<JavaCcLinkParamsProvider>
      implements JavaCcLinkParamsProviderApi.Provider<Artifact, CcInfo> {
    private Provider() {
      super(PROVIDER_NAME, JavaCcLinkParamsProvider.class);
    }

    @Override
    public JavaCcLinkParamsProviderApi<Artifact, CcInfo> createInfo(CcInfo ccInfo)
        throws EvalException {
      return new JavaCcLinkParamsProvider(ccInfo);
    }
  }
}
