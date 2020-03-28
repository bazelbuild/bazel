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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.PyCcLinkParamsProviderApi;

/** A target that provides C++ libraries to be linked into Python targets. */
@Immutable
@AutoCodec
public final class PyCcLinkParamsProvider extends NativeInfo
    implements PyCcLinkParamsProviderApi<Artifact> {
  public static final Provider PROVIDER = new Provider();

  private final CcInfo ccInfo;

  public PyCcLinkParamsProvider(CcInfo ccInfo) {
    super(PROVIDER);
    this.ccInfo = CcInfo.builder().setCcLinkingContext(ccInfo.getCcLinkingContext()).build();
  }

  @Override
  public CcInfo getCcInfo() {
    return ccInfo;
  }

  /** Provider class for {@link PyCcLinkParamsProvider} objects. */
  public static class Provider extends BuiltinProvider<PyCcLinkParamsProvider>
      implements PyCcLinkParamsProviderApi.Provider {
    private Provider() {
      super("PyCcLinkParamsProvider", PyCcLinkParamsProvider.class);
    }
  }
}
