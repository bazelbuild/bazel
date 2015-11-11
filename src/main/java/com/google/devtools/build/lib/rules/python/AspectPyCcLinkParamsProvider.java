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
package com.google.devtools.build.lib.rules.python;

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;

/**
 * Wrapper around PyCcLinkParamsProvider, to allow PythonProtoAspect to add Providers to
 * proto_library rules with py_api_version. If PythonProtoAspect provides PyCcLinkParamsProvider
 * directly on such a proto_library rule, Bazel crashes with
 *
 *     Provider class PyCcLinkParamsProvider provided twice
 */
@ThreadSafety.Immutable
public final class AspectPyCcLinkParamsProvider implements TransitiveInfoProvider {
  public final PyCcLinkParamsProvider provider;
  public AspectPyCcLinkParamsProvider(PyCcLinkParamsProvider provider) {
    this.provider = provider;
  }

  public static final Function<TransitiveInfoCollection, CcLinkParamsStore> TO_LINK_PARAMS =
      new Function<TransitiveInfoCollection, CcLinkParamsStore>() {
        @Override
        public CcLinkParamsStore apply(TransitiveInfoCollection input) {
          AspectPyCcLinkParamsProvider wrapper = input.getProvider(
              AspectPyCcLinkParamsProvider.class);
          return wrapper == null ? null : wrapper.provider.getLinkParams();
        }
      };
}
