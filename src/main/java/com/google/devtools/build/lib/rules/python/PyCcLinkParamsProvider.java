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

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore.CcLinkParamsStoreImpl;

/**
 * A target that provides C++ libraries to be linked into Python targets.
 */
@Immutable
public final class PyCcLinkParamsProvider implements TransitiveInfoProvider {
  private final CcLinkParamsStoreImpl store;

  public PyCcLinkParamsProvider(CcLinkParamsStore store) {
    this.store = new CcLinkParamsStoreImpl(store);
  }

  public CcLinkParamsStore getLinkParams() {
    return store;
  }

  public static final Function<TransitiveInfoCollection, CcLinkParamsStore> TO_LINK_PARAMS =
      input -> {
        PyCcLinkParamsProvider provider = input.getProvider(PyCcLinkParamsProvider.class);
        return provider == null ? null : provider.getLinkParams();
      };
}
