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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore.CcLinkParamsStoreImpl;

/** A target that provides C linker parameters. */
@Immutable
public final class CcLinkParamsProvider extends SkylarkClassObject
    implements TransitiveInfoProvider {
  public static final NativeClassObjectConstructor<CcLinkParamsProvider> CC_LINK_PARAMS =
      new NativeClassObjectConstructor<CcLinkParamsProvider>(
          CcLinkParamsProvider.class, "link_params") {};
  public static final Function<TransitiveInfoCollection, CcLinkParamsStore> TO_LINK_PARAMS =
      input -> {

        // Try native first...
        CcLinkParamsProvider provider = input.getProvider(CcLinkParamsProvider.class);
        if (provider != null) {
          return provider.getCcLinkParamsStore();
        }

        // ... then try Skylark.
        provider = input.get(CC_LINK_PARAMS);
        if (provider != null) {
          return provider.getCcLinkParamsStore();
        }
        return null;
      };

  private final CcLinkParamsStoreImpl store;

  public CcLinkParamsProvider(CcLinkParamsStore store) {
    super(CC_LINK_PARAMS, ImmutableMap.<String, Object>of());
    this.store = new CcLinkParamsStoreImpl(store);
  }

  public static CcLinkParamsProvider merge(final Iterable<CcLinkParamsProvider> providers) {
    CcLinkParamsStore ccLinkParamsStore =
        new CcLinkParamsStore() {
          @Override
          protected void collect(
              CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
            for (CcLinkParamsProvider provider : providers) {
              builder.add(provider.getCcLinkParamsStore());
            }
          }
        };
    return new CcLinkParamsProvider(ccLinkParamsStore);
  }

  /** Returns the link params store. */
  public CcLinkParamsStore getCcLinkParamsStore() {
    return store;
  }

  /**
   * Returns link parameters given static / shared linking settings.
   */
  public CcLinkParams getCcLinkParams(boolean linkingStatically, boolean linkShared) {
    return store.get(linkingStatically, linkShared);
  }
}
