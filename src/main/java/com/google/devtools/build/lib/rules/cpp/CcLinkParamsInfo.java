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
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore.CcLinkParamsStoreImpl;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/** A target that provides C linker parameters. */
@Immutable
@AutoCodec
public final class CcLinkParamsInfo {
  public static final Function<TransitiveInfoCollection, CcLinkParamsStore> TO_LINK_PARAMS =
      input -> {
        // ... then try Skylark.
        CcLinkingInfo provider = input.get(CcLinkingInfo.PROVIDER);
        CcLinkParamsInfo ccLinkParamsInfo =
            provider == null ? null : provider.getCcLinkParamsInfo();
        if (ccLinkParamsInfo != null) {
          return ccLinkParamsInfo.getCcLinkParamsStore();
        }
        return null;
      };

  private final CcLinkParamsStoreImpl store;

  @AutoCodec.Instantiator
  public CcLinkParamsInfo(CcLinkParamsStore store) {
    this.store = new CcLinkParamsStoreImpl(store);
  }

  @AutoCodec
  @VisibleForSerialization
  static class CcLinkParamsInfoCollection extends CcLinkParamsStore {
    private final Iterable<CcLinkParamsInfo> providers;

    CcLinkParamsInfoCollection(Iterable<CcLinkParamsInfo> providers) {
      this.providers = providers;
    }

    @Override
    protected void collect(
        CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
      for (CcLinkParamsInfo provider : providers) {
        builder.add(provider.getCcLinkParamsStore());
      }
    }
  }

  public static CcLinkParamsInfo merge(final Iterable<CcLinkParamsInfo> providers) {
    return new CcLinkParamsInfo(new CcLinkParamsInfoCollection(providers));
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
