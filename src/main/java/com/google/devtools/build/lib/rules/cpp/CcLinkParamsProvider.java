// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

/**
 * A target that provides C linker parameters.
 */
@Immutable
public final class CcLinkParamsProvider implements TransitiveInfoProvider {

  private final CcLinkParams staticSharedParams;
  private final CcLinkParams staticNonSharedParams;
  private final CcLinkParams nonStaticSharedParams;
  private final CcLinkParams nonStaticNonSharedParams;

  public CcLinkParamsProvider(CcLinkParamsStore ccLinkParamsStore) {
    this.staticSharedParams = ccLinkParamsStore.get(true, true);
    this.staticNonSharedParams = ccLinkParamsStore.get(true, false);
    this.nonStaticSharedParams = ccLinkParamsStore.get(false, true);
    this.nonStaticNonSharedParams = ccLinkParamsStore.get(false, false);
  }

  public CcLinkParamsProvider(
      CcLinkParams staticSharedParams,
      CcLinkParams staticNonSharedParams,
      CcLinkParams nonStaticSharedParams,
      CcLinkParams nonStaticNonSharedParams) {
    this.staticSharedParams = staticSharedParams;
    this.staticNonSharedParams = staticNonSharedParams;
    this.nonStaticSharedParams = nonStaticSharedParams;
    this.nonStaticNonSharedParams = nonStaticNonSharedParams;
  }

  /**
   * Returns link parameters given static / shared linking settings.
   */
  public CcLinkParams getCcLinkParams(boolean linkingStatically, boolean linkShared) {
    if (linkingStatically) {
      if (linkShared) {
        return staticSharedParams;
      } else {
        return staticNonSharedParams;
      }
    } else {
      if (linkShared) {
        return nonStaticSharedParams;
      } else {
        return nonStaticNonSharedParams;
      }
    }
  }
}
