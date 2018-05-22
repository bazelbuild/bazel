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
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * Runfiles provider for C++ targets.
 *
 * <p>Contains two {@link Runfiles} objects: one for the eventual statically linked binary and one
 * for the one that uses shared libraries. Data dependencies are present in both.
 */
@Immutable
@AutoCodec
public final class CcRunfiles {

  private final Runfiles dynamicLibrariesForLinkingStatically;
  private final Runfiles dynamicLibrariesForLinkingDynamically;

  @AutoCodec.Instantiator
  public CcRunfiles(
      Runfiles dynamicLibrariesForLinkingStatically,
      Runfiles dynamicLibrariesForLinkingDynamically) {
    this.dynamicLibrariesForLinkingStatically = dynamicLibrariesForLinkingStatically;
    this.dynamicLibrariesForLinkingDynamically = dynamicLibrariesForLinkingDynamically;
  }

  public Runfiles getDynamicLibrariesForLinkingStatically() {
    return dynamicLibrariesForLinkingStatically;
  }

  public Runfiles getDynamicLibrariesForLinkingDynamically() {
    return dynamicLibrariesForLinkingDynamically;
  }

  /**
   * Returns a function that gets the static C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles>
      DYNAMIC_LIBRARIES_FOR_LINKING_STATICALLY =
          input -> {
            CcLinkingInfo provider = input.get(CcLinkingInfo.PROVIDER);
            CcRunfiles ccRunfiles = provider == null ? null : provider.getCcRunfiles();
            return ccRunfiles == null
                ? Runfiles.EMPTY
                : ccRunfiles.getDynamicLibrariesForLinkingStatically();
          };

  /**
   * Returns a function that gets the shared C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles>
      DYNAMIC_LIBRARIES_FOR_LINKING_DYNAMICALLY =
          input -> {
            CcLinkingInfo provider = input.get(CcLinkingInfo.PROVIDER);
            CcRunfiles ccRunfiles = provider == null ? null : provider.getCcRunfiles();
            return ccRunfiles == null
                ? Runfiles.EMPTY
                : ccRunfiles.getDynamicLibrariesForLinkingDynamically();
          };

  /**
   * Returns a function that gets the C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> runfilesFunction(
      boolean linkingStatically) {
    return linkingStatically
        ? DYNAMIC_LIBRARIES_FOR_LINKING_STATICALLY
        : DYNAMIC_LIBRARIES_FOR_LINKING_DYNAMICALLY;
  }
}
