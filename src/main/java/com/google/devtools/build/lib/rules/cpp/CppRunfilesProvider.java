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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Runfiles provider for C++ targets.
 *
 * <p>Contains two {@link Runfiles} objects: one for the eventual statically linked binary and
 * one for the one that uses shared libraries. Data dependencies are present in both.
 */
@Immutable
public final class CppRunfilesProvider implements TransitiveInfoProvider {
  private final Runfiles staticRunfiles;
  private final Runfiles sharedRunfiles;

  public CppRunfilesProvider(Runfiles staticRunfiles, Runfiles sharedRunfiles) {
    this.staticRunfiles = staticRunfiles;
    this.sharedRunfiles = sharedRunfiles;
  }

  public Runfiles getStaticRunfiles() {
    return staticRunfiles;
  }

  public Runfiles getSharedRunfiles() {
    return sharedRunfiles;
  }

  /**
   * Returns a function that gets the static C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> STATIC_RUNFILES =
      input -> {
        CppRunfilesProvider provider = input.getProvider(CppRunfilesProvider.class);
        return provider == null ? Runfiles.EMPTY : provider.getStaticRunfiles();
      };

  /**
   * Returns a function that gets the shared C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> SHARED_RUNFILES =
      input -> {
        CppRunfilesProvider provider = input.getProvider(CppRunfilesProvider.class);
        return provider == null ? Runfiles.EMPTY : provider.getSharedRunfiles();
      };

  /**
   * Returns a function that gets the C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> runfilesFunction(
      boolean linkingStatically) {
    return linkingStatically ? STATIC_RUNFILES : SHARED_RUNFILES;
  }
}
