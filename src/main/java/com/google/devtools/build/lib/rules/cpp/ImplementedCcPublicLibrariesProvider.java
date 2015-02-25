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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;

/**
 * A provider for cc_library rules to be able to convey the information about which
 * cc_public_library rules they implement to dependent targets.
 */
@Immutable
public final class ImplementedCcPublicLibrariesProvider implements TransitiveInfoProvider {

  private final ImmutableList<Label> implementedCcPublicLibraries;

  public ImplementedCcPublicLibrariesProvider(ImmutableList<Label> implementedCcPublicLibraries) {
    this.implementedCcPublicLibraries = implementedCcPublicLibraries;
  }

  /**
   * Returns the labels for the "$headers" target that are implemented by the target which
   * implements this interface.
   */
  public ImmutableList<Label> getImplementedCcPublicLibraries() {
    return implementedCcPublicLibraries;
  }
}
