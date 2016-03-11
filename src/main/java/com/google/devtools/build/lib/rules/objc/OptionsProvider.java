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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Provides information contained in a {@code objc_options} target.
 */
@Immutable
final class OptionsProvider
    extends Value<OptionsProvider>
    implements TransitiveInfoProvider {

  private final Iterable<String> copts;

  public OptionsProvider(Iterable<String> copts) {
    super(copts);
    this.copts = Preconditions.checkNotNull(copts);
  }

  public Iterable<String> getCopts() {
    return copts;
  }
}
