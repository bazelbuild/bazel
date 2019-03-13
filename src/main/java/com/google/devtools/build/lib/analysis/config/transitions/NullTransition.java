// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A {@link PatchTransition} to a null configuration. */
public class NullTransition implements PatchTransition {

  @AutoCodec public static final NullTransition INSTANCE = new NullTransition();

  private NullTransition() {
  }

  @Override
  public BuildOptions patch(BuildOptions options) {
    throw new UnsupportedOperationException(
        "This is only referenced in a few places, so it's easier and more efficient to optimize "
            + "Blaze's transition logic in the presence of null transitions vs. actually call this "
            + "method to get results we know ahead of time. If there's ever a need to properly "
            + "implement this method we can always do so.");
  }
}
