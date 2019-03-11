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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** No-op configuration transition. */
public final class NoTransition implements PatchTransition {

  @AutoCodec public static final NoTransition INSTANCE = new NoTransition();

  private NoTransition() {}

  @Override
  public BuildOptions patch(BuildOptions options) {
    return options;
  }

  public static <T extends TransitionFactoryData> TransitionFactory<T> createFactory() {
    return new AutoValue_NoTransition_NoTransitionFactory<>();
  }

  @AutoValue
  public abstract static class NoTransitionFactory<T extends TransitionFactoryData>
      implements TransitionFactory<T> {
    @Override
    public ConfigurationTransition create(TransitionFactoryData unused) {
      return INSTANCE;
    }
  }
}
