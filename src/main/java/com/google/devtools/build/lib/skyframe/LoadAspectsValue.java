// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * {@link SkyValue} for {@code LoadAspectsKey} wraps a list of the {@code Aspect} of the top level
 * aspects.
 */
public final class LoadAspectsValue implements SkyValue {
  private final ImmutableList<Aspect> aspects;

  LoadAspectsValue(Collection<Aspect> aspects) {
    this.aspects = ImmutableList.copyOf(aspects);
  }

  public ImmutableList<Aspect> getAspects() {
    return aspects;
  }
}
