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
package com.google.devtools.build.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import java.util.Collection;
import java.util.Map;

/**
 * Immutable implementation of {@link Differencer.Diff}.
 */
public class ImmutableDiff implements Differencer.Diff {

  private final ImmutableList<SkyKey> valuesToInvalidate;
  private final ImmutableMap<SkyKey, Delta> valuesToInject;

  public ImmutableDiff(Iterable<SkyKey> valuesToInvalidate, Map<SkyKey, Delta> valuesToInject) {
    this.valuesToInvalidate = ImmutableList.copyOf(valuesToInvalidate);
    this.valuesToInject = ImmutableMap.copyOf(valuesToInject);
  }

  @Override
  public Collection<SkyKey> changedKeysWithoutNewValues() {
    return valuesToInvalidate;
  }

  @Override
  public Map<SkyKey, Delta> changedKeysWithNewValues() {
    return valuesToInject;
  }
}
