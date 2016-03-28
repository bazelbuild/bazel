// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.truth.FailureStrategy;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import com.google.common.truth.Truth;

import javax.annotation.Nullable;

/**
 * {@link Subject} for {@link CycleInfo}. Please add to this class if you need more functionality!
 */
public class CycleInfoSubject extends Subject<CycleInfoSubject, CycleInfo> {
  CycleInfoSubject(FailureStrategy failureStrategy, @Nullable CycleInfo cycleInfo) {
    super(failureStrategy, cycleInfo);
  }

  public IterableSubject hasPathToCycleThat() {
    return Truth.assertThat(getSubject().getPathToCycle())
        .named("Path to cycle in " + getDisplaySubject());
  }

  public IterableSubject hasCycleThat() {
    return Truth.assertThat(getSubject().getCycle()).named("Cycle in " + getDisplaySubject());
  }
}
