// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.skyframe.SkyValue;

/** The result of evaluating a {@link BuildDriverKey}. */
public class BuildDriverValue implements SkyValue {
  private final SkyValue wrappedSkyValue;
  private final boolean skipped;

  BuildDriverValue(SkyValue wrappedSkyValue, boolean skipped) {
    this.wrappedSkyValue = wrappedSkyValue;
    this.skipped = skipped;
  }

  public SkyValue getWrappedSkyValue() {
    return wrappedSkyValue;
  }

  public boolean isSkipped() {
    return skipped;
  }
}
