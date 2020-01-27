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

import java.io.Serializable;

/** A return value of a {@code SkyFunction}. */
public interface SkyValue extends Serializable {

  /**
   * Returns true for values that can be reused across builds. Some values are inherently "flaky",
   * like test statuses or stamping information, and in certain circumstances, those values cannot
   * be shared across builds/servers.
   */
  default boolean dataIsShareable() {
    return true;
  }
}
