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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * Interface for a value representing a version with multiple components, separated by periods, such
 * as "1.2.3.4".
 */
@SkylarkModule(
    name = "DottedVersion",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A value representing a version with multiple components, separated by periods, such as "
            + "1.2.3.4.")
public interface DottedVersionApi<SelfT extends DottedVersionApi<?>>
    extends StarlarkValue, Comparable<SelfT> {

  @SkylarkCallable(
      name = "compare_to",
      doc =
          "Compares based on most signifigant (first) not-matching version component. "
              + "So, for example, 1.2.3 < 1.2.4",
      parameters = {
        @Param(
            name = "other",
            positional = true,
            named = false,
            type = DottedVersionApi.class,
            doc = "The other dotted version.")
      })
  int compareTo_skylark(SelfT other);
}
