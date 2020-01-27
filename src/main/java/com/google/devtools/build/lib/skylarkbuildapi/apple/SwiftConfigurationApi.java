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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** A configuration fragment for Swift tools. */
@SkylarkModule(
    name = "swift",
    doc = "A configuration fragment for Swift tools.",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT)
public interface SwiftConfigurationApi extends StarlarkValue {

  @SkylarkCallable(
      name = "copts",
      doc =
          "A list of compiler options that should be passed to <code>swiftc</code> when compiling "
              + "Swift code.")
  ImmutableList<String> getCopts();
}
