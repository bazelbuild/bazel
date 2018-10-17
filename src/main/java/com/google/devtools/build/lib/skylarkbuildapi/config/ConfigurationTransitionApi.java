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

package com.google.devtools.build.lib.skylarkbuildapi.config;

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;

/**
 * Represents a configuration transition across a dependency edge.
 */
@SkylarkModule(
    name = "transition",
    doc = "<b>Experimental. This type is experimental and subject to change at any time. Do "
        + "not depend on it.</b><p> "
        + "Represents a configuration transition across a dependency edge. For example, if "
        + "<code>//package:foo</code> depends on <code>//package:bar</code> with a configuration "
        + "transition, then the configuration of these two targets will differ: "
        + "<code>//package:bar</code>'s transition will be determined by that of "
        + "<code>//package:foo</code>, as subject to the function defined by a transition object."
)
public interface ConfigurationTransitionApi extends SkylarkValue {}
