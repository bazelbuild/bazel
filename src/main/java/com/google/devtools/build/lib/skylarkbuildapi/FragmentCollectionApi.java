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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.ClassObject;

/** Represents a collection of configuration fragments in Skylark. */
@SkylarkModule(
    name = "fragments",
    category = SkylarkModuleCategory.NONE,
    doc =
        "Possible fields are "
            + "<a href=\"android.html\">android</a>, <a href=\"apple.html\">apple</a>, "
            + "<a href=\"cpp.html\">cpp</a>, <a href=\"java.html\">java</a>, "
            + "<a href=\"jvm.html\">jvm</a>, and <a href=\"objc.html\">objc</a>. "
            + "Access a specific fragment by its field name "
            + "ex:</p><code>ctx.fragments.apple</code></p>"
            + "Note that rules have to declare their required fragments in order to access them "
            + "(see <a href=\"../rules.md#fragments\">here</a>).")
public interface FragmentCollectionApi extends ClassObject {}
