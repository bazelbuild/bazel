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

package com.google.devtools.build.lib.skylarkbuildapi.core;

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Interface for a build target. */
@SkylarkModule(
    name = "Target",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A BUILD target. It is essentially a <code>struct</code> with the following fields:"
            + "<ul><li><h3 id=\"modules.Target.label\">label</h3><code><a class=\"anchor\""
            + " href=\"Label.html\">Label</a> Target.label</code><br>The identifier of the "
            + "target.</li><li><h3 id=\"modules.Target.files\">files</h3><code><a class=\"anchor\""
            + " href=\"depset.html\">depset</a> Target.files </code><br>The set of <a"
            + " class=\"anchor\" href=\"File.html\">File</a>s produced directly by this "
            + "target.</li><li><h3 id=\"modules.Target.aspect_ids\">aspect_ids</h3><code><a"
            + " class=\"anchor\"href=\"list.html\">list</a> Target.aspect_ids </code><br>The list"
            + " of <a class=\"anchor\" href=\"ctx.html#aspect_id\">aspect_id</a>s applied to this "
            + "target.</li><li><h3 id=\"modules.Target.extraproviders\">Extra providers</h3>For"
            + " rule targets all additional providers provided by this target are accessible as"
            + " <code>struct</code> fields. These extra providers are defined in the"
            + " <code>struct</code> returned by the rule implementation function.</li></ul>")
public interface TransitiveInfoCollectionApi extends StarlarkValue {
}
