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

package com.google.devtools.build.lib.starlarkbuildapi.core;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a build target. */
@StarlarkBuiltin(
    name = "Target",
    category = DocCategory.BUILTIN,
    doc =
        "The BUILD target for a dependency. Appears in the fields of <code><a"
            + " href='../builtins/ctx.html#attr'>ctx.attr</a></code> corresponding to <a"
            + " href='https://bazel.build/extending/rules#dependency_attributes'>dependency"
            + " attributes</a> (<code><a href='../toplevel/attr.html#label'>label</a></code> or"
            + " <code><a href='../toplevel/attr.html#label_list'>label_list</a></code>). Has the"
            + " following fields:\n"
            //
            + "<ul>\n" //
            + "<li><h3 id='modules.Target.label'>label</h3>\n" //
            + "<code><a href='../builtins/Label.html'>Label</a> Target.label</code><br/>\n" //
            + "The identifier of the target.</li>\n" //
            //
            + "<li><h3 id='modules.Target.files'>files</h3>\n" //
            + "<code><a href='../builtins/depset.html'>depset</a> Target.files </code><br/>\n" //
            + "The set of <code><a href='../builtins/File.html'>File</a></code>s in the default"
            + " outputs for this target. Equivalent to <code><a"
            + " href='../providers/DefaultInfo.html#files'>target[DefaultInfo].files</a></code>.</li>\n" //
            //
            + "<li><h3 id='modules.Target.aspect_ids'>aspect_ids</h3>\n" //
            + "<code><a href='../core/list.html'>list</a> Target.aspect_ids </code><br/>\n" //
            + "The list of <code><a href='../builtins/ctx.html#aspect_ids'>aspect_ids</a></code> "
            + "applied to this target.</li>\n" //
            //
            + "<li><h3 id='modules.Target.providers'>Providers</h3>\n" //
            + "The <a href='https://bazel.build/extending/rules#providers'>providers</a> of a rule"
            + " target can be accessed by type using index notation"
            + " (<code>target[DefaultInfo]</code>). The presence of providers can be checked using"
            + " the <code>in</code> operator (<code>SomeInfo in target</code>).<br/>\n" //
            + "<br/>\n" //
            + "If the rule's implementation function returns a <code><a"
            + " href='../builtins/struct.html'>struct</a></code> instead of a list of <code><a"
            + " href='../builtins/Provider.html'>Provider</a></code> instances, the struct's fields"
            + " can be accessed via the corresponding fields of the <code>Target</code>"
            + " (<code>target.some_legacy_info</code>). This behavior <a"
            + " href='https://bazel.build/extending/rules#migrating_from_legacy_providers'>is"
            + " deprecated</a>.</li>\n" //
            + "</ul>")
public interface TransitiveInfoCollectionApi extends StarlarkValue {}
