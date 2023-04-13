// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.proto;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/**
 * Interface for protocol buffers support in Bazel.
 *
 * <p>Currently just a placeholder so that we have a {@code proto_common} symbol.
 */
@StarlarkBuiltin(
    name = "proto_common",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc =
        "Utilities for protocol buffers. <p>Please consider using"
            + " <code>load(\"@rules_proto//proto:defs.bzl\", \"proto_common\")</code> to load this"
            + " symbol from <a href=\"https://github.com/bazelbuild/rules_proto\">rules_proto</a>."
            + "</p>")
public interface ProtoCommonApi extends StarlarkValue {}
