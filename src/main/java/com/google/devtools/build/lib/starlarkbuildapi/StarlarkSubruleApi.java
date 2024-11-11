// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import java.util.List;
import java.util.Optional;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** The interface for Starlark-defined subrules in the Build API. */
@StarlarkBuiltin(
    name = "Subrule",
    category = DocCategory.BUILTIN,
    doc =
        "Experimental: a building block for writing rules with shared code. For more information,"
            + " please see the subrule proposal:"
            + " https://docs.google.com/document/d/1RbNC88QieKvBEwir7iV5zZU08AaMlOzxhVkPnmKDedQ")
public interface StarlarkSubruleApi extends StarlarkValue {

  static Optional<String> getUserDefinedNameIfSubruleAttr(
      List<? extends StarlarkSubruleApi> subrules, String attributeName) {
    return subrules.stream()
        .map(s -> s.getUserDefinedNameIfSubruleAttr(attributeName))
        .flatMap(Optional::stream)
        .findFirst();
  }

  Optional<String> getUserDefinedNameIfSubruleAttr(String attrName);
}
