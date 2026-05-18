// Copyright 2025 The Bazel Authors. All rights reserved.
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

import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/**
 * Marker interface for {@code select()} expressions in the Starlark build API.
 *
 * <p>Implemented by {@code SelectorList} so that {@code @Param} annotations in {@link
 * StarlarkAttrModuleApi} can accept {@code select()} values for the {@code default} parameter
 * without introducing a dependency on the {@code packages} package.
 */
@StarlarkBuiltin(
    name = "select",
    doc = "A selector between configuration-dependent entities.",
    documented = false)
public interface SelectApi extends StarlarkValue {}
