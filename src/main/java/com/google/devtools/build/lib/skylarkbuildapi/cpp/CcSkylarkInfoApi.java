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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Skylark rules need to provide this to be a dependency of native cc_* rules. */
@SkylarkModule(
    name = "cc_skylark_info",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "Temporary provider to mark C++ Skylark rules")
// TODO(b/77669139): Delete this class. Use {@link CcInfo} instead.
public interface CcSkylarkInfoApi extends StructApi {}
