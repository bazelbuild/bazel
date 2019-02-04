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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;

/** Contains information about instrumented files sources and instrumentation metadata. */
@SkylarkModule(
    name = "InstrumentedFilesInfo",
    category = SkylarkModuleCategory.NONE,
    doc =
        "Contains information about instrumented file sources and instrumentation metadata "
            + "for purposes of code coverage. Rule targets which return an instance of this "
            + "provider signal to the build system that certain sources should be targeted for "
            + "code coverage analysis.")
public interface InstrumentedFilesInfoApi extends SkylarkValue {}
