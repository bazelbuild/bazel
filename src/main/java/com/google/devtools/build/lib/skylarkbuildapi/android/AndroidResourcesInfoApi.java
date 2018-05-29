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
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** A provider that supplies resource information from its transitive closure. */
@SkylarkModule(
    name = "AndroidResourcesInfo",
    doc = "Android resources provided by a rule",
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidResourcesInfoApi extends StructApi {

  /** Returns the r.txt file for the target. */
  @SkylarkCallable(
      name = "r_txt",
      doc = "Returns the R.txt file for the target.",
      structField = true)
  FileApi getRTxt();
}
