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

package com.google.devtools.build.lib.skylarkbuildapi.go;

import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Information about the transitive closure of a target that is relevant to Go compilation. */
@SkylarkModule(
    name = "GoContextInfo",
    doc = "",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER)
public interface GoContextInfoApi extends StructApi {

  /** Provider for GoContextInfo objects. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface Provider extends ProviderApi {}
}
