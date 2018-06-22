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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;

/** A provider that supplies resource information from its transitive closure. */
@SkylarkModule(
    name = "AndroidResourcesInfo",
    doc = "Android resources provided by a rule",
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidResourcesInfoApi extends StructApi {

  /**
   * Name of this info object.
   */
  public static String NAME = "AndroidResourcesInfo";

  /** Returns the compiletime r.txt file for the target. */
  @SkylarkCallable(
      name = "compiletime_r_txt",
      doc =
          "A txt file containing compiled resource file information for this target. This is a"
              + " stubbed out compiletime file and should not be built into APKs, inherited from"
              + " dependencies, or used at runtime.",
      structField = true)
  FileApi getRTxt();

  /** Provider for {@link AndroidResourcesInfoApi}. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface AndroidResourcesInfoApiProvider extends ProviderApi {

    @SkylarkCallable(
        name = "AndroidResourcesInfo",
        // This is left undocumented as it throws a "not-implemented in Skylark" error when invoked.
        documented = false,
        extraKeywords = @Param(name = "kwargs"),
        useLocation = true,
        selfCall = true)
    public AndroidResourcesInfoApi createInfo(
        SkylarkDict<?, ?> kwargs, Location loc) throws EvalException;
  }
}
