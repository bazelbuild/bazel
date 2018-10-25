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

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Wrapper for every C++ linking provider. */
@SkylarkModule(
    name = "CcLinkingContext",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER,
    doc = "Wrapper for every C++ linking provider")
public interface CcLinkingInfoApi {

  @SkylarkCallable(
      name = "static_mode_params_for_dynamic_library",
      documented = false,
      allowReturnNones = true,
      structField = true)
  CcLinkParamsApi getStaticModeParamsForDynamicLibrary();

  @SkylarkCallable(
      name = "static_mode_params_for_executable",
      documented = false,
      allowReturnNones = true,
      structField = true)
  CcLinkParamsApi getStaticModeParamsForExecutable();

  @SkylarkCallable(
      name = "dynamic_mode_params_for_dynamic_library",
      documented = false,
      allowReturnNones = true,
      structField = true)
  CcLinkParamsApi getDynamicModeParamsForDynamicLibrary();

  @SkylarkCallable(
      name = "dynamic_mode_params_for_executable",
      documented = false,
      allowReturnNones = true,
      structField = true)
  CcLinkParamsApi getDynamicModeParamsForExecutable();
}
