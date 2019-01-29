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

/**
 * Helper object containing CC linking info objects.
 */
@SkylarkModule(
  name = "linking_info",
  documented = false,
  category = SkylarkModuleCategory.BUILTIN,
  doc = "Helper class containing CC linking providers."
)
public interface LinkingInfoApi {

  @SkylarkCallable(name = "linking_context", structField = true, documented = false)
  CcLinkingContextApi getCcLinkingContext();

  @SkylarkCallable(name = "cc_linking_outputs", structField = true, documented = false)
  CcLinkingOutputsApi getCcLinkingOutputs();
}
