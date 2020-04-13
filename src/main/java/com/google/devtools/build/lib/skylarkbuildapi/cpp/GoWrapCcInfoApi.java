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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * Provider returned by go_wrap_cc rules that encapsulates C++ information.
 *
 * <p>The provider wrapped is CcInfo. Go SWIG have C++ dependencies that will have to be linked
 * later, however, we don't want C++ targets to be able to depend on Go SWIG, only Python targets
 * should be able to do so. Therefore, we wrap the C++ providers in a different provider which C++
 * rules do not recognize.
 */
@SkylarkModule(
    name = "GoWrapCcInfo",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER,
    doc = "")
public interface GoWrapCcInfoApi<FileT extends FileApi> extends StructApi {

  @SkylarkCallable(name = "cc_info", structField = true, documented = false, doc = "")
  CcInfoApi<FileT> getCcInfo();

  /** Provider for GoWrapCcInfo objects. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface Provider extends ProviderApi {}
}
