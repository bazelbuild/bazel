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
 * Provider returned by py_wrap_cc rules that encapsulates C++ information.
 *
 * <p>The provider wrapped is CcInfo. Python SWIG have C++ dependencies that will have to be linked
 * later, however, we don't want C++ targets to be able to depend on Python SWIG, only Python
 * targets should be able to do so. Therefore, we wrap the C++ providers in a different provider
 * which C++ rules do not recognize.
 */
@SkylarkModule(
    name = "PyWrapCcInfo",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER,
    doc = "")
public interface PyWrapCcInfoApi<FileT extends FileApi> extends StructApi {

  @SkylarkCallable(name = "cc_info", documented = false, structField = true, doc = "")
  CcInfoApi<FileT> getCcInfo();

  /** Provider for PyWrapCcInfo objects. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface Provider extends ProviderApi {}
}
