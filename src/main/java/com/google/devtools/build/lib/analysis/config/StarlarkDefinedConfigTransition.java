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

package com.google.devtools.build.lib.analysis.config;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigurationTransitionApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.BaseFunction;

/**
 * Implementation of {@link ConfigurationTransitionApi}.
 *
 * Represents a configuration transition across a dependency edge defined in Starlark.
 */
public class StarlarkDefinedConfigTransition implements ConfigurationTransitionApi {

  private final BaseFunction impl;

  public StarlarkDefinedConfigTransition(BaseFunction impl) {
    this.impl = impl;
  }

  /**
   * Returns the implementation function of this transition object.
   *
   * @see com.google.devtools.build.lib.skylarkbuildapi.config.ConfigGlobalLibraryApi#transition
   *     for details on the function signature
   */
  public BaseFunction getImplementation() {
    return impl;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<transition object>");
  }
}
