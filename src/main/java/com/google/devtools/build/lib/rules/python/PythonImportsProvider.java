// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A {@link TransitiveInfoProvider} that supplies import directories for Python dependencies. */
// TODO(brandjon): Merge this provider into PyProvider.
@Immutable
@AutoCodec
public final class PythonImportsProvider implements TransitiveInfoProvider {

  private final NestedSet<String> transitivePythonImports;

  public PythonImportsProvider(NestedSet<String> transitivePythonImports) {
    this.transitivePythonImports = transitivePythonImports;
  }

  public NestedSet<String> getTransitivePythonImports() {
    return transitivePythonImports;
  }
}
