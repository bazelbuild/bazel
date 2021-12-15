// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A {@link TransitiveInfoProvider} that supplies runfiles for Python dependencies.
 *
 * <p>Should only be used in proto_library, and even then only until a better mechanism is found.
 */
@Immutable
public final class PythonRunfilesProvider implements TransitiveInfoProvider {
  private final Runfiles pythonRunfiles;

  public PythonRunfilesProvider(Runfiles pythonRunfiles) {
    this.pythonRunfiles = pythonRunfiles;
  }

  public Runfiles getPythonRunfiles() {
    return pythonRunfiles;
  }

  /**
   * Returns a function that gets the Python runfiles from a {@link TransitiveInfoCollection} or the
   * empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> TO_RUNFILES =
      input -> {
        PythonRunfilesProvider provider = input.getProvider(PythonRunfilesProvider.class);
        return provider == null ? Runfiles.EMPTY : provider.getPythonRunfiles();
      };
}
