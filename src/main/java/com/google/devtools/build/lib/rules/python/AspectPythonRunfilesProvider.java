// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety;

/**
 * Wrapper around PythonRunfilesProvider, to allow PythonProtoAspect to add Providers to
 * proto_library rules with py_api_version. If PythonProtoAspect provides PythonRunfilesProvider
 * directly on such a proto_library rule, Bazel crashes with
 *
 *     Provider class PythonRunfilesProvider provided twice
 */
@ThreadSafety.Immutable
public final class AspectPythonRunfilesProvider implements TransitiveInfoProvider {
  public final PythonRunfilesProvider provider;
  public AspectPythonRunfilesProvider(PythonRunfilesProvider provider) {
    this.provider = provider;
  }

  /**
   * A function that gets the Python runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> TO_RUNFILES =
      new Function<TransitiveInfoCollection, Runfiles>() {
        @Override
        public Runfiles apply(TransitiveInfoCollection input) {
          AspectPythonRunfilesProvider wrapper =
              input.getProvider(AspectPythonRunfilesProvider.class);
          return wrapper == null
              ? Runfiles.EMPTY
              : wrapper.provider.getPythonRunfiles();
        }
      };
}
