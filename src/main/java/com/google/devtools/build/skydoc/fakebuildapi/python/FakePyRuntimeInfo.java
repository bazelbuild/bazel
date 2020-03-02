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

package com.google.devtools.build.skydoc.fakebuildapi.python;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.python.PyRuntimeInfoApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/** Fake implementation of {@link PyRuntimeInfoApi}. */
public class FakePyRuntimeInfo implements PyRuntimeInfoApi<FileApi> {

  @Override
  public String getInterpreterPathString() {
    return null;
  }

  @Override
  public FileApi getInterpreter() {
    return null;
  }

  @Override
  public Depset getFilesForStarlark() {
    return null;
  }

  @Override
  public String getPythonVersionForStarlark() {
    return "";
  }

  @Override
  public void repr(Printer printer) {}

  /** Fake implementation of {@link PyRuntimeInfoProviderApi}. */
  public static class FakePyRuntimeInfoProvider implements PyRuntimeInfoProviderApi {

    @Override
    public PyRuntimeInfoApi<?> constructor(
        Object interpreterPathUncast,
        Object interpreterUncast,
        Object filesUncast,
        String pythonVersion,
        StarlarkThread thread)
        throws EvalException {
      return new FakePyRuntimeInfo();
    }

    @Override
    public void repr(Printer printer) {}
  }
}
