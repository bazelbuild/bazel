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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.devtools.build.lib.skylarkbuildapi.DefaultInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.DefaultInfoApi.DefaultInfoApiProvider;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.RunfilesApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/**
 * Fake implementation of {@link DefaultInfoApiProvider}.
 */
public class FakeDefaultInfoProvider implements DefaultInfoApiProvider<RunfilesApi, FileApi> {

  @Override
  public DefaultInfoApi constructor(
      Object files,
      Object runfiles,
      Object dataRunfiles,
      Object defaultRunfiles,
      Object executable,
      StarlarkThread thread)
      throws EvalException {
    return null;
  }

  @Override
  public void repr(Printer printer) {}
}
