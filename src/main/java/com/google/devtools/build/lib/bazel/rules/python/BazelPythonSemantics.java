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

package com.google.devtools.build.lib.bazel.rules.python;

import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.rules.python.PythonSemantics;
import com.google.devtools.build.lib.rules.python.PythonUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.util.UUID;
import java.util.function.Predicate;

/** Functionality specific to the Python rules in Bazel. */
public class BazelPythonSemantics implements PythonSemantics {

  private static final UUID GUID = UUID.fromString("0211a192-1b1e-40e6-80e9-7352360b12b1");
  public static final Runfiles.EmptyFilesSupplier GET_INIT_PY_FILES =
      new PythonUtils.GetInitPyFiles(
          (Predicate<PathFragment> & Serializable) source -> false, GUID);

  @Override
  public Runfiles.EmptyFilesSupplier getEmptyRunfilesSupplier() {
    return GET_INIT_PY_FILES;
  }
}
