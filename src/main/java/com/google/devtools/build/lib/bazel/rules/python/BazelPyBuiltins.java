// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.rules.python.PyBuiltins;
import com.google.devtools.build.lib.rules.python.PythonUtils;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import java.util.UUID;

/** PyBuiltins with Bazel-specific functionality. */
public final class BazelPyBuiltins extends PyBuiltins {

  private static final UUID GUID = UUID.fromString("0211a192-1b1e-40e6-80e9-7352360b12b1");

  @SerializationConstant
  public static final Runfiles.EmptyFilesSupplier GET_INIT_PY_FILES =
      new PythonUtils.GetInitPyFiles(source -> false, GUID);

  public BazelPyBuiltins() {
    super(GET_INIT_PY_FILES);
  }
}
