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
package com.google.devtools.build.lib.vfs;

import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * The SkyKey for FileStateValue.
 *
 * <p>This interface exists to allow {@link RootedPath} to implement {@link SkyKey} in such a way
 * that makes it obvious that {@link RootedPath} is the {@link SkyKey} for FileStateValue.
 */
public interface FileStateKey extends SkyKey {
  SkyFunctionName FILE_STATE = SkyFunctionName.createNonHermetic("FILE_STATE");

  @Override
  default SkyFunctionName functionName() {
    return FILE_STATE;
  }

  @Override
  RootedPath argument();
}
