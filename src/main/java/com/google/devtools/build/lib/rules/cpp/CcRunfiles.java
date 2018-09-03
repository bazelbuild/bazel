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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcRunfilesApi;

/**
 * Runfiles for C++ targets.
 *
 * <p>Contains two {@link Runfiles} objects: one for the statically linked binary and one for the
 * dynamically linked binary. Both contain dynamic libraries needed at runtime and data
 * dependencies.
 */
// TODO(plf): Remove class once Skylark rules have been migrated off it.
@Immutable
@AutoCodec
public final class CcRunfiles implements CcRunfilesApi {

  @Override
  public Runfiles getRunfilesForLinkingStatically() {
    return Runfiles.EMPTY;
  }

  @Override
  public Runfiles getRunfilesForLinkingDynamically() {
    return Runfiles.EMPTY;
  }

}
