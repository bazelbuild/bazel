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

package com.google.devtools.build.skydoc.fakebuildapi.test;

import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.test.CoverageCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.test.InstrumentedFilesInfoApi;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;

/** Fake implementation of {@link CoverageCommonApi}. */
public class FakeCoverageCommon
    implements CoverageCommonApi<
        ConstraintValueInfoApi, StarlarkRuleContextApi<ConstraintValueInfoApi>> {

  @Override
  public InstrumentedFilesInfoApi instrumentedFilesInfo(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      Sequence<?> sourceAttributes,
      Sequence<?> dependencyAttributes,
      Object extensions) {
    return null;
  }

  @Override
  public void repr(Printer printer) {}
}
