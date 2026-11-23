// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlarkprofiler;

import com.google.devtools.build.lib.runtime.BlazeService;
import javax.annotation.Nullable;
import net.starlark.java.eval.CpuProfilerNativeSupport;

/** A {@link BlazeService} that provides access to {@link CpuProfilerNativeSupport}. */
@SuppressWarnings("GoodTime")
public interface CpuProfilerService extends BlazeService {
  /**
   * Returns the {@link CpuProfilerNativeSupport} implementation, or null if one isn't available for
   * the current platform.
   */
  @Nullable
  CpuProfilerNativeSupport getCpuProfilerNativeSupport();
}
