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

package com.google.devtools.build.lib.bazel.repository.starlark;

import net.starlark.java.eval.EvalException;

/**
 * An exception thrown from within Starlark code when a Skyframe dependency is missing, in order to
 * notify the SkyFunction from an evaluation.
 */
public class NeedsSkyframeRestartException extends EvalException {

  public NeedsSkyframeRestartException() {
    super("Internal exception");
  }
}
