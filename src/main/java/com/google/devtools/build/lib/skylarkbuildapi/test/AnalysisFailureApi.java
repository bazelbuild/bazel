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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * Encapsulates information about an analysis-phase error which would have occurred during a build.
 */
@SkylarkModule(
    name = "AnalysisFailure",
    doc =
        "Encapsulates information about an analysis-phase error which would have occurred during "
            + "a build. In most builds, an analysis-phase error would result in a build failure "
            + "and the error description would be output to the console. However, if "
            + "<code>--allow_analysis_failure</code> is set, targets which would otherwise fail in "
            + "analysis will instead propagate an <code>AnalysisFailureInfo</code> object "
            + "containing one or more instances of this object.",
    documented = false)
public interface AnalysisFailureApi extends StarlarkValue {

  @SkylarkCallable(
      name = "label",
      doc =
          "The label of the target that exhibited an analysis-phase error. This is the label "
              + "of the target responsible for construction of this object.",
      documented = false,
      structField = true)
  Label getLabel();

  @SkylarkCallable(
      name = "message",
      doc = "A string representation of the analysis-phase error which occurred.",
      documented = false,
      structField = true)
  String getMessage();
}
