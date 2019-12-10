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

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * Encapsulates information about an analysis-phase error which would have occurred during a build.
 */
@SkylarkModule(
    name = "AnalysisFailureInfo",
    doc =
        "<b>Experimental. This API is experimental and subject to change at any time</b><p>"
            + " Encapsulates information about an analysis-phase error which would have occurred"
            + " during a build. In most builds, an analysis-phase error would result in a build"
            + " failure and the error description would be output to the console. However, if"
            + " <code>--allow_analysis_failure</code> is set, targets which would otherwise fail"
            + " in analysis will instead propagate an instance of this object (and no other"
            + " provider instances). <p>Under <code>--allow_analysis_failure</code>,"
            + " <code>AnalysisFailureInfo</code> objects are automatically re-propagated up a"
            + " dependency tree using the following logic:<ul><li>If a target fails but none of"
            + " its direct dependencies propagated <code>AnalysisFailureInfo</code>, then"
            + " propagate an instance of this provider containing an <code>AnalysisFailure</code>"
            + " object describing the failure.</li> <li>If one or more of a target's dependencies"
            + " propagated <code>AnalysisFailureInfo</code>, then propagate a provider with"
            + " <code>causes</code> equal to the union of the <code>causes</code> of the "
            + "dependencies.</li></ul>",
    documented = false)
public interface AnalysisFailureInfoApi<AnalysisFailureApiT extends AnalysisFailureApi>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "causes",
      doc =
          "A depset of <code>AnalysisFailure</code> objects describing the failures that "
              + "occurred in this target or its dependencies.",
      documented = false,
      structField = true)
  Depset /*<AnalysisFailureApiT>*/ getCauses();

  /** Provider class for {@link AnalysisFailureInfoApi} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface AnalysisFailureInfoProviderApi extends ProviderApi {}
}
