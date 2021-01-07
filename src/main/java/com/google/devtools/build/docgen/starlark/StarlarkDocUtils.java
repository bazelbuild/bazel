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

package com.google.devtools.build.docgen.starlark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.DocgenConsts;
import net.starlark.java.annot.Param;

/** A utility class for the documentation generator. */
public final class StarlarkDocUtils {
  private StarlarkDocUtils() {}

  /**
   * Substitute special variables in the documentation with their actual values
   *
   * @return a string with substituted variables
   */
  public static String substituteVariables(String documentation) {
    return documentation
        .replace("$BE_ROOT", DocgenConsts.BeDocsRoot)
        .replace("$DOC_EXT", DocgenConsts.documentationExtension);
  }

  /**
   * Returns a list of parameter documentation elements for a given method doc and the method's
   * parameters.
   */
  static ImmutableList<StarlarkParamDoc> determineParams(
      StarlarkMethodDoc methodDoc,
      Param[] userSuppliedParams,
      Param extraPositionals,
      Param extraKeywords) {
    ImmutableList.Builder<StarlarkParamDoc> paramsBuilder = ImmutableList.builder();
    for (Param param : userSuppliedParams) {
      if (param.documented()) {
        paramsBuilder.add(new StarlarkParamDoc(methodDoc, param));
      }
    }
    if (!extraPositionals.name().isEmpty()) {
      paramsBuilder.add(new StarlarkParamDoc(methodDoc, extraPositionals));
    }
    if (!extraKeywords.name().isEmpty()) {
      paramsBuilder.add(new StarlarkParamDoc(methodDoc, extraKeywords));
    }
    return paramsBuilder.build();
  }
}
