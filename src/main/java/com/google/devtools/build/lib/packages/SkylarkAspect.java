// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAspectApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** Represents an aspect which can be attached to a skylark-defined rule attribute. */
public interface SkylarkAspect extends SkylarkAspectApi {

  /**
   * Attaches this aspect to an attribute.
   *
   * @param attrBuilder the builder of the attribute to add this aspect to
   * @throws EvalException if this aspect cannot be successfully applied to the given attribute
   */
  void attachToAttribute(Attribute.Builder<?> attrBuilder) throws EvalException;

  /** Returns the aspect class for this aspect. */
  AspectClass getAspectClass();

  /** Returns a set of the names of parameters required to create this aspect. */
  ImmutableSet<String> getParamAttributes();

  /** Returns the name of this aspect. */
  String getName();
}
