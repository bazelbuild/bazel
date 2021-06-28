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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAspectApi;
import net.starlark.java.eval.EvalException;

/** Represents an aspect which can be attached to a Starlark-defined rule attribute. */
public interface StarlarkAspect extends StarlarkAspectApi {

  /**
   * Attaches this aspect and its required aspects to the given builder.
   *
   * <p>Also pass the list of required_providers of the base aspect to its required aspects to
   * ensure that they will be propgataed to the same targets. But whether the required aspects will
   * run on these targets or not depends on their required providers.
   *
   * <p>The list of attr_aspects of the base aspects is also passed to its required aspects to
   * ensure that they will be propagated with it along the same attributes.
   *
   * @param baseAspectName is the name of the base aspect requiring this aspect, can be {@code null}
   *     if the aspect is directly listed in the attribute aspects list
   * @param builder is the builder of the attribute to add this aspect to
   * @param inheritedRequiredProviders is the list of required providers inherited from the aspect
   *     parent aspects
   * @param inheritedAttributeAspects is the list of attribute aspects inherited from the aspect
   *     parent aspects
   * @throws EvalException if this aspect cannot be successfully applied to the given attribute
   */
  void attachToAttribute(
      String baseAspectName,
      Attribute.Builder<?> builder,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders,
      ImmutableList<String> inheritedAttributeAspects)
      throws EvalException;

  /** Returns the aspect class for this aspect. */
  AspectClass getAspectClass();

  /** Returns a set of the names of parameters required to create this aspect. */
  ImmutableSet<String> getParamAttributes();

  /** Returns the name of this aspect. */
  String getName();

  /** Returns a function to extract the aspect parameters values from its base rule. */
  Function<Rule, AspectParameters> getDefaultParametersExtractor();
}
