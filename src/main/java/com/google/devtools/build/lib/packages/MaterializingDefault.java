// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.EventHandler;
import net.starlark.java.eval.EvalException;

/**
 * The default value of attributes with materializers.
 *
 * <p>It's just a reference to the function that does the materializing.
 */
public class MaterializingDefault<ValueT, AnalysisContextT> {
  private final Type<ValueT> type;
  private final Resolver<ValueT, AnalysisContextT> resolver;
  private final Class<? extends AnalysisContextT> analysisContextClass;

  public MaterializingDefault(
      Type<ValueT> type,
      Class<? extends AnalysisContextT> analysisContextClass,
      Resolver<ValueT, AnalysisContextT> resolver) {
    Preconditions.checkArgument(type == BuildType.LABEL || type == BuildType.LABEL_LIST);
    this.type = type;
    this.resolver = resolver;
    this.analysisContextClass = analysisContextClass;
  }

  public ValueT getDefault() {
    // Materializers can only return dormant dependencies, which are already present in the
    // transitive closure. So we can safely return "nothing" for "bazel query": the invariant that
    // everything needed to build a target is returned by "bazel query deps()" holds because
    // whatever a materializer returns is reachable through dormant dependency edges, which are
    // traversed by "bazel query".
    return type.getDefaultValue();
  }

  /**
   * The implementation of the actual resolution of the late-bound default.
   *
   * <p>This is a separate interface because MaterializingDefault must be known to the loading phase
   * but its implementation necessarily deals with analysis-phase data structures.
   */
  public interface Resolver<ValueT, PrerequisiteT> {

    /**
     * Resolves an attribute with a materializer.
     *
     * <p>param rule the rule whose attribute is to be resolved.
     *
     * @param attributes the attributes of the rule, after resolving {@code select()} and the like
     * @param prerequisiteMap a map from attribute name to the prerequisites on that attribute. Only
     *     those attributes are present that represent dependencies and which are available for
     *     dependency resolution. The value of the map is in fact {@code List<? extends
     *     TransitiveInfoCollection}, but we can't say that because this class needs to be available
     *     in the loading phase.
     * @param eventHandler messages from Starlark should be reported here
     * @return the value of the resolved attribute.
     */
    ValueT resolve(
        Rule rule,
        AttributeMap attributes,
        PrerequisiteT prerequisiteMap,
        EventHandler eventHandler)
        throws InterruptedException, EvalException;
  }

  public ValueT resolve(
      Rule rule,
      AttributeMap attributes,
      Object analysisContext,
      EventHandler eventHandler)
      throws InterruptedException, EvalException {
    return resolver.resolve(
        rule, attributes, analysisContextClass.cast(analysisContext), eventHandler);
  }
}
