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
package com.google.devtools.build.lib.rules.cpp;


import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Aspect for constructing a tree of labels that is used to prune static libraries that are already
 * linked dynamically into a cc_binary. TODO(b/145508948): Try to remove this class in the future.
 */
public final class GraphNodeAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  // When the dynamic_deps attribute is not set, we return null. We would only want the graph to be
  // analyzed with the aspect in the cases that we have set dynamic_deps. Otherwise it would be a
  // waste of memory in the cases where we don't need the aspect. If we return null, the aspect is
  // not used analyze anything.
  // See
  // https://github.com/bazelbuild/bazel/blob/df52777aac8cbfc7719af9f0dbb23335e59c42df/src/main/java/com/google/devtools/build/lib/packages/Attribute.java#L114
  public static final Function<Rule, AspectParameters> ASPECT_PARAMETERS =
      new Function<Rule, AspectParameters>() {
        @Nullable
        @Override
        public AspectParameters apply(Rule rule) {
          return rule.isAttributeValueExplicitlySpecified("dynamic_deps")
              ? AspectParameters.EMPTY
              : null;
        }
      };

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return new AspectDefinition.Builder(this).propagateAlongAllAttributes().build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters params,
      String toolsRepository)
      throws ActionConflictException {
    List<Label> linkedStaticallyBy = null;
    ImmutableList.Builder<GraphNodeInfo> children = ImmutableList.builder();
    if (ruleContext.attributes().has("deps")) {
      children.addAll(
          AnalysisUtils.getProviders(
              ruleContext.getPrerequisites("deps", Mode.TARGET), GraphNodeInfo.class));
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("linked_statically_by")) {
      linkedStaticallyBy =
          ruleContext.attributes().get("linked_statically_by", BuildType.NODEP_LABEL_LIST);
    }
    return new ConfiguredAspect.Builder(this, params, ruleContext)
        .addProvider(
            GraphNodeInfo.class,
            new GraphNodeInfo(ruleContext.getLabel(), linkedStaticallyBy, children.build()))
        .build();
  }
}
