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
package com.google.devtools.build.lib.rules.android.databinding;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import java.util.ArrayList;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

class DisabledDataBindingV1Context implements DataBindingContext {

  @Override
  public void supplyJavaCoptsUsing(RuleContext ruleContext, boolean isBinary,
      Consumer<Iterable<String>> consumer) { }

  @Override
  public void supplyAnnotationProcessor(RuleContext ruleContext,
      BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) { }

  @Override
  public ImmutableList<Artifact> processDeps(RuleContext ruleContext, boolean isBinary) {
    return ImmutableList.of();
  }

  @Override
  public ImmutableList<Artifact> getAnnotationSourceFiles(RuleContext ruleContext) {
    return ImmutableList.of();
  }

  @Override
  public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {

    ArrayList<Artifact> dataBindingMetadataOutputs = new ArrayList<>();
    // Expose the data binding provider if there are outputs.
    dataBindingMetadataOutputs.addAll(DataBinding.getTransitiveMetadata(ruleContext, "exports"));
    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      // If this rule doesn't declare direct resources, no resource processing is run so no data
      // binding outputs are produced. In that case, we need to explicitly propagate data binding
      // outputs from the deps to make sure they continue up the build graph.
      dataBindingMetadataOutputs.addAll(DataBinding.getTransitiveMetadata(ruleContext, "deps"));
    }
    if (!dataBindingMetadataOutputs.isEmpty()) {
      builder.addNativeDeclaredProvider(new UsesDataBindingProvider(dataBindingMetadataOutputs));
    }
  }

  @Override
  public AndroidResources processResources(
      AndroidDataContext dataContext, AndroidResources resources, String appId) {
    return resources;
  }

  @Override
  public void supplyLayoutInfo(Consumer<Artifact> consumer) {

  }
}
