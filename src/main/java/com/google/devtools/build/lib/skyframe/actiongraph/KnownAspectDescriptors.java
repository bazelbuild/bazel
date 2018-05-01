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
package com.google.devtools.build.lib.skyframe.actiongraph;

import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtos.KeyValuePair;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.util.Map;

/**
 * Cache for AspectDescriptors in the action graph.
 */
public class KnownAspectDescriptors
    extends BaseCache<AspectDescriptor, AnalysisProtos.AspectDescriptor> {

  KnownAspectDescriptors(ActionGraphContainer.Builder actionGraphBuilder) {
    super(actionGraphBuilder);
  }

  @Override
  AnalysisProtos.AspectDescriptor createProto(AspectDescriptor aspectDescriptor, String id) {
    AnalysisProtos.AspectDescriptor.Builder aspectDescriptorBuilder =
        AnalysisProtos.AspectDescriptor.newBuilder()
            .setId(id)
            .setName(aspectDescriptor.getAspectClass().getName());
    for (Map.Entry<String, String> parameter :
        aspectDescriptor.getParameters().getAttributes().entries()) {
      KeyValuePair.Builder keyValuePairBuilder = KeyValuePair.newBuilder();
      keyValuePairBuilder.setKey(parameter.getKey()).setValue(parameter.getValue());
      aspectDescriptorBuilder.addParameters(keyValuePairBuilder.build());
    }
    return aspectDescriptorBuilder.build();
  }

  @Override
  void addToActionGraphBuilder(AnalysisProtos.AspectDescriptor aspectDescriptorProto) {
    actionGraphBuilder.addAspectDescriptors(aspectDescriptorProto);
  }
}
