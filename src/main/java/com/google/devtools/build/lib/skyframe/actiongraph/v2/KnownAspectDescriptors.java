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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.KeyValuePair;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.io.IOException;
import java.util.Map;

/** Cache for AspectDescriptors in the action graph. */
public class KnownAspectDescriptors
    extends BaseCache<AspectDescriptor, AnalysisProtosV2.AspectDescriptor> {

  KnownAspectDescriptors(AqueryOutputHandler aqueryOutputHandler) {
    super(aqueryOutputHandler);
  }

  @Override
  AnalysisProtosV2.AspectDescriptor createProto(AspectDescriptor aspectDescriptor, int id)
      throws IOException {
    AnalysisProtosV2.AspectDescriptor.Builder aspectDescriptorBuilder =
        AnalysisProtosV2.AspectDescriptor.newBuilder()
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
  void toOutput(AnalysisProtosV2.AspectDescriptor aspectDescriptorProto) throws IOException {
    aqueryOutputHandler.outputAspectDescriptor(aspectDescriptorProto);
  }
}
