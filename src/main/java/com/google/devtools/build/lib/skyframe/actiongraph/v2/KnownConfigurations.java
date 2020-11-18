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

import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import java.io.IOException;

/** Cache for BuildConfigurations in the action graph. */
public class KnownConfigurations extends BaseCache<BuildEvent, Configuration> {

  KnownConfigurations(AqueryOutputHandler aqueryOutputHandler) {
    super(aqueryOutputHandler);
  }

  @Override
  Configuration createProto(BuildEvent config, int id) throws InterruptedException {
    BuildEventStreamProtos.Configuration configProto =
        config.asStreamProto(/*context=*/ null).getConfiguration();
    return Configuration.newBuilder()
        .setChecksum(config.getEventId().getConfiguration().getId())
        .setMnemonic(configProto.getMnemonic())
        .setPlatformName(configProto.getPlatformName())
        .setId(id)
        .build();
  }

  @Override
  void toOutput(Configuration configurationProto) throws IOException {
    aqueryOutputHandler.outputConfiguration(configurationProto);
  }
}
