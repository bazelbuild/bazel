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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.CodecRegisterer;
import com.google.devtools.build.lib.skyframe.serialization.MessageLiteCodec;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;

class CrosstoolConfigProtoCodecRegisterer implements CodecRegisterer<MessageLiteCodec> {
  @Override
  public Iterable<MessageLiteCodec> getCodecsToRegister() {
    return ImmutableList.of(
        new MessageLiteCodec(CrosstoolConfig.CrosstoolRelease::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.FlagGroup::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.VariableWithValue::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.EnvEntry::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.FeatureSet::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.WithFeatureSet::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.FlagSet::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.EnvSet::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.Feature::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.Tool::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.ArtifactNamePattern::newBuilder),
        new MessageLiteCodec(CrosstoolConfig.CToolchain.ActionConfig::newBuilder));
  }
}
