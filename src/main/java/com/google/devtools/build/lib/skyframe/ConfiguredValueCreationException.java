// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/**
 * An exception indicating that there was a problem during the construction of a
 * ConfiguredTargetValue.
 */
@AutoCodec
public final class ConfiguredValueCreationException extends Exception
    implements SaneAnalysisException {

  @Nullable private final BuildEventId configuration;
  private final NestedSet<Cause> rootCauses;

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  ConfiguredValueCreationException(
      String message, @Nullable BuildEventId configuration, NestedSet<Cause> rootCauses) {
    super(message);
    this.configuration = configuration;
    this.rootCauses = rootCauses;
  }

  public ConfiguredValueCreationException(
      String message, Label currentTarget, @Nullable BuildConfiguration configuration) {
    this(
        message,
        configuration == null ? null : configuration.getEventId(),
        NestedSetBuilder.<Cause>stableOrder()
            .add(
                new AnalysisFailedCause(
                    currentTarget,
                    configuration == null ? null : configuration.getEventId().getConfiguration(),
                    message))
            .build());
  }

  public ConfiguredValueCreationException(
      String message, @Nullable BuildConfiguration configuration, NestedSet<Cause> rootCauses) {
    this(message, configuration == null ? null : configuration.getEventId(), rootCauses);
  }

  public NestedSet<Cause> getRootCauses() {
    return rootCauses;
  }

  @Nullable
  public BuildEventId getConfiguration() {
    return configuration;
  }
}
