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
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import javax.annotation.Nullable;

/** An exception indicating that there was a problem creating an aspect. */
public final class AspectCreationException extends Exception implements SaneAnalysisException {
  private static ConfigurationId toId(BuildConfiguration config) {
    return config == null ? null : config.getEventId().getConfiguration();
  }

  private final NestedSet<Cause> causes;

  public AspectCreationException(String message, NestedSet<Cause> causes) {
    super(message);
    this.causes = causes;
  }

  public AspectCreationException(
      String message, Label currentTarget, @Nullable BuildConfiguration configuration) {
    this(
        message,
        NestedSetBuilder.<Cause>stableOrder()
            .add(new AnalysisFailedCause(currentTarget, toId(configuration), message))
            .build());
  }

  public AspectCreationException(String message, Label currentTarget) {
    this(message, currentTarget, null);
  }

  public AspectCreationException(String message, Cause cause) {
    this(message, NestedSetBuilder.<Cause>stableOrder().add(cause).build());
  }

  public NestedSet<Cause> getCauses() {
    return causes;
  }
}
