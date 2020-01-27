// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions.util;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Objects;

/** ArtifactOwner wrapper for Labels, for use in tests. */
@VisibleForTesting
@AutoCodec
public class LabelArtifactOwner implements ArtifactOwner {
  private final Label label;

  @VisibleForTesting
  public LabelArtifactOwner(Label label) {
    this.label = label;
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public int hashCode() {
    return label == null ? super.hashCode() : label.hashCode();
  }

  @Override
  public boolean equals(Object that) {
    if (!(that instanceof LabelArtifactOwner)) {
      return false;
    }
    return Objects.equals(this.label, ((LabelArtifactOwner) that).label);
  }
}
