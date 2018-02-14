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

package com.google.devtools.build.lib.analysis;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * Container for some attributes of a {@link Target} that is significantly less heavyweight than an
 * actual {@link Target} for purposes of serialization. Should still not be used indiscriminately,
 * since {@link Location} can be quite heavy on its own and each of these wrapper objects costs 24
 * bytes over an existing {@link Target}.
 */
@AutoCodec
@AutoValue
public abstract class LabelAndLocation {
  @AutoCodec.Instantiator
  static LabelAndLocation create(Label label, Location location) {
    return new AutoValue_LabelAndLocation(label, location);
  }

  public static LabelAndLocation of(Target target) {
    return create(target.getLabel(), target.getLocation());
  }

  public abstract Label getLabel();

  public abstract Location getLocation();
}
