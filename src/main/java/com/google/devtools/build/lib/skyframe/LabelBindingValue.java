// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A Skyframe value representing a bound target.
 */
public class LabelBindingValue implements SkyValue {

  private final Label actual;

  LabelBindingValue(Label actual) {
    this.actual = actual;
  }

  public Label getActualLabel() {
    return actual;
  }

  /**
   * Returns a SkyKey for the given bound label. The label must start with //external/.
   */
  public static SkyKey key(Label virtual) {
    Preconditions.checkState(isBoundLabel(virtual));
    return new SkyKey(SkyFunctions.LABEL_BINDING, virtual);
  }

  /**
   * Checks if the label is bound, i.e., starts with //external/.
   */
  public static boolean isBoundLabel(Label label) {
    return label.getPackageName().equals("external");
  }

}
