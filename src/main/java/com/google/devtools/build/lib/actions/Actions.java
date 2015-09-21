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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.Iterables;
import com.google.common.escape.Escaper;
import com.google.common.escape.Escapers;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * Helper class for actions.
 */
@ThreadSafe
public final class Actions {
  private static final Escaper PATH_ESCAPER = Escapers.builder()
      .addEscape('_', "_U")
      .addEscape('/', "_S")
      .addEscape('\\', "_B")
      .addEscape(':', "_C")
      .build();

  /**
   * Checks if the two actions are equivalent. This method exists to support sharing actions between
   * configured targets for cases where there is no canonical target that could own the action. In
   * the action graph construction this case shows up as two actions generating the same output
   * file.
   *
   * <p>This method implements an equivalence relationship across actions, based on the action
   * class, the key, and the list of inputs and outputs.
   */
  public static boolean canBeShared(Action a, Action b) {
    if (!a.getMnemonic().equals(b.getMnemonic())) {
      return false;
    }
    if (!a.getKey().equals(b.getKey())) {
      return false;
    }
    // Don't bother to check input and output counts first; the expected result for these tests is
    // to always be true (i.e., that this method returns true).
    if (!Iterables.elementsEqual(a.getMandatoryInputs(), b.getMandatoryInputs())) {
      return false;
    }
    if (!Iterables.elementsEqual(a.getOutputs(), b.getOutputs())) {
      return false;
    }
    return true;
  }

  /**
   * Returns the escaped name for a given relative path as a string. This takes
   * a short relative path and turns it into a string suitable for use as a
   * filename. Invalid filename characters are escaped with an '_' + a single
   * character token.
   */
  public static String escapedPath(String path) {
    return PATH_ESCAPER.escape(path);
  }

  /**
   * Returns a string that is usable as a unique path component for a label. It is guaranteed
   * that no other label maps to this string.
   */
  public static String escapeLabel(Label label) {
    return PATH_ESCAPER.escape(label.getPackageName() + ":" + label.getName());
  }
}
