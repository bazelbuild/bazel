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

package com.google.devtools.build.lib.query2;

import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.syntax.Label;

/**
 * Record errors, such as missing package/target or rules containing errors,
 * encountered during visitation. Emit an error message upon encountering
 * missing edges
 *
 * The accessor {@link #hasErrors}) may not be called until the concurrent phase
 * is over, i.e. all external calls to visit() methods have completed.
 */
@ThreadSafety.ConditionallyThreadSafe // condition: only call hasErrors
                                      // once the visitation is complete.
class ErrorPrintingTargetEdgeErrorObserver extends TargetEdgeErrorObserver {

  private final ErrorEventListener listener;

  /**
   * @param listener listener to route exceptions to as errors.
   */
  public ErrorPrintingTargetEdgeErrorObserver(ErrorEventListener listener) {
    this.listener = listener;
  }

  @ThreadSafety.ThreadSafe
  @Override
  public void missingEdge(Target target, Label label, NoSuchThingException e) {
    listener.error(TargetUtils.getLocationMaybe(target),
        TargetUtils.formatMissingEdge(target, label, e));
    super.missingEdge(target, label, e);
  }
}
