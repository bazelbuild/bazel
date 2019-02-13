// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetEdgeObserver;

/**
 * Record errors, such as missing package/target or rules containing errors,
 * encountered during visitation. Emit an error message upon encountering
 * missing edges
 *
 * The accessor {@link #hasErrors}) may not be called until the concurrent phase
 * is over, i.e. all external calls to visit() methods have completed.
 *
 * If you need to report errors to the console during visitation, use the
 * subclass {@link com.google.devtools.build.lib.query2.ErrorPrintingTargetEdgeErrorObserver}.
 */
class TargetEdgeErrorObserver implements TargetEdgeObserver {

  /**
   * True iff errors were encountered.  Note, may be set to "true" during the
   * concurrent phase.  Volatile, because it is assigned by worker threads and
   * read by the main thread without monitor synchronization.
   */
  private volatile boolean hasErrors = false;

  /**
   * Reports an unresolved label error and records the fact that an error was encountered.
   *
   * @param target the target that referenced the unresolved label
   * @param label the label that could not be resolved
   * @param e the exception that was thrown when the label could not be resolved
   */
  @ThreadSafety.ThreadSafe
  @Override
  public void missingEdge(Target target, Label label, NoSuchThingException e) {
    hasErrors = true;
  }

  /**
   * Returns true iff any errors (such as missing targets or packages, or rules
   * with errors) have been encountered during any work.
   *
   * <p>Not thread-safe; do not call during visitation.
   *
   *  @return true iff no errors (such as missing targets or packages, or rules
   *              with errors) have been encountered during any work.
   */
  public boolean hasErrors() {
    return hasErrors;
  }

  @Override
  public void edge(Target from, Attribute attribute, Target to) {
    // No-op.
  }

  @Override
  public void node(Target node) {
    if (node.getPackage().containsErrors()
        || ((node instanceof Rule) && ((Rule) node).containsErrors())) {
      this.hasErrors = true;  // Note, this is thread-safe.
    }
  }
}
