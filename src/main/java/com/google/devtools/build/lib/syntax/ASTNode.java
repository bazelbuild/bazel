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

package com.google.devtools.build.lib.syntax;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.events.Location;

import java.io.Serializable;

/**
 * Root class for nodes in the Abstract Syntax Tree of the Build language.
 */
public abstract class ASTNode implements Serializable {

  private Location location;

  protected ASTNode() {}

  /**
   * Returns whether this node represents a new scope, e.g. a function call.
   */
  protected boolean isNewScope()  {
    return false;
  }

  /** Returns an exception which should be thrown instead of the original one. */
  protected final EvalException maybeTransformException(EvalException original) {
    // If there is already a non-empty stack trace, we only add this node iff it describes a
    // new scope (e.g. FuncallExpression).
    if (original instanceof EvalExceptionWithStackTrace) {
      EvalExceptionWithStackTrace real = (EvalExceptionWithStackTrace) original;
      if (isNewScope()) {
        real.registerNode(this);
      }
      return real;
    }

    if (original.canBeAddedToStackTrace()) {
      return new EvalExceptionWithStackTrace(original, this);
    } else {
      return original;
    }
  }

  @VisibleForTesting  // productionVisibility = Visibility.PACKAGE_PRIVATE
  public void setLocation(Location location) {
    this.location = location;
  }

  public Location getLocation() {
    return location;
  }

  /** @return the same node with its location set, in a slightly more fluent style */
  public static <NODE extends ASTNode> NODE setLocation(Location location, NODE node) {
    node.setLocation(location);
    return node;
  }

  /**
   * Print the syntax node in a form useful for debugging.  The output is not
   * precisely specified, and should not be used by pretty-printing routines.
   */
  @Override
  public abstract String toString();

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException(); // avoid nondeterminism
  }

  @Override
  public boolean equals(Object that) {
    throw new UnsupportedOperationException();
  }

  /**
   * Implements the double dispatch by calling into the node specific
   * <code>visit</code> method of the {@link SyntaxTreeVisitor}
   *
   * @param visitor the {@link SyntaxTreeVisitor} instance to dispatch to.
   */
  public abstract void accept(SyntaxTreeVisitor visitor);

}
