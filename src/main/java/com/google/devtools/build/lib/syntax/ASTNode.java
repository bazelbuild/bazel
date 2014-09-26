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

  @VisibleForTesting  // productionVisibility = Visibility.PACKAGE_PRIVATE
  public void setLocation(Location location) {
    this.location = location;
  }

  public Location getLocation() {
    return location;
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
