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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value that represents an AST file lookup result.
 */
public class ASTLookupValue implements SkyValue {

  private final ImmutableList<Statement> statements;

  public ASTLookupValue(ImmutableList<Statement> statements) {
    this.statements = statements;
  }

  /**
   * Returns the list of statements present parsed from the original AST file.
   */
  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  static SkyKey key(PathFragment directory) {
    return new SkyKey(SkyFunctions.AST_LOOKUP, directory);
  }
}
