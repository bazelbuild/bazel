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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * A value that represents an AST file lookup result.
 */
public class ASTFileLookupValue implements SkyValue {

  private static final ASTFileLookupValue NO_FILE = new ASTFileLookupValue(null);

  @Nullable private final BuildFileAST ast;

  private ASTFileLookupValue(@Nullable BuildFileAST ast) {
    this.ast = ast;
  }

  public static ASTFileLookupValue noFile() {
    return NO_FILE;
  }

  public static ASTFileLookupValue withFile(BuildFileAST ast) {
    return new ASTFileLookupValue(ast);
  }

  /**
   * Returns the original AST file.
   */
  @Nullable public BuildFileAST getAST() {
    return ast;
  }

  static SkyKey key(PackageIdentifier astFileIdentifier) {
    return new SkyKey(SkyFunctions.AST_FILE_LOOKUP, astFileIdentifier);
  }

  static final class ASTLookupInputException extends Exception {
    ASTLookupInputException(String msg) {
      super(msg);
    }
  }
}
