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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value that represents an AST file lookup result. There are two subclasses: one for the
 * case where the file is found, and another for the case where the file is missing (but there
 * are no other errors).
 */
abstract class ASTFileLookupValue implements SkyValue {
  public abstract boolean lookupSuccessful();
  public abstract BuildFileAST getAST();
  public abstract String getErrorMsg();
  
  /** If the file is found, this class encapsulates the parsed AST. */
  private static class ASTLookupWithFile extends ASTFileLookupValue {
    private final BuildFileAST ast;

    private ASTLookupWithFile(BuildFileAST ast) {
      Preconditions.checkNotNull(ast);
      this.ast = ast;
    }

    @Override
    public boolean lookupSuccessful() {
      return true;
    }

    @Override
    public BuildFileAST getAST() {
      return this.ast;
    }

    @Override
    public String getErrorMsg() {
      throw new IllegalStateException(
          "attempted to retrieve unsuccessful lookup reason for successful lookup");
    }
  }
 
  /** If the file isn't found, this class encapsulates a message with the reason. */
  private static class ASTLookupNoFile extends ASTFileLookupValue {
    private final String errorMsg;

    private ASTLookupNoFile(String errorMsg) {
      this.errorMsg = Preconditions.checkNotNull(errorMsg);
    }

    @Override
    public boolean lookupSuccessful() {
      return false;
    }

    @Override
    public BuildFileAST getAST() {
      throw new IllegalStateException("attempted to retrieve AST from an unsuccessful lookup");
    }

    @Override
    public String getErrorMsg() {
      return this.errorMsg;
    }
  }

  static ASTFileLookupValue forBadPackage(Label fileLabel, String reason) {
    return new ASTLookupNoFile(
        String.format("Unable to load package for '%s': %s", fileLabel, reason));
  }
  
  static ASTFileLookupValue forBadFile(Label fileLabel) {
    return new ASTLookupNoFile(
        String.format("Unable to load file '%s': file doesn't exist or isn't a file", fileLabel));
  }
  
  public static ASTFileLookupValue withFile(BuildFileAST ast) {
    return new ASTLookupWithFile(ast);
  }

  static SkyKey key(Label astFileLabel) {
    return SkyKey.create(SkyFunctions.AST_FILE_LOOKUP, astFileLabel);
  }
}
