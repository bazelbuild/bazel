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

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.NotComparableSkyValue;
import com.google.devtools.build.skyframe.SkyFunctionName;

/**
 * A value that represents an AST file lookup result. There are two subclasses: one for the case
 * where the file is found, and another for the case where the file is missing (but there are no
 * other errors).
 */
// In practice, if a ASTFileLookupValue is re-computed (i.e. not changed pruned), then it will
// almost certainly be unequal to the previous value. This is because of (i) the change-pruning
// semantics of the PackageLookupValue dep and the FileValue dep; consider the latter: if the
// FileValue for the bzl file has changed, then the contents of the bzl file probably changed and
// (ii) we don't currently have skylark-semantic-equality in BuildFileAST, so two BuildFileAST
// instances representing two different contents of a bzl file will be different.
// TODO(bazel-team): Consider doing better here. As a pre-req, we would need
// skylark-semantic-equality in BuildFileAST, rather than equality naively based on the contents of
// the bzl file. For a concrete example, the contents of comment lines do not currently impact
// skylark semantics.
public abstract class ASTFileLookupValue implements NotComparableSkyValue {
  public abstract boolean lookupSuccessful();
  public abstract BuildFileAST getAST();
  public abstract String getErrorMsg();

  /** If the file is found, this class encapsulates the parsed AST. */
  @AutoCodec.VisibleForSerialization
  public static class ASTLookupWithFile extends ASTFileLookupValue {
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
  @AutoCodec.VisibleForSerialization
  public static class ASTLookupNoFile extends ASTFileLookupValue {
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

  static ASTFileLookupValue forMissingFile(Label fileLabel) {
    return new ASTLookupNoFile(
        String.format("Unable to load file '%s': file doesn't exist", fileLabel));
  }

  static ASTFileLookupValue forBadFile(Label fileLabel) {
    return new ASTLookupNoFile(
        String.format("Unable to load file '%s': it isn't a regular file", fileLabel));
  }

  public static ASTFileLookupValue withFile(BuildFileAST ast) {
    return new ASTLookupWithFile(ast);
  }

  public static Key key(Label astFileLabel) {
    return ASTFileLookupValue.Key.create(astFileLabel);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<Label> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(Label arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(Label arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.AST_FILE_LOOKUP;
    }
  }
}
