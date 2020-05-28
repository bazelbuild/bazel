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
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.NotComparableSkyValue;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.errorprone.annotations.FormatMethod;

/**
 * A value that represents an AST file lookup result. There are two subclasses: one for the case
 * where the file is found, and another for the case where the file is missing (but there are no
 * other errors).
 */
// In practice, almost any change to a .bzl causes the ASTFileLookupValue to be recomputed.
// We could do better with a finer-grained notion of equality for StarlarkFile than "the source
// files differ". In particular, a trivial change such as fixing a typo in a comment should not
// cause invalidation. (Changes that are only slightly more substantial may be semantically
// significant. For example, inserting a blank line affects subsequent line numbers, which appear
// in error messages and query output.)
//
// Comparing syntax trees for equality is complex and expensive, so the most practical
// implementation of this optimization will have to wait until Starlark files are compiled,
// at which point byte-equality of the compiled representation (which is simple to compute)
// will serve. (At that point, ASTFileLookup should be renamed CompileStarlark.)
//
public abstract class ASTFileLookupValue implements NotComparableSkyValue {

  // TODO(adonovan): flatten this hierarchy into a single class.
  // It would only cost one word per Starlark file.
  // Eliminate lookupSuccessful; use getAST() != null.

  public abstract boolean lookupSuccessful();

  public abstract StarlarkFile getAST();

  public abstract byte[] getDigest();

  public abstract String getError();

  /** If the file is found, this class encapsulates the parsed AST. */
  @AutoCodec.VisibleForSerialization
  public static class ASTLookupWithFile extends ASTFileLookupValue {
    private final StarlarkFile ast;
    private final byte[] digest;

    private ASTLookupWithFile(StarlarkFile ast, byte[] digest) {
      this.ast = Preconditions.checkNotNull(ast);
      this.digest = Preconditions.checkNotNull(digest);
    }

    @Override
    public boolean lookupSuccessful() {
      return true;
    }

    @Override
    public StarlarkFile getAST() {
      return this.ast;
    }

    @Override
    public byte[] getDigest() {
      return this.digest;
    }

    @Override
    public String getError() {
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
    public StarlarkFile getAST() {
      throw new IllegalStateException("attempted to retrieve AST from an unsuccessful lookup");
    }

    @Override
    public byte[] getDigest() {
      throw new IllegalStateException("attempted to retrieve digest for successful lookup");
    }

    @Override
    public String getError() {
      return this.errorMsg;
    }
  }

  /** Constructs a value from a failure before parsing a file. */
  @FormatMethod
  static ASTFileLookupValue noFile(String format, Object... args) {
    return new ASTLookupNoFile(String.format(format, args));
  }

  /** Constructs a value from a parsed file. */
  public static ASTFileLookupValue withFile(StarlarkFile ast, byte[] digest) {
    return new ASTLookupWithFile(ast, digest);
  }

  public static Key key(Label label) {
    return ASTFileLookupValue.Key.create(label);
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
