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
package net.starlark.java.syntax;

/** Base class for all statements nodes in the AST. */
public abstract class Statement extends Node {

  /**
   * Kind of the statement. This is similar to using instanceof, except that it's more efficient and
   * can be used in a switch/case.
   */
  public enum Kind {
    ASSIGNMENT,
    EXPRESSION,
    FLOW,
    FOR,
    DEF,
    IF,
    LOAD,
    RETURN,
    TYPE_ALIAS,
    VAR,
  }

  // The ordinals of the kinds as constants, so that the interpreter can switch on kindOrdinal()
  // without loading the Kind.
  public static final int ASSIGNMENT = 0;
  public static final int EXPRESSION = 1;
  public static final int FLOW = 2;
  public static final int FOR = 3;
  public static final int DEF = 4;
  public static final int IF = 5;
  public static final int LOAD = 6;
  public static final int RETURN = 7;
  public static final int TYPE_ALIAS = 8;
  public static final int VAR = 9;

  private static final Kind[] KINDS = Kind.values();

  // The kind and an optional offset, packed by packKindAndOffset.
  private final int kindAndOffset;

  Statement(FileLocations locs, Kind kind) {
    super(locs);
    this.kindAndOffset = kind.ordinal();
  }

  /** Also stores an offset that subclasses can retrieve with {@link #packedOffset}. */
  Statement(FileLocations locs, Kind kind, int offset) {
    super(locs);
    this.kindAndOffset = packKindAndOffset(kind, offset);
  }

  /**
   * Kind of the statement. This is similar to using instanceof, except that it's more efficient and
   * can be used in a switch/case.
   */
  // Final to avoid cost of virtual call (see #12967).
  public final Kind kind() {
    return KINDS[kindOrdinal()];
  }

  /** Returns the ordinal of {@link #kind}, which equals the int constant of the same name. */
  public final int kindOrdinal() {
    return kindAndOffset & ((1 << KIND_BITS) - 1);
  }

  /** Returns the offset passed to the constructor. */
  final int packedOffset() {
    return kindAndOffset >>> KIND_BITS;
  }
}
