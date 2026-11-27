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

/** Syntax node for a dot expression. e.g. obj.field, but not obj.method() */
public sealed class DotExpression extends Expression {

  private final Expression object;
  private final int dotOffset;
  private final String fieldName;

  static DotExpression create(
      FileLocations locs, Expression object, int dotOffset, Identifier field) {
    // The field name almost always follows the dot directly, in which case its offset isn't stored.
    return field.getStartOffset() == dotOffset + 1
        ? new DotExpression(locs, object, dotOffset, field.getName())
        : new SeparatedField(locs, object, dotOffset, field.getName(), field.getStartOffset());
  }

  private DotExpression(FileLocations locs, Expression object, int dotOffset, String fieldName) {
    super(locs);
    this.object = object;
    this.dotOffset = dotOffset;
    this.fieldName = fieldName;
  }

  public Expression getObject() {
    return object;
  }

  /** Returns a new Identifier node for the field name. */
  public Identifier getField() {
    return new Identifier(locs, fieldName, getFieldOffset());
  }

  public String getFieldName() {
    return fieldName;
  }

  @Override
  public int getStartOffset() {
    return object.getStartOffset();
  }

  @Override
  public int getEndOffset() {
    return getFieldOffset() + fieldName.length();
  }

  public Location getDotLocation() {
    return locs.getLocation(dotOffset);
  }

  int getFieldOffset() {
    return dotOffset + 1;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  /** A dot expression whose field name doesn't directly follow the dot. */
  private static final class SeparatedField extends DotExpression {
    private final int fieldOffset;

    private SeparatedField(
        FileLocations locs, Expression object, int dotOffset, String fieldName, int fieldOffset) {
      super(locs, object, dotOffset, fieldName);
      this.fieldOffset = fieldOffset;
    }

    @Override
    int getFieldOffset() {
      return fieldOffset;
    }
  }
}
