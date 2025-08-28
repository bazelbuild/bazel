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

import javax.annotation.Nullable;

/** Syntax node for comments. */
public final class Comment extends Node {

  private final int offset;
  private final String text;

  Comment(FileLocations locs, int offset, String text) {
    super(locs);
    this.offset = offset;
    this.text = text;
  }

  /** Returns the text of the comment, including the leading '#' but not the trailing newline. */
  public String getText() {
    return text;
  }

  /**
   * Returns true if the comment starts with {@code #:}, like a Sphinx autodoc-style doc comment.
   */
  public boolean hasDocCommentPrefix() {
    return text.startsWith("#:");
  }

  /**
   * If the comment starts with a {@code #: } or {@code #:} prefix, returns the text following it;
   * otherwise, returns null.
   */
  @Nullable
  public String getDocCommentText() {
    if (hasDocCommentPrefix()) {
      return text.startsWith("#: ") ? text.substring(3) : text.substring(2);
    }
    return null;
  }

  @Override
  public int getStartOffset() {
    return offset;
  }

  @Override
  public int getEndOffset() {
    return offset + text.length();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public String toString() {
    return text;
  }
}
