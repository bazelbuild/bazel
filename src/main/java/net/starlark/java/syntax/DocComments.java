// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import java.util.List;

/** A block of Sphinx autodoc-style doc comments. */
public final class DocComments {
  private final ImmutableList<Comment> lines;

  public DocComments(List<Comment> lines) {
    checkArgument(!lines.isEmpty());
    checkArgument(lines.stream().allMatch(Comment::hasDocCommentPrefix));
    this.lines = ImmutableList.copyOf(lines);
  }

  public ImmutableList<Comment> getLines() {
    return lines;
  }

  public Location getStartLocation() {
    return lines.getFirst().getStartLocation();
  }

  public Location getEndLocation() {
    return lines.getLast().getEndLocation();
  }

  /**
   * Returns the text content (trimmed of the leading {@code #: } or {@code #:} prefixes, and joined
   * with newlines) of the doc comment block.
   */
  public String getText() {
    return Joiner.on("\n").join(lines.stream().map(Comment::getDocCommentText).iterator());
  }

  @Override
  public String toString() {
    return Joiner.on("\n").join(lines.stream().map(Comment::toString).iterator());
  }
}
