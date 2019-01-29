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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Syntax node for a function argument.
 *
 * <p>Argument is a base class for arguments passed in a call (@see Argument.Passed)
 * or defined as part of a function definition (@see Parameter).
 * It is notably used by some {@link Parser} and printer functions.
 */
public abstract class Argument extends ASTNode {

  public boolean isStar() {
    return false;
  }

  public boolean isStarStar() {
    return false;
  }

  /**
   * Argument.Passed is the class of arguments passed in a function call
   * (as opposed to being used in a definition -- @see Parameter for that).
   * Argument.Passed is usually what we mean when informally say "argument".
   *
   * <p>An Argument.Passed can be Positional, Keyword, Star, or StarStar.
   */
  public abstract static class Passed extends Argument {
    /** the value to be passed by this argument */
    protected final Expression value;

    private Passed(Expression value) {
      this.value = Preconditions.checkNotNull(value);
    }

    public boolean isPositional() {
      return false;
    }

    public boolean isKeyword() {
      return false;
    }

    /** @deprecated Prefer {@link #getIdentifier()} instead. */
    @Deprecated
    @Nullable
    public String getName() { // only for keyword arguments
      return null;
    }

    @Nullable
    public Identifier getIdentifier() {
      return null;
    }

    public Expression getValue() {
      return value;
    }

    @Override
    public void accept(SyntaxTreeVisitor visitor) {
      visitor.visit(this);
    }
  }

  /** positional argument: Expression */
  public static final class Positional extends Passed {

    public Positional(Expression value) {
      super(value);
    }

    @Override
    public boolean isPositional() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      value.prettyPrint(buffer);
    }
  }

  /** keyword argument: K = Expression */
  public static final class Keyword extends Passed {

    final Identifier identifier;

    public Keyword(Identifier identifier, Expression value) {
      super(value);
      this.identifier = identifier;
    }

    @Override
    public String getName() {
      return identifier.getName();
    }

    @Override
    public Identifier getIdentifier() {
      return identifier;
    }

    @Override
    public boolean isKeyword() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append(identifier.getName());
      buffer.append(" = ");
      value.prettyPrint(buffer);
    }
  }

  /** positional rest (starred) argument: *Expression */
  public static final class Star extends Passed {

    public Star(Expression value) {
      super(value);
    }

    @Override
    public boolean isStar() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append('*');
      value.prettyPrint(buffer);
    }
  }

  /** keyword rest (star_starred) parameter: **Expression */
  public static final class StarStar extends Passed {

    public StarStar(Expression value) {
      super(value);
    }

    @Override
    public boolean isStarStar() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append("**");
      value.prettyPrint(buffer);
    }
  }

  /** Some arguments failed to satisfy python call convention strictures */
  static class ArgumentException extends Exception {
    Location location;

    /** construct an ArgumentException from a message only */
    ArgumentException(Location location, String message) {
      super(message);
      this.location = location;
    }

    Location getLocation() {
      return location;
    }
  }

  /**
   * Validate that the list of Argument's, whether gathered by the Parser or from annotations,
   * satisfies the requirements of the Python calling conventions: all Positional's first, at most
   * one Star, at most one StarStar, at the end only.
   *
   * <p>TODO(laurentlb): remove this function and use only validateFuncallArguments.
   */
  public static void legacyValidateFuncallArguments(List<Passed> arguments)
      throws ArgumentException {
    boolean hasNamed = false;
    boolean hasStar = false;
    boolean hasKwArg = false;
    for (Passed arg : arguments) {
      if (hasKwArg) {
        throw new ArgumentException(arg.getLocation(), "argument after **kwargs");
      }
      if (arg.isPositional()) {
        if (hasNamed) {
          throw new ArgumentException(arg.getLocation(), "non-keyword arg after keyword arg");
        } else if (arg.isStar()) {
          throw new ArgumentException(
              arg.getLocation(), "only named arguments may follow *expression");
        }
      } else if (arg.isKeyword()) {
        hasNamed = true;
      } else if (arg.isStar()) {
        if (hasStar) {
          throw new ArgumentException(arg.getLocation(), "more than one *stararg");
        }
        hasStar = true;
      } else {
        hasKwArg = true;
      }
    }
  }

  /**
   * Validate that the list of Argument's, whether gathered by the Parser or from annotations,
   * satisfies the requirements: first Positional arguments, then Keyword arguments, then an
   * optional *arg argument, finally an optional **kwarg argument.
   */
  public static void validateFuncallArguments(List<Passed> arguments) throws ArgumentException {
    int i = 0;
    int len = arguments.size();

    while (i < len && arguments.get(i).isPositional()) {
      i++;
    }

    while (i < len && arguments.get(i).isKeyword()) {
      i++;
    }

    if (i < len && arguments.get(i).isStar()) {
      i++;
    }

    if (i < len && arguments.get(i).isStarStar()) {
      i++;
    }

    // If there's no argument left, everything is correct.
    if (i == len) {
      return;
    }

    Location loc = arguments.get(i).getLocation();
    if (arguments.get(i).isPositional()) {
      throw new ArgumentException(
          loc, "positional argument is misplaced (positional arguments come first)");
    }

    if (arguments.get(i).isKeyword()) {
      throw new ArgumentException(
          loc,
          "keyword argument is misplaced (keyword arguments must be before any *arg or **kwarg)");
    }

    if (i < len && arguments.get(i).isStar()) {
      throw new ArgumentException(loc, "*arg argument is misplaced");
    }

    if (i < len && arguments.get(i).isStarStar()) {
      throw new ArgumentException(loc, "**kwarg argument is misplaced (there can be only one)");
    }
  }

  @Override
  public final void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    prettyPrint(buffer);
  }

  @Override
  public abstract void prettyPrint(Appendable buffer) throws IOException;
}
