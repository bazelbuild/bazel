// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion.Kind;
import java.util.Objects;

/**
 * An {@link Inclusion} together with the context where on the include path the inclusion was found,
 * and whether the containing file was included using angle brackets or quotes.
 */
class InclusionWithContext {
  private final Inclusion inclusion;
  private final int contextPathPos;
  private final Kind contextKind;

  /**
   * Attaches context to an inclusion.
   *
   * @param inclusion the inclusion
   * @param contextPathPos the position on the include path on which the containing file was found.
   *        Used directly only for {@code #include_next} inclusions, but stored for all inclusions
   *        so that include_next inclusions found inside this one can have proper context.
   * @param contextKind how the containing file was included. Used only for include_next inclusions.
   *        Must not be a {@link Kind#NEXT_ANGLE} or {@link Kind#NEXT_QUOTE}
   */
  InclusionWithContext(Inclusion inclusion, int contextPathPos, Kind contextKind) {
    this.inclusion = Preconditions.checkNotNull(inclusion);

    Preconditions.checkArgument(contextKind == null || !contextKind.isNext(), inclusion);

    this.contextPathPos = contextPathPos;
    // The context kind is only stored for #include_next inclusions.
    if (this.inclusion.kind.isNext()) {
      this.contextKind = contextKind;
    } else {
      this.contextKind = this.inclusion.kind;
    }
  }

  /**
   * Creates a simple {@link Kind#QUOTE} or {@link Kind#ANGLE} inclusion with empty context.
   *
   * @param name the name of the included file
   * @param kind the kind of the inclusion, must not be a {@code
   *        #include_next} inclusion
   */
  InclusionWithContext(String name, Kind kind) {
    this(Inclusion.create(name, kind), -1, null);
  }

  Inclusion getInclusion() {
    return inclusion;
  }

  /**
   * The position on the include path on which the containing file was found. Local inclusions
   * correspond conceptually to the first entry on the include, so the values are used like this:
   * <ul>
   * <li>-1: top-level or not a {@code #include_next} inclusion,
   * <li>0: {@code #include_next} inclusion and locally found header,
   * <li>>0: {@code #include_next} inclusion and found on the include path.
   * </ul>
   */
  int getContextPathPos() {
    return contextPathPos;
  }

  /**
   * On which include path to continue searching. For {@link Kind#QUOTE} and {@link Kind#ANGLE}
   * inclusions this is the inclusion kind itself. For {@link Kind#NEXT_QUOTE} and
   * {@link Kind#NEXT_ANGLE} inclusion it is the kind of the last inclusion that was not a
   * {@code #include_next} inclusion.
   */
  Kind getContextKind() {
    return contextKind;
  }

  @Override
  public String toString() {
    return inclusion.kind.isNext()
        ? inclusion + "(" + contextKind + ":" + contextPathPos + ")"
        : inclusion.toString();
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof InclusionWithContext)) {
      return false;
    }
    InclusionWithContext that = (InclusionWithContext) o;
    return Objects.equals(this.inclusion, that.inclusion)
        && (!this.inclusion.kind.isNext() || this.contextPathPos == that.contextPathPos)
        && this.contextKind == that.contextKind;
  }

  @Override
  public int hashCode() {
    int result = 1;
    result = 31 * result + inclusion.hashCode();
    result = 31 * result + (inclusion.kind.isNext() ? Integer.hashCode(contextPathPos) : 0);
    result = 31 * result + (contextKind != null ? contextKind.hashCode() : 0);
    return result;
  }

}
