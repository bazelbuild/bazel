// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen.starlark;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A typical Starlark documentation page, containing a bunch of field/method documentation entries.
 */
public abstract class StarlarkDocPage extends StarlarkDoc {
  // Contains all members; must be sorted for output - we cannot sort before output because
  // overloading can change a member doc's sort key.
  private final HashMultimap<String, MemberDoc> membersByShortName = HashMultimap.create();
  // Contains overloaded members; used only for uniqueness checks in overloadMember().
  private final HashMap<String, MemberDoc> overloadsBySignature = new HashMap<>();
  @Nullable private MemberDoc constructor;

  protected StarlarkDocPage(StarlarkDocExpander expander) {
    super(expander);
  }

  public abstract String getTitle();

  public void setConstructor(MemberDoc method) {
    checkArgument(method.isConstructor(), "Expected a constructor, got %s", method);
    checkState(
        constructor == null,
        "Constructor method doc already set for %s:\n  existing: %s\n  attempted: %s",
        getName(),
        constructor,
        method);
    constructor = method;
  }

  public void addMember(MemberDoc member) {
    if (!member.documented()) {
      return;
    }

    String shortName = member.getShortName();
    Set<MemberDoc> overloads = membersByShortName.get(shortName);
    if (!overloads.isEmpty()) {
      // Overload information only needs to be updated if we're discovering the first overload
      // (= the second method of the same name).
      if (overloads.size() == 1) {
        overloadMember(Iterables.getOnlyElement(overloads));
      }
      overloadMember(member);
    }
    membersByShortName.put(shortName, member);
  }

  private void overloadMember(MemberDoc member) {
    if (member instanceof AnnotStarlarkOrdinaryMethodDoc javaMethod) {
      javaMethod.setOverloaded(true);
      MemberDoc prevOverloadWithSameSignature = overloadsBySignature.put(member.getName(), member);
      if (prevOverloadWithSameSignature != null) {
        throw new IllegalStateException(
            String.format(
                "Starlark type '%s' has multiple overloads with signature %s: %s, %s",
                getName(), member.getName(), member, prevOverloadWithSameSignature));
      }
    } else {
      throw new IllegalArgumentException(
          "Only non-constructor Java-defined methods can be overloaded; got " + member);
    }
  }

  /**
   * Returns the list of members of this doc page,; first the constructor method (if one is
   * defined), and then the remaining methods in case-insensitive name order.
   */
  public ImmutableList<MemberDoc> getMembers() {
    ImmutableList.Builder<MemberDoc> members = ImmutableList.builder();
    if (constructor != null) {
      members.add(constructor);
    }
    // membersByShortName is a hash map,
    return members
        .addAll(
            ImmutableList.sortedCopyOf(
                Comparator.comparing(m -> m.getName().toLowerCase(Locale.ROOT)),
                membersByShortName.values()))
        .build();
  }

  @Nullable
  public MemberDoc getConstructor() {
    return constructor;
  }

  /** Returns the path to the source file backing this doc page. */
  // This method may seem unused, but it's actually used in the template file (starlark-library.vm).
  public abstract String getSourceFile();
}
