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

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import java.text.Collator;
import java.util.Collection;
import java.util.Comparator;
import java.util.Locale;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A typical Starlark documentation page, containing a bunch of field/method documentation entries.
 */
public abstract class StarlarkDocPage extends StarlarkDoc {
  private final Multimap<String, StarlarkJavaMethodDoc> javaMethods;
  private final TreeMap<String, StarlarkMethodDoc> methodMap;
  @Nullable private StarlarkConstructorMethodDoc javaConstructor;

  protected StarlarkDocPage(StarlarkDocExpander expander) {
    super(expander);
    this.methodMap = new TreeMap<>(Collator.getInstance(Locale.US));
    this.javaMethods = HashMultimap.create();
  }

  public abstract String getTitle();

  public void setConstructor(StarlarkConstructorMethodDoc method) {
    Preconditions.checkState(
        javaConstructor == null,
        "Constructor method doc already set for %s:\n  existing: %s\n  attempted: %s",
        getName(),
        javaConstructor,
        method);
    javaConstructor = method;
  }

  public void addMethod(StarlarkJavaMethodDoc method) {
    if (!method.documented()) {
      return;
    }

    String shortName = method.getName();
    Collection<StarlarkJavaMethodDoc> overloads = javaMethods.get(shortName);
    if (!overloads.isEmpty()) {
      method.setOverloaded(true);
      // Overload information only needs to be updated if we're discovering the first overload
      // (= the second method of the same name).
      if (overloads.size() == 1) {
        Iterables.getOnlyElement(overloads).setOverloaded(true);
      }
    }
    javaMethods.put(shortName, method);

    // If the method is overloaded, getName() now returns a longer,
    // unique name including the names of the parameters.
    StarlarkMethodDoc prev = methodMap.put(method.getName(), method);
    if (prev != null && !prev.getMethod().equals(method.getMethod())) {
      throw new IllegalStateException(
          String.format(
              "Starlark type '%s' has distinct overloads of %s: %s, %s",
              getName(), method.getName(), method.getMethod(), prev.getMethod()));
    }
  }

  public Collection<StarlarkMethodDoc> getJavaMethods() {
    ImmutableSortedSet.Builder<StarlarkMethodDoc> methods =
        new ImmutableSortedSet.Builder<>(Comparator.comparing(StarlarkMethodDoc::getName));

    if (javaConstructor != null) {
      methods.add(javaConstructor);
    }
    methods.addAll(javaMethods.values());
    return methods.build();
  }

  public ImmutableCollection<? extends StarlarkMethodDoc> getMethods() {
    ImmutableList.Builder<StarlarkMethodDoc> methods = ImmutableList.builder();
    if (javaConstructor != null) {
      methods.add(javaConstructor);
    }
    return methods.addAll(methodMap.values()).build();
  }

  @Nullable
  public StarlarkConstructorMethodDoc getConstructor() {
    return javaConstructor;
  }

  /** Returns the path to the source file backing this doc page. */
  // This method may seem unused, but it's actually used in the template file (starlark-library.vm).
  public abstract String getSourceFile();
}
