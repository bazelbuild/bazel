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
package com.google.devtools.build.docgen.starlark;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import java.text.Collator;
import java.util.Collection;
import java.util.Locale;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/**
 * A class representing documentation for a Starlark built-in object with its {@link StarlarkModule}
 * annotation and with the {@link StarlarkMethod} methods it documents.
 */
public final class StarlarkBuiltinDoc extends StarlarkDoc {
  private final StarlarkBuiltin module;
  private final String title;
  private final Class<?> classObject;
  private final Multimap<String, StarlarkJavaMethodDoc> javaMethods;
  private TreeMap<String, StarlarkMethodDoc> methodMap;
  @Nullable private StarlarkConstructorMethodDoc javaConstructor;

  public StarlarkBuiltinDoc(StarlarkBuiltin module, String title, Class<?> classObject) {
    this.module =
        Preconditions.checkNotNull(
            module, "Class has to be annotated with StarlarkBuiltin: %s", classObject);
    this.title = title;
    this.classObject = classObject;
    this.methodMap = new TreeMap<>(Collator.getInstance(Locale.US));
    this.javaMethods = HashMultimap.<String, StarlarkJavaMethodDoc>create();
  }

  @Override
  public String getName() {
    return module.name();
  }

  @Override
  public String getDocumentation() {
    return StarlarkDocUtils.substituteVariables(module.doc());
  }

  public String getTitle() {
    return title;
  }

  public StarlarkBuiltin getAnnotation() {
    return module;
  }

  public Class<?> getClassObject() {
    return classObject;
  }

  public void setConstructor(StarlarkConstructorMethodDoc method) {
    Preconditions.checkState(javaConstructor == null);
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
              "Starlark type '%s' (%s) has distinct overloads of %s: %s, %s",
              module.name(), classObject, method.getName(), method.getMethod(), prev.getMethod()));
    }
  }

  public boolean javaMethodsNotCollected() {
    return javaMethods.isEmpty();
  }

  public Collection<StarlarkMethodDoc> getJavaMethods() {
    ImmutableList.Builder<StarlarkMethodDoc> returnedMethods = ImmutableList.builder();
    if (javaConstructor != null) {
      returnedMethods.add(javaConstructor);
    }
    return returnedMethods.addAll(javaMethods.values()).build();
  }

  public ImmutableCollection<? extends StarlarkMethodDoc> getMethods() {
    ImmutableList.Builder<StarlarkMethodDoc> methods = ImmutableList.builder();
    if (javaConstructor != null) {
      methods.add(javaConstructor);
    }
    return methods.addAll(methodMap.values()).build();
  }
}
