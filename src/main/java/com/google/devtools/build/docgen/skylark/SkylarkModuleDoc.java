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
package com.google.devtools.build.docgen.skylark;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

/**
 * A class representing a Skylark built-in object with its {@link SkylarkSignature} annotation
 * and the {@link SkylarkCallable} methods it might have.
 */
public final class SkylarkModuleDoc extends SkylarkDoc {
  private final SkylarkModule module;
  private final Class<?> classObject;
  private final Map<String, SkylarkBuiltinMethodDoc> builtinMethodMap;
  private final Multimap<String, SkylarkJavaMethodDoc> javaMethods;
  private TreeMap<String, SkylarkMethodDoc> methodMap;
  private final String title;

  public SkylarkModuleDoc(SkylarkModule module, Class<?> classObject) {
    this.module = Preconditions.checkNotNull(
        module, "Class has to be annotated with SkylarkModule: %s", classObject);
    this.classObject = classObject;
    this.builtinMethodMap = new TreeMap<>();
    this.methodMap = new TreeMap<>();
    this.javaMethods = HashMultimap.<String, SkylarkJavaMethodDoc>create();
    if (module.title().isEmpty()) {
      this.title = module.name();
    } else {
      this.title = module.title();
    }
  }

  @Override
  public String getName() {
    return module.name();
  }

  @Override
  public String getDocumentation() {
    return module.doc();
  }

  public String getTitle() {
    return title;
  }

  public SkylarkModule getAnnotation() {
    return module;
  }

  public Class<?> getClassObject() {
    return classObject;
  }

  public void addMethod(SkylarkBuiltinMethodDoc method) {
    methodMap.put(method.getName(), method);
    builtinMethodMap.put(method.getName(), method);
  }

  public void addMethod(SkylarkJavaMethodDoc method) {
    if (!method.documented()) {
      return;
    }

    String shortName = method.getName();
    Collection<SkylarkJavaMethodDoc> overloads = javaMethods.get(shortName);
    if (!overloads.isEmpty()) {
      method.setOverloaded(true);
      // Overload information only needs to be updated if we're discovering the first overload
      // (= the second method of the same name).
      if (overloads.size() == 1) {
        Iterables.getOnlyElement(overloads).setOverloaded(true);
      }
    }
    javaMethods.put(shortName, method);
    // If the method is overloaded, getName() now returns a longer, unique name including
    // the names of the parameters.
    String uniqueName = method.getName();
    Preconditions.checkState(
        !methodMap.containsKey(uniqueName),
        "There are multiple overloads of %s with the same signature!",
        uniqueName);
    methodMap.put(uniqueName, method);
  }

  public boolean javaMethodsNotCollected() {
    return javaMethods.isEmpty();
  }

  public Map<String, SkylarkBuiltinMethodDoc> getBuiltinMethods() {
    return builtinMethodMap;
  }

  public Collection<SkylarkJavaMethodDoc> getJavaMethods() {
    return javaMethods.values();
  }

  public Collection<SkylarkMethodDoc> getMethods() {
    return methodMap.values();
  }
}
