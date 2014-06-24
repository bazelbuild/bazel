// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.castList;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkCallable;

import java.util.Collection;
import java.util.Iterator;

/**
 * A wrapper class for NestedSet of Artifacts in Skylark to ensure type safety.
 */
@SkylarkBuiltin(name = "Files", doc = "A helper class to extract path from files.")
@Immutable
public final class SkylarkFileset implements Iterable<Artifact> {

  private final NestedSet<Artifact> fileset;

  @SuppressWarnings("unchecked")
  public SkylarkFileset(Order order, Iterable<Object> items) throws ConversionException {
    NestedSetBuilder<Artifact> builder = new NestedSetBuilder<>(order);
    for (Object item : items) {
      if (item instanceof Artifact) {
        builder.add((Artifact) item);
      } else if (item instanceof SkylarkFileset) {
        builder.addTransitive(((SkylarkFileset) item).fileset);
      } else if (item instanceof NestedSet<?>) {
        // TODO(bazel-team): This is here because of the conversion between the Java and the Skylark
        // rule implementations. When we have only Skylark this will be removed.
        builder.addTransitive((NestedSet<Artifact>) item);
      } else if (item instanceof Iterable<?>) {
        builder.addAll(castList(item, Artifact.class, "fileset item"));
      } else {
        throw new IllegalArgumentException(
            String.format("Invalid fileset item: %s", EvalUtils.getDatatypeName(item)));
      }
    }
    fileset = builder.build();
  }

  public NestedSet<Artifact> getFileset() {
    return fileset;
  }

  @Override
  public Iterator<Artifact> iterator() {
    return fileset.iterator();
  }

  @SkylarkCallable(doc = "Flattens this nested set of file to a list.")
  public Collection<Artifact> toCollection() {
    return fileset.toCollection();
  }

  @SkylarkCallable(doc = "Returns true if this file set is empty.")
  public boolean isEmpty() {
    return fileset.isEmpty();
  }

  @SkylarkCallable(doc = "Returns the relative path of this file relative to its root.")
  public static Object rootRelativePath(Artifact artifact) {
    return artifact.getRootRelativePath().getPathString();
  }

  @SkylarkCallable(doc = "Returns the execution path of this file.")
  public static Object execPath(Artifact artifact) {
    return artifact.getExecPathString();
  }
}
