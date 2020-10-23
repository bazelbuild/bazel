// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.tree.ClassNode;

/** Simple wrapper around a map that holds generated classes so they can be processed later. */
class GeneratedClassStore {

  /** Map from internal names to generated classes with deterministic iteration order. */
  private final Map<String, ClassNode> classes = new LinkedHashMap<>();

  /**
   * Adds a class for the given internal name. It's the caller's responsibility to {@link
   * ClassVisitor#visit} the returned object to initialize the desired class, and to avoid
   * confusion, this method throws if the class had already been present.
   */
  public ClassVisitor add(String internalClassName) {
    ClassNode result = new ClassNode();
    checkState(
        classes.put(internalClassName, result) == null, "Already present: %s", internalClassName);
    return result;
  }

  public ImmutableMap<String, ClassNode> drain() {
    ImmutableMap<String, ClassNode> result = ImmutableMap.copyOf(classes);
    classes.clear();
    return result;
  }
}
