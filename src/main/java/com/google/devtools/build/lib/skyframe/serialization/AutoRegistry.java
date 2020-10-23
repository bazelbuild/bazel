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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import java.io.IOException;
import java.util.Comparator;

/**
 * A lazy, automatically populated registry.
 *
 * <p>Must not be accessed by any {@link CodecRegisterer} or {@link ObjectCodec} constructors or
 * static initializers.
 */
public class AutoRegistry {

  private static final Supplier<ObjectCodecRegistry> SUPPLIER =
      Suppliers.memoize(AutoRegistry::create);

  // Build codecs only for Bazel and Starlark classes.
  private static boolean packageFilter(String name) {
    return name.startsWith("com.google.devtools.build")
        || name.startsWith("com.google.devtools.common.options") // e.g. for Tristate
        || name.startsWith("net.starlark.java");
  }

  /** Class name prefixes to blacklist for {@link DynamicCodec}. */
  private static final ImmutableList<String> CLASS_NAME_PREFIX_BLACKLIST =
      ImmutableList.of(
          "com.google.devtools.build.lib.google",
          "com.google.devtools.build.lib.vfs",
          "com.google.devtools.build.lib.bazel.rules.ninja",
          "com.google.devtools.build.lib.actions.ArtifactFactory",
          "com.google.devtools.build.lib.packages.PackageFactory$BuiltInRuleFunction",
          "com.google.devtools.build.skyframe.SkyFunctionEnvironment");

  /** Classes outside {@link AutoRegistry#PACKAGE_PREFIX} that need to be serialized. */
  private static final ImmutableList<String> EXTERNAL_CLASS_NAMES_TO_REGISTER =
      ImmutableList.of(
          "java.io.FileNotFoundException",
          "java.io.IOException",
          "java.lang.StackTraceElement",
          "java.lang.invoke.SerializedLambda",
          "com.google.common.base.Predicates$InPredicate",
          // Sadly, these builders are serialized as part of StarlarkCustomCommandLine$Builder,
          // which apparently can be preserved through analysis. We may investigate if this actually
          // has performance/correctness implications.
          "com.google.common.collect.ImmutableList$Builder");

  private static final ImmutableList<Object> REFERENCE_CONSTANTS_TO_REGISTER =
      ImmutableList.of(
          Predicates.alwaysTrue(),
          Predicates.alwaysFalse(),
          Predicates.isNull(),
          Predicates.notNull(),
          ImmutableList.of(),
          ImmutableSet.of(),
          Comparator.naturalOrder(),
          Ordering.natural());

  public static ObjectCodecRegistry get() {
    return SUPPLIER.get();
  }

  private static ObjectCodecRegistry create() {
    try {
      ObjectCodecRegistry.Builder registry =
          CodecScanner.initializeCodecRegistry(AutoRegistry::packageFilter);
      for (String className : EXTERNAL_CLASS_NAMES_TO_REGISTER) {
        registry.addClassName(className);
      }
      for (Object constant : REFERENCE_CONSTANTS_TO_REGISTER) {
        registry.addReferenceConstant(constant);
      }
      for (String classNamePrefix : CLASS_NAME_PREFIX_BLACKLIST) {
        registry.blacklistClassNamePrefix(classNamePrefix);
      }
      return registry.build();
    } catch (IOException | ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }
}
