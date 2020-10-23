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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;

/**
 * This provider is exported by j2objc_library to export entry class information necessary for
 * J2ObjC dead code removal performed at the binary level in ObjC rules.
 */
@Immutable
public final class J2ObjcEntryClassProvider implements TransitiveInfoProvider {
  private final NestedSet<String> entryClasses;

  /**
   * A builder for J2ObjcEntryClassProvider.
   */
  public static class Builder {
    private final NestedSetBuilder<String> entryClassesBuilder = NestedSetBuilder.stableOrder();

    /**
     * Constructs a new, empty J2ObjcEntryClassProvider builder.
     */
    public Builder() {}

    /**
     * Transitively adds the given {@link J2ObjcEntryClassProvider}
     * and all its properties to this builder.
     *
     * @param provider the J2ObjcEntryClassProvider to add
     * @return this builder
     */
    public Builder addTransitive(J2ObjcEntryClassProvider provider) {
      entryClassesBuilder.addTransitive(provider.getEntryClasses());
      return this;
    }

    /**
     * Transitively adds the given {@link J2ObjcEntryClassProvider}s
     * and all their properties to this builder.
     *
     * @param providers the J2ObjcEntryClassProviders to add
     * @return this builder
     */
    public Builder addTransitive(Iterable<J2ObjcEntryClassProvider> providers) {
      for (J2ObjcEntryClassProvider provider : providers) {
        addTransitive(provider);
      }
      return this;
    }

    /**
     * Transitively adds all the J2ObjcEntryClassProviders and all their properties
     * that can be reached through the "deps" attribute.
     *
     * @param ruleContext the rule context
     * @return this builder
     */
    public Builder addTransitive(RuleContext ruleContext) {
      if (ruleContext.attributes().has("deps", BuildType.LABEL_LIST)) {
        addTransitive(ruleContext.getPrerequisites("deps", J2ObjcEntryClassProvider.class));
      }

      return this;
    }

    /**
     * Adds the given entry classes to this builder. See {@link #getEntryClasses()}.
     *
     * @param entryClasses the entry classes to add
     * @return this builder
     */
    public Builder addEntryClasses(Iterable<String> entryClasses) {
      entryClassesBuilder.addAll(entryClasses);
      return this;
    }

    /**
     * Builds a J2ObjcEntryClassProvider from the information in this builder.
     *
     * @return the J2ObjcEntryClassProvider to be built
     */
    public J2ObjcEntryClassProvider build() {
      return new J2ObjcEntryClassProvider(entryClassesBuilder.build());
    }
  }

  /**
   * Constructs a {@link J2ObjcEntryClassProvider} to supply J2ObjC-translated ObjC sources to
   * objc_binary for compilation and linking.
   *
   * @param entryClasses a set of names of Java classes to used as entry point for J2ObjC dead code
   *     analysis. The Java class names should be in canonical format as defined by the Java
   *     Language Specification.
   */
  private J2ObjcEntryClassProvider(NestedSet<String> entryClasses) {
    this.entryClasses = entryClasses;
  }

  /**
   * Returns a set of entry classes specified on attribute entry_classes of j2objc_library targets
   * transitively.
   */
  public NestedSet<String> getEntryClasses() {
    return entryClasses;
  }
}
