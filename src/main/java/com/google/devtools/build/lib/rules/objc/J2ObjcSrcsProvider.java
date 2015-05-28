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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;

/**
 * This provider is exported by java_library rules to supply J2ObjC-translated ObjC sources to
 * objc_binary for compilation and linking.
 */
@Immutable
public final class J2ObjcSrcsProvider implements TransitiveInfoProvider {
  private final NestedSet<J2ObjcSource> srcs;
  private final NestedSet<String> entryClasses;
  private final boolean hasProtos;

  /**
   * A builder for J2ObjCSrcsProvider.
   */
  public static class Builder {
    private final NestedSetBuilder<J2ObjcSource> srcsBuilder = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<String> entryClassesBuilder = NestedSetBuilder.stableOrder();
    private boolean hasProtos = false;

    /**
     * Constructs a new, empty J2ObjCSrcsProvider builder.
     */
    public Builder() {}

    /**
     * Transitively adds the given {@link J2ObjcSrcsProvider}
     * and all its properties to this builder.
     *
     * @param provider the J2ObjcSrcsProvider to add
     * @return this builder
     */
    public Builder addTransitive(J2ObjcSrcsProvider provider) {
      srcsBuilder.addTransitive(provider.getSrcs());
      entryClassesBuilder.addTransitive(provider.getEntryClasses());
      hasProtos |= provider.hasProtos();
      return this;
    }

    /**
     * Transitively adds all the J2ObjcSrcsProviders and all their properties
     * that can be reached through the "deps" attribute of the given RuleContext.
     *
     * @param ruleContext the rule context in which to look for deps
     * @return this builder
     */
    public Builder addTransitiveFromDeps(RuleContext ruleContext) {
      if (ruleContext.attributes().has("deps", Type.LABEL_LIST)) {
        for (J2ObjcSrcsProvider provider :
            ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcSrcsProvider.class)) {
          addTransitive(provider);
        }
      }

      return this;
    }

    /**
     * Adds the given {@link J2ObjcSource} to this builder.
     *
     * @param source the source to add
     * @return this builder
     */
    public Builder addSource(J2ObjcSource source) {
      srcsBuilder.add(source);
      hasProtos |= source.getSourceType() == SourceType.PROTO;
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
     * Builds a J2ObjCSrcsProvider from the information in this builder.
     *
     * @return the J2ObjCSrcsProvider to be built
     */
    public J2ObjcSrcsProvider build() {
      return new J2ObjcSrcsProvider(srcsBuilder.build(), entryClassesBuilder.build(), hasProtos);
    }
  }

  /**
   * Constructs a new J2ObjCSrcsProvider that contains all the information
   * that can be transitively reached through the "deps" attribute of the given rule context.
   *
   * @param ruleContext the rule context in which to look for deps
   */
  public static J2ObjcSrcsProvider buildFrom(RuleContext ruleContext) {
    return new Builder().addTransitiveFromDeps(ruleContext).build();
  }

  /**
   * Constructs a {@link J2ObjcSrcsProvider} to supply J2ObjC-translated ObjC sources to
   * objc_binary for compilation and linking.
   *
   * @param srcs a nested set of {@link J2ObjcSource}s containing translated source files
   * @param entryClasses a set of names of Java classes to used as entry point for J2ObjC dead code
   *     analysis. The Java class names should be in canonical format as defined by the Java
   *     Language Specification.
   * @param hasProtos whether the translated files in this provider have J2ObjC proto files
   */
  private J2ObjcSrcsProvider(NestedSet<J2ObjcSource> srcs, NestedSet<String> entryClasses,
      boolean hasProtos) {
    this.srcs = srcs;
    this.entryClasses = entryClasses;
    this.hasProtos = hasProtos;
  }

  public NestedSet<J2ObjcSource> getSrcs() {
    return srcs;
  }

  /**
   * Returns a set of entry classes specified on attribute entry_classes of j2objc_library targets
   * transitively.
   */
  public NestedSet<String> getEntryClasses() {
    return entryClasses;
  }

  /**
   * Returns whether the translated source files in the provider has proto files.
   */
  public boolean hasProtos() {
    return hasProtos;
  }
}
