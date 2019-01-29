// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.ExtraLinkTimeLibrary.BuildLibraryOutput;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A list of extra libraries to include in a link. These are non-C++ libraries that are built from
 * inputs gathered from all the dependencies. The dependencies have no way to coordinate, so each
 * one will add an ExtraLinkTimeLibrary to its CcLinkParams. ExtraLinkTimeLibrary is an interface,
 * and all ExtraLinkTimeLibrary objects of the same class will be gathered together.
 */
@AutoCodec
public final class ExtraLinkTimeLibraries {
  /**
   * We can have multiple different kinds of lists of libraries to include
   * at link time.  We map from the class type to an actual instance.
   */
  private final Collection<ExtraLinkTimeLibrary> extraLibraries;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  ExtraLinkTimeLibraries(Collection<ExtraLinkTimeLibrary> extraLibraries) {
    this.extraLibraries = extraLibraries;
  }

  /**
   * Return the set of extra libraries.
   */
  public Collection<ExtraLinkTimeLibrary> getExtraLibraries() {
    return extraLibraries;
  }

  public static final Builder builder() {
    return new Builder();
  }

  /**
   * Builder for {@link ExtraLinkTimeLibraries}.
   */
  public static final class Builder {
    private Map<Class<? extends ExtraLinkTimeLibrary>, ExtraLinkTimeLibrary.Builder> libraries =
          new LinkedHashMap<>();

    private Builder() {
      // Nothing to do.
    }

    /**
     * Build a {@link ExtraLinkTimeLibraries} object.
     */
    public ExtraLinkTimeLibraries build() {
      List<ExtraLinkTimeLibrary> extraLibraries = Lists.newArrayList();
      for (ExtraLinkTimeLibrary.Builder builder : libraries.values()) {
        extraLibraries.add(builder.build());
      }
      return new ExtraLinkTimeLibraries(extraLibraries);
    }

    /**
     * Add a transitive dependency.
     */
    public final Builder addTransitive(ExtraLinkTimeLibraries dep) {
      for (ExtraLinkTimeLibrary depLibrary : dep.getExtraLibraries()) {
        Class<? extends ExtraLinkTimeLibrary> c = depLibrary.getClass();
        libraries.computeIfAbsent(c, k -> depLibrary.getBuilder());
        libraries.get(c).addTransitive(depLibrary);
      }
      return this;
    }

    /**
     * Add a single library to build.
     */
    public final Builder add(ExtraLinkTimeLibrary b) {
      Class<? extends ExtraLinkTimeLibrary> c = b.getClass();
      libraries.computeIfAbsent(c, k -> b.getBuilder());
      libraries.get(c).addTransitive(b);
      return this;
    }
  }

  public BuildLibraryOutput buildLibraries(
      RuleContext ruleContext, boolean staticMode, boolean forDynamicLibrary)
      throws InterruptedException, RuleErrorException {
    NestedSetBuilder<LibraryToLinkWrapper> librariesToLink = NestedSetBuilder.linkOrder();
    NestedSetBuilder<Artifact> runtimeLibraries = NestedSetBuilder.linkOrder();
    for (ExtraLinkTimeLibrary extraLibrary : getExtraLibraries()) {
      BuildLibraryOutput buildLibraryOutput =
          extraLibrary.buildLibraries(ruleContext, staticMode, forDynamicLibrary);
      librariesToLink.addTransitive(buildLibraryOutput.getLibrariesToLink());
      runtimeLibraries.addTransitive(buildLibraryOutput.getRuntimeLibraries());
    }
    return new BuildLibraryOutput(librariesToLink.build(), runtimeLibraries.build());
  }
}
