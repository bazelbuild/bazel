// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkLateBoundDefault;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.DomainSpecificTraverser;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.Traversal;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.lang.reflect.Field;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

final class BuildObjectTraverser implements DomainSpecificTraverser {
  private final boolean reportConfiguration;

  public BuildObjectTraverser(boolean reportConfiguration) {
    this.reportConfiguration = reportConfiguration;
  }

  @Override
  public boolean maybeTraverse(Object o, Traversal traversal) {
    switch (o) {
      case Path p -> {
        traversal.objectFound(o, null);
        traversal.edgeFound(p.getPathString(), null);
        return true;
      }

      case PathFragment pf -> {
        traversal.objectFound(o, null);
        traversal.edgeFound(pf.getPathString(), null);
        return true;
      }

      default -> {
        return false;
      }
    }
  }

  @Override
  public boolean admit(Object o) {
    if (!reportConfiguration) {
      if (o instanceof BuildConfigurationValue) {
        return false;
      }

      if (o instanceof BuildConfigurationKey) {
        return false;
      }
    }

    if (o instanceof RuleClass) {
      return false;
    }

    if (o instanceof Provider) {
      return false;
    }

    if (o instanceof com.google.devtools.build.lib.packages.Type) {
      // These are BUILD types and are all singletons
      return false;
    }

    if (o instanceof StarlarkLateBoundDefault) {
      // These are cached and thus not assignable to individual Skyframe objects
      return false;
    }

    if (o instanceof PackageIdentifier) {
      // These are cached and thus not assignable to individual Skyframe objects
      return false;
    }

    if (o instanceof RepositoryName) {
      // These are cached and thus not assignable to individual Skyframe objects
      return false;
    }

    if (o instanceof StarlarkSemantics) {
      return false;
    }

    return true;
  }

  @Nullable
  @Override
  public String contextForArrayItem(Object from, String fromContext, Object to) {
    return null;
  }

  @Nullable
  @Override
  public String contextForField(Object from, String fromContext, Field field, Object to) {
    return null;
  }

  @Nullable
  @Override
  public ImmutableSet<String> ignoredFields(Class<?> clazz) {
    if (clazz == StarlarkDefinedConfigTransition.class) {
      return ImmutableSet.of("ruleTransitionCache");
    }

    if (clazz == StarlarkDefinedAspect.class) {
      return ImmutableSet.of("definitionCache");
    }

    return null;
  }
}
