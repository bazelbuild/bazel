// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Substitution.ComputedSubstitution;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.TemplateDictApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** Implementation of the {@code TemplateDict} Starlark type */
public class TemplateDict implements TemplateDictApi {

  private final List<Substitution> substitutions = Lists.newArrayList();

  private TemplateDict() {}

  public static TemplateDictApi newDict() {
    return new TemplateDict();
  }

  @CanIgnoreReturnValue
  @Override
  public TemplateDictApi addArgument(String key, String value, StarlarkThread thread)
      throws EvalException {
    substitutions.add(Substitution.of(key, value));
    return this;
  }

  @CanIgnoreReturnValue
  @Override
  public TemplateDictApi addJoined(
      String key,
      Depset valuesSet,
      String joinWith,
      StarlarkCallable mapEach,
      StarlarkThread thread)
      throws EvalException {
    if (mapEach instanceof StarlarkFunction) {
      StarlarkFunction sfn = (StarlarkFunction) mapEach;
      if (sfn.getModule().getGlobal(sfn.getName()) != sfn) {
        throw Starlark.errorf(
            "to avoid unintended retention of analysis data structures, "
                + "the map_each function (declared at %s) must be declared "
                + "by a top-level def statement",
            sfn.getLocation());
      }
    }
    substitutions.add(
        new LazySubstitution(key, thread.getSemantics(), valuesSet, mapEach, joinWith));
    return this;
  }

  public Iterable<? extends Substitution> getAll() {
    return substitutions;
  }

  private static class LazySubstitution extends ComputedSubstitution {

    private final StarlarkSemantics semantics;
    private final Depset valuesSet;
    private final StarlarkCallable mapEach;
    private final String joinWith;

    public LazySubstitution(
        String key,
        StarlarkSemantics semantics,
        Depset valuesSet,
        StarlarkCallable mapEach,
        String joinWith) {
      super(key);
      this.semantics = semantics;
      this.valuesSet = valuesSet;
      this.mapEach = mapEach;
      this.joinWith = joinWith;
    }

    @Override
    public String getValue() throws EvalException {
      StarlarkThread execThread =
          new StarlarkThread(Mutability.create("expand_template"), semantics);
      ImmutableList<?> values = valuesSet.toList();
      List<String> parts = new ArrayList<>(values.size());
      for (Object val : values) {
        try {
          Object ret =
              Starlark.call(
                  execThread,
                  mapEach,
                  /*args=*/ ImmutableList.of(val),
                  /*kwargs=*/ ImmutableMap.of());
          if (ret instanceof String) {
            parts.add((String) ret);
            continue;
          }
          throw Starlark.errorf(
              "Function provided to map_each must return a String, but returned type %s for key:"
                  + " %s",
              Starlark.type(ret), getKey());
        } catch (InterruptedException e) {
          // Report the error to the user, but the stack trace is not of use to them
          throw Starlark.errorf("Could not evaluate substitution for %s: %s", val, e.getMessage());
        }
      }
      return Joiner.on(joinWith).join(parts);
    }
  }
}
