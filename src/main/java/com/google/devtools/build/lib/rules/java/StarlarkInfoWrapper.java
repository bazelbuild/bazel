// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

/** Base class for sharing utility code between wrapped Starlark provider instances */
abstract class StarlarkInfoWrapper {

  protected final StructImpl underlying;

  protected StarlarkInfoWrapper(StructImpl underlying) {
    this.underlying = underlying;
  }

  protected <T> T getUnderlyingValue(String key, Class<T> type) throws RuleErrorException {
    try {
      if (underlying.getValue(key) == Starlark.NONE) {
        return null;
      } else {
        return underlying.getValue(key, type);
      }
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  protected <T> NestedSet<T> getUnderlyingNestedSet(String key, Class<T> type)
      throws RuleErrorException {
    try {
      return Depset.noneableCast(noneIfNull(underlying.getValue(key)), type, key);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  protected <T> Sequence<T> getUnderlyingSequence(String key, Class<T> type)
      throws RuleErrorException {
    try {
      return Sequence.noneableCast(noneIfNull(underlying.getValue(key)), type, key);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  private static Object noneIfNull(Object value) {
    return value == null ? Starlark.NONE : value;
  }
}
