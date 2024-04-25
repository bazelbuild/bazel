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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Map;
import org.junit.runner.RunWith;

/** Tests for {@link GlobFunction}. */
@RunWith(TestParameterInjector.class)
public final class GlobFunctionTest extends GlobTestBase {

  @TestParameter private boolean recursionInSingleFunction;

  @Override
  protected void createGlobSkyFunction(Map<SkyFunctionName, SkyFunction> skyFunctions) {
    PrecomputedValue.CONVENIENCE_SYMLINKS_PATHS.set(new SequencedRecordingDifferencer(), ImmutableSet.of());
    skyFunctions.put(SkyFunctions.GLOB, GlobFunction.create(recursionInSingleFunction));
  }

  @Override
  protected void assertSingleGlobMatches(
      String pattern, Globber.Operation globberOperation, String... expecteds) throws Exception {
    Iterable<String> matches =
        Iterables.transform(
            runSingleGlob(pattern, globberOperation).getMatches(), Functions.toStringFunction());
    if (recursionInSingleFunction) {
      assertThat(matches).containsExactlyElementsIn(ImmutableList.copyOf(expecteds));
    } else {
      // The order requirement is not strictly necessary -- a change to GlobFunction semantics that
      // changes the output order is fine, but we require that the order be the same here to detect
      // potential non-determinism in output order, which would be bad.
      // The current order in the case of "**" or "*" is roughly that of
      // nestedset.Order.STABLE_ORDER, putting subdirectories before directories, but putting
      // ordinary files after their parent directories.
      assertThat(matches).containsExactlyElementsIn(ImmutableList.copyOf(expecteds)).inOrder();
    }
  }

  @Override
  protected GlobValue runSingleGlob(String pattern, Globber.Operation globberOperation)
      throws Exception {
    SkyKey skyKey =
        GlobValue.key(
            PKG_ID, Root.fromPath(root), pattern, globberOperation, PathFragment.EMPTY_FRAGMENT);
    EvaluationResult<SkyValue> result =
        evaluator.evaluate(ImmutableList.of(skyKey), EVALUATION_OPTIONS);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    return (GlobValue) result.get(skyKey);
  }

  @Override
  protected void assertIllegalPattern(String pattern) {
    assertThrows(
        "invalid pattern not detected: " + pattern,
        InvalidGlobPatternException.class,
        () ->
            GlobValue.key(
                PKG_ID,
                Root.fromPath(root),
                pattern,
                Globber.Operation.FILES_AND_DIRS,
                PathFragment.EMPTY_FRAGMENT));
  }

  @Override
  protected GlobDescriptor createdGlobRelatedSkyKey(
      String pattern, Globber.Operation globberOperation) throws InvalidGlobPatternException {
    return GlobValue.key(
        PKG_ID, Root.fromPath(root), pattern, globberOperation, PathFragment.EMPTY_FRAGMENT);
  }

  @Override
  protected Iterable<String> getSubpackagesMatches(String pattern) throws Exception {
    return Iterables.transform(
        runSingleGlob(pattern, Globber.Operation.SUBPACKAGES).getMatches(),
        Functions.toStringFunction());
  }
}
