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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil.TargetAndErrorIfAny;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.GroupedDeps;
import com.google.devtools.build.skyframe.SimpleSkyframeLookupResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.build.skyframe.ValueOrUntypedException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Test for {@link TransitiveTraversalFunction}. */
@RunWith(JUnit4.class)
public class TransitiveTraversalFunctionTest extends BuildViewTestCase {

  @Test
  public void noRepeatedLabelVisitationForTransitiveTraversalFunction() throws Exception {
    // Create a basic package with a target //foo:foo.
    Label label = Label.parseCanonical("//foo:foo");
    scratch.file("foo/BUILD", "sh_library(name = '" + label.getName() + "')");
    Package pkg = loadPackage(label.getPackageIdentifier());
    TargetAndErrorIfAny targetAndErrorIfAny =
        new TargetAndErrorIfAny(
            /* packageLoadedSuccessfully= */ true,
            /* errorLoadingTarget= */ null,
            pkg.getTarget(label.getName()));
    TransitiveTraversalFunction function =
        new TransitiveTraversalFunction() {
          @Override
          TargetAndErrorIfAny loadTarget(Environment env, Label label) {
            return targetAndErrorIfAny;
          }
        };
    // Create the GroupedDeps saying we had already requested two targets the last time we called
    // #compute.
    GroupedDeps groupedDeps = new GroupedDeps();
    groupedDeps.appendSingleton(label.getPackageIdentifier());
    // Note that these targets don't actually exist in the package we created initially. It doesn't
    // matter for the purpose of this test, the original package was just to create some objects
    // that we needed.
    SkyKey fakeDep1 = function.getKey(Label.parseCanonical("//foo:bar"));
    SkyKey fakeDep2 = function.getKey(Label.parseCanonical("//foo:baz"));
    groupedDeps.appendGroup(ImmutableList.of(fakeDep1, fakeDep2));

    AtomicBoolean wasOptimizationUsed = new AtomicBoolean(false);
    SkyFunction.Environment mockEnv = Mockito.mock(SkyFunction.Environment.class);
    when(mockEnv.getTemporaryDirectDeps()).thenReturn(groupedDeps);
    when(mockEnv.getValuesAndExceptions(groupedDeps.getDepGroup(1)))
        .thenAnswer(
            (invocationOnMock) -> {
              wasOptimizationUsed.set(true);
              // It doesn't matter what this SkyframeLookupResult is, we'll return true in the
              // valuesMissing() call.
              return new SimpleSkyframeLookupResult(
                  /* valuesMissingCallback= */ () -> {},
                  k -> {
                    throw new IllegalStateException("Shouldn't have been called: " + k);
                  });
            });
    when(mockEnv.valuesMissing()).thenReturn(true);

    // Run the compute function and check that we returned null.
    assertThat(function.compute(function.getKey(label), mockEnv)).isNull();

    // Verify that the mock was called with the arguments we expected.
    assertThat(wasOptimizationUsed.get()).isTrue();
  }

  @Test
  public void multipleErrorsForTransitiveTraversalFunction() throws Exception {
    Label label = Label.parseCanonical("//foo:foo");
    scratch.file(
        "foo/BUILD", "sh_library(name = '" + label.getName() + "', deps = [':bar', ':baz'])");
    Package pkg = loadPackage(label.getPackageIdentifier());
    TargetAndErrorIfAny targetAndErrorIfAny =
        new TargetAndErrorIfAny(
            /* packageLoadedSuccessfully= */ true,
            /* errorLoadingTarget= */ null,
            pkg.getTarget(label.getName()));
    TransitiveTraversalFunction function =
        new TransitiveTraversalFunction() {
          @Override
          TargetAndErrorIfAny loadTarget(Environment env, Label label) {
            return targetAndErrorIfAny;
          }
        };
    SkyKey dep1 = function.getKey(Label.parseCanonical("//foo:bar"));
    SkyKey dep2 = function.getKey(Label.parseCanonical("//foo:baz"));
    SkyFunction.Environment mockEnv = Mockito.mock(SkyFunction.Environment.class);
    NoSuchTargetException exp1 = new NoSuchTargetException("bad bar");
    NoSuchTargetException exp2 = new NoSuchTargetException("bad baz");
    SkyframeLookupResult returnedDeps =
        new SimpleSkyframeLookupResult(
            () -> {},
            key ->
                key.equals(dep1)
                    ? ValueOrUntypedException.ofExn(exp1)
                    : key.equals(dep2) ? ValueOrUntypedException.ofExn(exp2) : null);

    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(dep1, dep2))).thenReturn(returnedDeps);
    when(mockEnv.valuesMissing()).thenReturn(false);

    assertThat(
            ((TransitiveTraversalValue) function.compute(function.getKey(label), mockEnv))
                .getErrorMessage())
        .isEqualTo("bad bar");
  }

  @Test
  public void selfErrorWins() throws Exception {
    Label label = Label.parseCanonical("//foo:foo");
    scratch.file("foo/BUILD", "sh_library(name = '" + label.getName() + "', deps = [':bar'])");
    Package pkg = loadPackage(label.getPackageIdentifier());
    TargetAndErrorIfAny targetAndErrorIfAny =
        new TargetAndErrorIfAny(
            /* packageLoadedSuccessfully= */ true,
            /* errorLoadingTarget= */ new NoSuchTargetException("self error is long and last"),
            pkg.getTarget(label.getName()));
    TransitiveTraversalFunction function =
        new TransitiveTraversalFunction() {
          @Override
          TargetAndErrorIfAny loadTarget(Environment env, Label label) {
            return targetAndErrorIfAny;
          }
        };
    SkyKey dep = function.getKey(Label.parseCanonical("//foo:bar"));
    NoSuchTargetException exp = new NoSuchTargetException("bad bar");
    SkyframeLookupResult returnedDep =
        new SimpleSkyframeLookupResult(
            () -> {}, key -> key.equals(dep) ? ValueOrUntypedException.ofExn(exp) : null);
    SkyFunction.Environment mockEnv = Mockito.mock(SkyFunction.Environment.class);
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(dep))).thenReturn(returnedDep);
    when(mockEnv.valuesMissing()).thenReturn(false);

    TransitiveTraversalValue transitiveTraversalValue =
        (TransitiveTraversalValue) function.compute(function.getKey(label), mockEnv);
    assertThat(transitiveTraversalValue.getErrorMessage()).isEqualTo("self error is long and last");
  }

  @Test
  public void getStrictLabelAspectKeys() throws Exception {
    Label label = Label.parseCanonical("//test:foo");
    scratch.file(
        "test/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            return []

        def _rule_impl(ctx):
            return []

        MyAspect = aspect(
            implementation = _aspect_impl,
            attr_aspects = ["deps"],
            attrs = {"_extra_deps": attr.label(default = Label("//foo:bar"))},
        )
        my_rule = rule(
            implementation = _rule_impl,
            attrs = {
                "attr": attr.label_list(mandatory = True, aspects = [MyAspect]),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:aspect.bzl", "my_rule")

        my_rule(
            name = "foo",
            attr = [":bad"],
        )
        """);
    Package pkg = loadPackage(label.getPackageIdentifier());
    TargetAndErrorIfAny targetAndErrorIfAny =
        new TargetAndErrorIfAny(
            /* packageLoadedSuccessfully= */ true,
            /* errorLoadingTarget= */ null,
            pkg.getTarget(label.getName()));
    TransitiveTraversalFunction function =
        new TransitiveTraversalFunction() {
          @Override
          TargetAndErrorIfAny loadTarget(Environment env, Label label) {
            return targetAndErrorIfAny;
          }
        };
    SkyKey badDep = function.getKey(Label.parseCanonical("//test:bad"));
    NoSuchTargetException exp = new NoSuchTargetException("bad test");
    AtomicBoolean valuesMissing = new AtomicBoolean(false);
    SkyframeLookupResult returnedDep =
        new SimpleSkyframeLookupResult(
            () -> valuesMissing.set(true),
            key -> key.equals(badDep) ? ValueOrUntypedException.ofExn(exp) : null);
    SkyFunction.Environment mockEnv = Mockito.mock(SkyFunction.Environment.class);
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(badDep))).thenReturn(returnedDep);

    TransitiveTraversalValue transitiveTraversalValue =
        (TransitiveTraversalValue) function.compute(function.getKey(label), mockEnv);
    assertThat(transitiveTraversalValue.getErrorMessage()).isEqualTo("bad test");
    assertThat(valuesMissing.get()).isFalse();
  }

  /* Invokes the loading phase, using Skyframe. */
  private Package loadPackage(PackageIdentifier pkgid)
      throws InterruptedException, NoSuchPackageException {
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), pkgid, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      throw (NoSuchPackageException) result.getError(pkgid).getException();
    }
    return result.get(pkgid).getPackage();
  }
}
