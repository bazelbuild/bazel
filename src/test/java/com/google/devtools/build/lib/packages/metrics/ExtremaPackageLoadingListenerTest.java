// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.metrics;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyMap;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.metrics.ExtremaPackageLoadingListener.PackageIdentifierAndLong;
import com.google.devtools.build.lib.packages.metrics.ExtremaPackageLoadingListener.TopPackages;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExtremaPackageLoadingListener}. */
@RunWith(JUnit4.class)
public class ExtremaPackageLoadingListenerTest {

  private final ExtremaPackageLoadingListener underTest =
      ExtremaPackageLoadingListener.getInstance();

  @Test
  public void testRecordsTopSlowestPackagesPerBuild() {
    underTest.setNumPackagesToTrack(2);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 42_000_000);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 43_000_000);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 44_000_000);

    assertThat(toStrings(underTest.getAndResetTopPackages().getSlowestPackages()))
        .containsExactly("my/pkg3 (44)", "my/pkg2 (43)")
        .inOrder();

    assertAllTopPackagesEmpty(underTest.getAndResetTopPackages());
  }

  @Test
  public void testRecordsTopLargestPackagesPerBuild() {
    underTest.setNumPackagesToTrack(2);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1",
            ImmutableMap.of("target1", mock(Target.class)),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2",
            ImmutableMap.of("target1", mock(Target.class), "target2", mock(Target.class)),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3",
            ImmutableMap.of(
                "target1",
                mock(Target.class),
                "target2",
                mock(Target.class),
                "target3",
                mock(Target.class)),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100);

    assertThat(toStrings(underTest.getAndResetTopPackages().getLargestPackages()))
        .containsExactly("my/pkg3 (3)", "my/pkg2 (2)")
        .inOrder();

    assertAllTopPackagesEmpty(underTest.getAndResetTopPackages());
  }

  @Test
  public void testRecordsTransitiveLoadsPerBuild() {
    underTest.setNumPackagesToTrack(2);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1",
            /*targets=*/ ImmutableMap.of(),
            ImmutableList.of(Label.parseAbsoluteUnchecked("//load:1.bzl"))),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2",
            /*targets=*/ ImmutableMap.of(),
            ImmutableList.of(
                Label.parseAbsoluteUnchecked("//load:1.bzl"),
                Label.parseAbsoluteUnchecked("//load:2.bzl"))),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3",
            /*targets=*/ ImmutableMap.of(),
            ImmutableList.of(
                Label.parseAbsoluteUnchecked("//load:1.bzl"),
                Label.parseAbsoluteUnchecked("//load:2.bzl"),
                Label.parseAbsoluteUnchecked("//load:3.bzl"))),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100);

    assertThat(toStrings(underTest.getAndResetTopPackages().getPackagesWithMostTransitiveLoads()))
        .containsExactly("my/pkg3 (3)", "my/pkg2 (2)")
        .inOrder();

    assertAllTopPackagesEmpty(underTest.getAndResetTopPackages());
  }

  @Test
  public void testRecordsMostComputationStepsPerBuild() {
    underTest.setNumPackagesToTrack(2);

    Package mockPackage1 =
        mockPackage(
            "my/pkg1",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of());
    when(mockPackage1.getComputationSteps()).thenReturn(1000L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage1, StarlarkSemantics.DEFAULT, /*loadTimeNanos=*/ 100);

    Package mockPackage2 =
        mockPackage(
            "my/pkg2",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of());
    when(mockPackage2.getComputationSteps()).thenReturn(100L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2, StarlarkSemantics.DEFAULT, /*loadTimeNanos=*/ 100);

    Package mockPackage3 =
        mockPackage(
            "my/pkg3",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of());
    when(mockPackage3.getComputationSteps()).thenReturn(10L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3, StarlarkSemantics.DEFAULT, /*loadTimeNanos=*/ 100);

    assertThat(toStrings(underTest.getAndResetTopPackages().getPackagesWithMostComputationSteps()))
        .containsExactly("my/pkg1 (1000)", "my/pkg2 (100)")
        .inOrder();

    assertAllTopPackagesEmpty(underTest.getAndResetTopPackages());
  }

  @Test
  public void testDoesntRecordAnythingWhenNumPackagesToTrackIsZero() {
    underTest.setNumPackagesToTrack(0);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1",
            /*targets=*/ ImmutableMap.of(),
            /*starlarkDependencies=*/ ImmutableList.of()),
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 42_000_000);

    assertAllTopPackagesEmpty(underTest.getAndResetTopPackages());
  }

  private static void assertAllTopPackagesEmpty(TopPackages topPackages) {
    assertThat(topPackages.getSlowestPackages()).isEmpty();
    assertThat(topPackages.getLargestPackages()).isEmpty();
    assertThat(topPackages.getPackagesWithMostTransitiveLoads()).isEmpty();
    assertThat(topPackages.getPackagesWithMostComputationSteps()).isEmpty();
  }

  private static Package mockPackage(
      String pkgIdString, Map<String, Target> targets, List<Label> starlarkDependencies) {
    Package mockPackage = mock(Package.class);
    when(mockPackage.getPackageIdentifier())
        .thenReturn(PackageIdentifier.createInMainRepo(pkgIdString));
    when(mockPackage.getTargets()).thenReturn(ImmutableSortedKeyMap.copyOf(targets));
    when(mockPackage.getStarlarkFileDependencies())
        .thenReturn(ImmutableList.copyOf(starlarkDependencies));
    return mockPackage;
  }

  private static ImmutableList<String> toStrings(List<PackageIdentifierAndLong> from) {
    return from.stream().map(v -> v.toString()).collect(ImmutableList.toImmutableList());
  }
}
