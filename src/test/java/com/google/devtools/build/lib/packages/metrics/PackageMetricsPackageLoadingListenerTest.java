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
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageLoadingListener.Metrics;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageOptions.LazyMacroExpansionPackages;
import com.google.protobuf.util.Durations;
import java.util.Map;
import java.util.OptionalLong;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PackageMetricsPackageLoadingListener}. */
@RunWith(JUnit4.class)
public class PackageMetricsPackageLoadingListenerTest {

  private final PackageMetricsPackageLoadingListener underTest =
      PackageMetricsPackageLoadingListener.getInstance();

  private static final Metrics PLACEHOLDER_METRICS =
      new Metrics(/* loadTimeNanos= */ 123, /* globFilesystemOperationCost= */ 456);

  @Test
  public void testRecordsTopSlowestPackagesPerBuild_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordSlowPackages();

    assertThat(underTest.getPackageMetricsRecorder().getLoadTimes())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                PackageIdentifier.createInMainRepo("my/pkg3"),
                Durations.fromMillis(44),
                PackageIdentifier.createInMainRepo("my/pkg2"),
                Durations.fromMillis(43)))
        .inOrder();

    recorder.loadingFinished();
    assertAllMapsEmpty(recorder);
  }

  @Test
  public void testRecordsTopSlowestPackagesPerBuild_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordSlowPackages();

    assertThat(underTest.getPackageMetricsRecorder().getLoadTimes())
        .containsExactly(
            PackageIdentifier.createInMainRepo("my/pkg1"),
            Durations.fromMillis(42),
            PackageIdentifier.createInMainRepo("my/pkg2"),
            Durations.fromMillis(43),
            PackageIdentifier.createInMainRepo("my/pkg3"),
            Durations.fromMillis(44));
    recorder.clear();
    assertAllMapsEmpty(recorder);
  }

  private void recordSlowPackages() {
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 42_000_000, /* globFilesystemOperationCost= */ 0));

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 43_000_000, /* globFilesystemOperationCost= */ 0));

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 44_000_000, /* globFilesystemOperationCost= */ 0));
  }

  @Test
  public void testRecordsTopLargestPackagesPerBuild_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordLargePackages();

    assertThat(underTest.getPackageMetricsRecorder().getNumTargets())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                PackageIdentifier.createInMainRepo("my/pkg3"),
                3L,
                PackageIdentifier.createInMainRepo("my/pkg2"),
                2L))
        .inOrder();
    recorder.loadingFinished();

    assertAllMapsEmpty(recorder);
  }

  @Test
  public void testRecordsTopLargestPackagesPerBuild_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordLargePackages();

    assertThat(underTest.getPackageMetricsRecorder().getNumTargets())
        .containsExactly(
            PackageIdentifier.createInMainRepo("my/pkg3"),
            3L,
            PackageIdentifier.createInMainRepo("my/pkg2"),
            2L,
            PackageIdentifier.createInMainRepo("my/pkg1"),
            1L);
  }

  private void recordLargePackages() {
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1",
            ImmutableMap.of("target1", mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2",
            ImmutableMap.of("target1", mock(Target.class), "target2", mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

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
            /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);
  }

  @Test
  public void testRecordsTransitiveLoadsPerBuild_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordTransitiveLoads();

    assertThat(underTest.getPackageMetricsRecorder().getNumTransitiveLoads())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                PackageIdentifier.createInMainRepo("my/pkg3"),
                3L,
                PackageIdentifier.createInMainRepo("my/pkg2"),
                2L))
        .inOrder();
    recorder.loadingFinished();
    assertAllMapsEmpty(recorder);
  }

  @Test
  public void testRecordsTransitiveLoadsPerBuild_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordTransitiveLoads();

    assertThat(underTest.getPackageMetricsRecorder().getNumTransitiveLoads())
        .containsExactly(
            PackageIdentifier.createInMainRepo("my/pkg3"),
            3L,
            PackageIdentifier.createInMainRepo("my/pkg2"),
            2L,
            PackageIdentifier.createInMainRepo("my/pkg1"),
            1L);
  }

  private void recordTransitiveLoads() {
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 1),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 2),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 3),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);
  }

  @Test
  public void testRecordsMostComputationStepsPerBuild_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordComputationSteps();

    assertThat(underTest.getPackageMetricsRecorder().getComputationSteps())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                PackageIdentifier.createInMainRepo("my/pkg1"),
                1000L,
                PackageIdentifier.createInMainRepo("my/pkg2"),
                100L))
        .inOrder();
    recorder.loadingFinished();

    assertAllMapsEmpty(recorder);
  }

  @Test
  public void testRecordsMostComputationStepsPerBuild_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordComputationSteps();

    assertThat(underTest.getPackageMetricsRecorder().getComputationSteps())
        .containsExactly(
            PackageIdentifier.createInMainRepo("my/pkg1"),
            1000L,
            PackageIdentifier.createInMainRepo("my/pkg2"),
            100L,
            PackageIdentifier.createInMainRepo("my/pkg3"),
            10L);
    recorder.loadingFinished();

    assertAllMapsEmpty(recorder);
  }

  private void recordComputationSteps() {
    Package mockPackage1 =
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage1.getComputationSteps()).thenReturn(1000L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage1,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    Package mockPackage2 =
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage2.getComputationSteps()).thenReturn(100L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    Package mockPackage3 =
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage3.getComputationSteps()).thenReturn(10L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);
  }

  @Test
  public void testRecordsMostPackageOverheadPerBuild_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordPackageOverhead();

    assertThat(underTest.getPackageMetricsRecorder().getPackageOverhead())
        .containsExactly(
            PackageIdentifier.createInMainRepo("my/pkg1"),
            100L,
            PackageIdentifier.createInMainRepo("my/pkg3"),
            300L);
    recorder.loadingFinished();

    assertAllMapsEmpty(recorder);
  }

  @Test
  public void testRecordsTopPackageOverheadPackagesPerBuild_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordPackageOverhead();

    assertThat(underTest.getPackageMetricsRecorder().getPackageOverhead())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                PackageIdentifier.createInMainRepo("my/pkg3"),
                300L,
                PackageIdentifier.createInMainRepo("my/pkg1"),
                100L))
        .inOrder();
    recorder.loadingFinished();

    assertAllMapsEmpty(recorder);
  }

  private void recordPackageOverhead() {
    Package mockPackage1 =
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage1.getPackageOverhead()).thenReturn(OptionalLong.of(100));
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage1,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    // Record nothing for pkg2, will be missing from metrics.
    Package mockPackage2 =
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);

    Package mockPackage3 =
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage3.getComputationSteps()).thenReturn(10L);
    when(mockPackage3.getPackageOverhead()).thenReturn(OptionalLong.of(300));

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        PLACEHOLDER_METRICS);
  }

  @Test
  public void metricMap_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordEverything();

    PackageLoadMetrics pkg1 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg1")
            .setLoadDuration(Durations.fromMillis(42))
            .setGlobFilesystemOperationCost(100)
            .setComputationSteps(1000)
            .setNumTargets(1)
            .setNumTransitiveLoads(1)
            .setPackageOverhead(100_000)
            .build();

    PackageLoadMetrics pkg2 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg2")
            .setLoadDuration(Durations.fromMillis(43))
            .setGlobFilesystemOperationCost(200)
            .setComputationSteps(100)
            .setNumTargets(2)
            .setNumTransitiveLoads(2)
            .setPackageOverhead(200_000)
            .build();

    PackageLoadMetrics pkg3 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg3")
            .setLoadDuration(Durations.fromMillis(44))
            .setGlobFilesystemOperationCost(300)
            .setComputationSteps(10)
            .setNumTargets(3)
            .setNumTransitiveLoads(3)
            .setPackageOverhead(300_000)
            .build();

    assertThat(underTest.getPackageMetricsRecorder().getPackageLoadMetrics())
        .containsExactly(pkg1, pkg2, pkg3);
    recorder.loadingFinished();
    assertAllMapsEmpty(recorder);
  }

  @Test
  public void metricMap_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordEverything();

    PackageLoadMetrics pkg1 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg1")
            .setLoadDuration(Durations.fromMillis(42))
            .setGlobFilesystemOperationCost(100)
            .setComputationSteps(1000)
            .setNumTargets(1)
            .setNumTransitiveLoads(1)
            .setPackageOverhead(100_000)
            .build();

    PackageLoadMetrics pkg2 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg2")
            .setLoadDuration(Durations.fromMillis(43))
            .setGlobFilesystemOperationCost(200)
            .setComputationSteps(100)
            .setNumTargets(2)
            .setNumTransitiveLoads(2)
            .setPackageOverhead(200_000)
            .build();

    PackageLoadMetrics pkg3 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg3")
            .setLoadDuration(Durations.fromMillis(44))
            .setGlobFilesystemOperationCost(300)
            .setComputationSteps(10)
            .setNumTargets(3)
            .setNumTransitiveLoads(3)
            .setPackageOverhead(300_000)
            .build();

    assertThat(underTest.getPackageMetricsRecorder().getPackageLoadMetrics())
        .containsExactly(pkg1, pkg2, pkg3);
    recorder.loadingFinished();
    assertAllMapsEmpty(recorder);
  }

  private void recordEverything() {
    Package mockPackage1 =
        mockPackage(
            "my/pkg1",
            /* targets= */ ImmutableMap.of("target1", mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 1);
    when(mockPackage1.getComputationSteps()).thenReturn(1000L);
    when(mockPackage1.getPackageOverhead()).thenReturn(OptionalLong.of(100_000));

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage1,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 42_000_000, /* globFilesystemOperationCost= */ 100));

    Package mockPackage2 =
        mockPackage(
            "my/pkg2",
            /* targets= */ ImmutableMap.of(
                "target1", mock(Target.class), "target2", mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 2);
    when(mockPackage2.getComputationSteps()).thenReturn(100L);
    when(mockPackage2.getPackageOverhead()).thenReturn(OptionalLong.of(200_000));
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 43_000_000, /* globFilesystemOperationCost= */ 200));

    Package mockPackage3 =
        mockPackage(
            "my/pkg3",
            /* targets= */ ImmutableMap.of(
                "target1",
                mock(Target.class),
                "target2",
                mock(Target.class),
                "target3",
                mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 3);
    when(mockPackage3.getComputationSteps()).thenReturn(10L);
    when(mockPackage3.getPackageOverhead()).thenReturn(OptionalLong.of(300_000));
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3,
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 44_000_000, /* globFilesystemOperationCost= */ 300));
  }

  private void recordPackagesWithGlobCost() {
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 0, /* globFilesystemOperationCost= */ 111));

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 0, /* globFilesystemOperationCost= */ 222));

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 0, /* globFilesystemOperationCost= */ 333));
  }

  @Test
  public void testRecordsTopGlobFilesystemOperationCost_extrema() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(2);
    underTest.setPackageMetricsRecorder(recorder);

    recordPackagesWithGlobCost();

    assertThat(underTest.getPackageMetricsRecorder().getGlobFilesystemOperationCost())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                PackageIdentifier.createInMainRepo("my/pkg3"),
                333L,
                PackageIdentifier.createInMainRepo("my/pkg2"),
                222L))
        .inOrder();
    recorder.loadingFinished();

    assertAllMapsEmpty(recorder);
  }

  @Test
  public void testRecordsTopGlobFilesystemOperationCost_complete() {
    PackageMetricsRecorder recorder = new CompletePackageMetricsRecorder();
    underTest.setPackageMetricsRecorder(recorder);

    recordPackagesWithGlobCost();

    assertThat(underTest.getPackageMetricsRecorder().getGlobFilesystemOperationCost())
        .containsExactly(
            PackageIdentifier.createInMainRepo("my/pkg3"),
            333L,
            PackageIdentifier.createInMainRepo("my/pkg2"),
            222L,
            PackageIdentifier.createInMainRepo("my/pkg1"),
            111L);
  }

  @Test
  public void testDoesntRecordAnythingWhenNumPackagesToTrackIsZero() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(0);
    underTest.setPackageMetricsRecorder(recorder);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        LazyMacroExpansionPackages.NONE,
        new Metrics(/* loadTimeNanos= */ 42_000_000, /* globFilesystemOperationCost= */ 0));

    assertAllMapsEmpty(underTest.getPackageMetricsRecorder());
  }

  private static void assertAllMapsEmpty(PackageMetricsRecorder recorder) {
    assertThat(recorder.getLoadTimes()).isEmpty();
    assertThat(recorder.getGlobFilesystemOperationCost()).isEmpty();
    assertThat(recorder.getComputationSteps()).isEmpty();
    assertThat(recorder.getNumTargets()).isEmpty();
    assertThat(recorder.getNumTransitiveLoads()).isEmpty();
  }

  private static Package mockPackage(
      String pkgIdString, Map<String, Target> targets, int transitivelyLoadedStarlarkFiles) {
    ImmutableList.Builder<Label> fakeLoads = ImmutableList.builder();
    for (int i = 0; i < transitivelyLoadedStarlarkFiles; i++) {
      fakeLoads.add(Label.parseCanonicalUnchecked(String.format("//:%d.bzl", i)));
    }
    Package.Declarations fakeDeclarations =
        new Package.Declarations.Builder().setTransitiveLoads(fakeLoads.build()).build();
    Package mockPackage = mock(Package.class);
    when(mockPackage.getPackageIdentifier())
        .thenReturn(PackageIdentifier.createInMainRepo(pkgIdString));
    when(mockPackage.getTargets()).thenReturn(ImmutableSortedMap.copyOf(targets));
    when(mockPackage.getDeclarations()).thenReturn(fakeDeclarations);
    return mockPackage;
  }
}
