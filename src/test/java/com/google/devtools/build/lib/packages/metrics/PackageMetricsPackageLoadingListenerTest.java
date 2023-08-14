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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
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
        /* loadTimeNanos= */ 42_000_000,
        /* packageOverhead= */ OptionalLong.empty());

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 43_000_000,
        /* packageOverhead= */ OptionalLong.empty());

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 44_000_000,
        /* packageOverhead= */ OptionalLong.empty());
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
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2",
            ImmutableMap.of("target1", mock(Target.class), "target2", mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());

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
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());
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
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 2),
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 3),
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());
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
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.empty());

    Package mockPackage2 =
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage2.getComputationSteps()).thenReturn(100L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2,
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100,
        /*packageOverhead=*/ OptionalLong.empty());

    Package mockPackage3 =
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage3.getComputationSteps()).thenReturn(10L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3,
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100,
        /*packageOverhead=*/ OptionalLong.empty());
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
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage1,
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 100,
        /* packageOverhead= */ OptionalLong.of(100));

    // Record nothing for pkg2, will be missing from metrics.
    Package mockPackage2 =
        mockPackage(
            "my/pkg2", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2,
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100,
        /*packageOverhead=*/ OptionalLong.empty());

    Package mockPackage3 =
        mockPackage(
            "my/pkg3", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0);
    when(mockPackage3.getComputationSteps()).thenReturn(10L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3,
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 100,
        /*packageOverhead=*/ OptionalLong.of(300));
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
            .setComputationSteps(1000)
            .setNumTargets(1)
            .setNumTransitiveLoads(1)
            .setPackageOverhead(100_000)
            .build();

    PackageLoadMetrics pkg2 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg2")
            .setLoadDuration(Durations.fromMillis(43))
            .setComputationSteps(100)
            .setNumTargets(2)
            .setNumTransitiveLoads(2)
            .setPackageOverhead(200_000)
            .build();

    PackageLoadMetrics pkg3 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg3")
            .setLoadDuration(Durations.fromMillis(44))
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
            .setComputationSteps(1000)
            .setNumTargets(1)
            .setNumTransitiveLoads(1)
            .setPackageOverhead(100_000)
            .build();

    PackageLoadMetrics pkg2 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg2")
            .setLoadDuration(Durations.fromMillis(43))
            .setComputationSteps(100)
            .setNumTargets(2)
            .setNumTransitiveLoads(2)
            .setPackageOverhead(200_000)
            .build();

    PackageLoadMetrics pkg3 =
        PackageLoadMetrics.newBuilder()
            .setName("my/pkg3")
            .setLoadDuration(Durations.fromMillis(44))
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
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage1,
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 42_000_000,
        /* packageOverhead= */ OptionalLong.of(100_000));

    Package mockPackage2 =
        mockPackage(
            "my/pkg2",
            /* targets= */ ImmutableMap.of(
                "target1", mock(Target.class), "target2", mock(Target.class)),
            /* transitivelyLoadedStarlarkFiles= */ 2);
    when(mockPackage2.getComputationSteps()).thenReturn(100L);
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage2,
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 43_000_000,
        /*packageOverhead=*/ OptionalLong.of(200_000));

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
    underTest.onLoadingCompleteAndSuccessful(
        mockPackage3,
        StarlarkSemantics.DEFAULT,
        /*loadTimeNanos=*/ 44_000_000,
        /*packageOverhead=*/ OptionalLong.of(300_000));
  }

  @Test
  public void testDoesntRecordAnythingWhenNumPackagesToTrackIsZero() {
    PackageMetricsRecorder recorder = new ExtremaPackageMetricsRecorder(0);
    underTest.setPackageMetricsRecorder(recorder);

    underTest.onLoadingCompleteAndSuccessful(
        mockPackage(
            "my/pkg1", /* targets= */ ImmutableMap.of(), /* transitivelyLoadedStarlarkFiles= */ 0),
        StarlarkSemantics.DEFAULT,
        /* loadTimeNanos= */ 42_000_000,
        /* packageOverhead= */ OptionalLong.empty());

    assertAllMapsEmpty(underTest.getPackageMetricsRecorder());
  }

  private static void assertAllMapsEmpty(PackageMetricsRecorder recorder) {
    assertThat(recorder.getLoadTimes()).isEmpty();
    assertThat(recorder.getComputationSteps()).isEmpty();
    assertThat(recorder.getNumTargets()).isEmpty();
    assertThat(recorder.getNumTransitiveLoads()).isEmpty();
  }

  private static Package mockPackage(
      String pkgIdString, Map<String, Target> targets, int transitivelyLoadedStarlarkFiles) {
    Package mockPackage = mock(Package.class);
    when(mockPackage.getPackageIdentifier())
        .thenReturn(PackageIdentifier.createInMainRepo(pkgIdString));
    when(mockPackage.getTargets()).thenReturn(ImmutableSortedMap.copyOf(targets));
    when(mockPackage.countTransitivelyLoadedStarlarkFiles())
        .thenReturn(transitivelyLoadedStarlarkFiles);
    return mockPackage;
  }
}
