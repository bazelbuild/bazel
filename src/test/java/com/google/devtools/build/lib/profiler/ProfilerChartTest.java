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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.chart.AggregatingChartCreator;
import com.google.devtools.build.lib.profiler.chart.Chart;
import com.google.devtools.build.lib.profiler.chart.ChartBar;
import com.google.devtools.build.lib.profiler.chart.ChartBarType;
import com.google.devtools.build.lib.profiler.chart.ChartColumn;
import com.google.devtools.build.lib.profiler.chart.ChartCreator;
import com.google.devtools.build.lib.profiler.chart.ChartLine;
import com.google.devtools.build.lib.profiler.chart.ChartRow;
import com.google.devtools.build.lib.profiler.chart.ChartVisitor;
import com.google.devtools.build.lib.profiler.chart.Color;
import com.google.devtools.build.lib.profiler.chart.DetailedChartCreator;
import com.google.devtools.build.lib.profiler.chart.HtmlChartVisitor;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.Path;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the profiler chart generation. */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class ProfilerChartTest extends FoundationTestCase {
  private static final int COMMON_CHART_TYPES = ProfilePhase.values().length;
  private static final int DETAILED_CHART_TYPES = ProfilerTask.values().length;
  private static final int AGGREGATED_CHART_TYPES = 4;
  private static final int AGGREGATED_CHART_NO_VFS_TYPES = 3;

  @Test
  public void testChartCreators() throws Exception {
    Runnable run = new Runnable() {
      @Override
      public void run() {
        Profiler.instance().profile(ProfilerTask.ACTION, "action").close();
      }
    };
    int threads = 4; // there is one extra thread due due the event that finalizes the profiler
    ProfileInfo info = createProfileInfo(run, threads - 1);
    ChartCreator aggregatingCreator = new AggregatingChartCreator(info, true);
    Chart aggregatedChart = aggregatingCreator.create();
    assertThat(aggregatedChart.getRowCount()).isEqualTo(threads);
    assertThat(aggregatedChart.getSortedRows().get(0).getBars()).hasSize(1);

    ChartCreator detailedCreator = new DetailedChartCreator(info);
    Chart detailedChart = detailedCreator.create();
    assertThat(detailedChart.getSortedTypes()).hasSize(COMMON_CHART_TYPES + DETAILED_CHART_TYPES);
    assertThat(detailedChart.getRowCount()).isEqualTo(threads);
    assertThat(detailedChart.getSortedRows().get(0).getBars()).hasSize(1);
  }

  @Test
  public void testAggregatingChartCreator() throws Exception {
    Runnable run = new Runnable() {
      @Override
      public void run() {
        Profiler profiler = Profiler.instance();
        try (SilentCloseable c = profiler.profile(ProfilerTask.ACTION, "action")) { // Stays
          task(profiler, ProfilerTask.REMOTE_EXECUTION, "remote execution"); // Removed
          task(profiler, ProfilerTask.ACTION_CHECK, "check"); // Removed
          task(profiler, ProfilerTask.ACTION_LOCK, "lock"); // Stays
        }
        task(profiler, ProfilerTask.INFO, "info"); // Stays
        task(profiler, ProfilerTask.VFS_STAT, "stat"); // Stays, if showVFS
        task(profiler, ProfilerTask.WAIT, "wait"); // Stays
      }
    };
    ProfileInfo info = createProfileInfo(run, 1);

    ChartCreator aggregatingCreator = new AggregatingChartCreator(info, true);
    Chart aggregatedChart = aggregatingCreator.create();
    assertThat(aggregatedChart.getSortedTypes())
        .hasSize(COMMON_CHART_TYPES + AGGREGATED_CHART_TYPES);
    assertThat(aggregatedChart.getSortedRows().get(0).getBars()).hasSize(5);

    ChartCreator aggregatingNoVfsCreator = new AggregatingChartCreator(info, false);
    Chart aggregatedNoVfsChart = aggregatingNoVfsCreator.create();
    assertThat(aggregatedNoVfsChart.getSortedTypes())
        .hasSize(COMMON_CHART_TYPES + AGGREGATED_CHART_NO_VFS_TYPES);
    assertThat(aggregatedNoVfsChart.getSortedRows().get(0).getBars()).hasSize(4);

    ChartCreator detailedCreator = new DetailedChartCreator(info);
    Chart detailedChart = detailedCreator.create();
    assertThat(detailedChart.getSortedTypes())
        .hasSize(COMMON_CHART_TYPES + ProfilerTask.values().length);
    assertThat(detailedChart.getSortedRows().get(0).getBars()).hasSize(7);
  }

  @Test
  public void testChart() throws Exception {
    Chart chart = new Chart();

    ChartBarType type3 = chart.createType("name3", Color.GREEN);
    ChartBarType type2 = chart.createType("name2", Color.RED);
    ChartBarType type1 = chart.createType("name1", Color.BLACK);
    List<ChartBarType> types = chart.getSortedTypes();
    assertThat(types).hasSize(3);
    assertThat(types.get(0).getName()).isEqualTo(type1.getName());
    assertThat(types.get(0).getColor()).isEqualTo(type1.getColor());
    assertThat(types.get(1).getName()).isEqualTo(type2.getName());
    assertThat(types.get(1).getColor()).isEqualTo(type2.getColor());
    assertThat(types.get(2).getName()).isEqualTo(type3.getName());
    assertThat(types.get(2).getColor()).isEqualTo(type3.getColor());

    assertThat(chart.lookUpType("name3")).isSameAs(type3);
    assertThat(chart.lookUpType("name2")).isSameAs(type2);
    assertThat(chart.lookUpType("name1")).isSameAs(type1);

    assertThat(chart.lookUpType("wergl")).isSameAs(Chart.UNKNOWN_TYPE);
    types = chart.getSortedTypes();
    assertThat(types).hasSize(4);

    chart.addBar(1, 2, 3, type1, "label1");
    chart.addBar(2, 3, 4, type2, "label2");
    chart.addBar(2, 4, 5, type3, "label3");
    chart.addBar(3, 3, 4, type2, "label4");
    chart.addBar(3, 4, 5, type3, "label5");
    chart.addBar(3, 5, 6, type3, "label6");

    assertThat(chart.getMaxStop()).isEqualTo(6);
    assertThat(chart.getRowCount()).isEqualTo(3);

    List<ChartRow> rows = chart.getSortedRows();
    assertThat(rows).hasSize(3);
    assertThat(rows.get(0).getBars()).hasSize(1);
    assertThat(rows.get(1).getBars()).hasSize(2);
    assertThat(rows.get(2).getBars()).hasSize(3);

    ChartBar bar = rows.get(0).getBars().get(0);
    assertThat(bar.getStart()).isEqualTo(2);
    assertThat(bar.getStop()).isEqualTo(3);
    assertThat(bar.getType()).isSameAs(type1);
    assertThat(bar.getLabel()).isEqualTo("label1");
  }

  @Test
  public void testChartRows() throws Exception {
    ChartRow row1 = new ChartRow("1", 0);
    ChartRow row2 = new ChartRow("2", 1);
    ChartRow row3 = new ChartRow("3", 1);

    assertThat(row1.getId()).isEqualTo("1");
    assertThat(row1.getIndex()).isEqualTo(0);

    assertThat(row1.compareTo(row2)).isEqualTo(-1);
    assertThat(row2.compareTo(row1)).isEqualTo(1);
    assertThat(row2.compareTo(row3)).isEqualTo(0);

    row1.addBar(new ChartBar(row1, 1, 2, new ChartBarType("name1", Color.BLACK), false, "label1"));
    row1.addBar(new ChartBar(row1, 2, 3, new ChartBarType("name2", Color.RED), false, "label2"));

    assertThat(row1.getBars()).hasSize(2);
    assertThat(row1.getBars().get(0).getLabel()).isEqualTo("label1");
    assertThat(row1.getBars().get(1).getLabel()).isEqualTo("label2");
  }

  @Test
  public void testChartBarTypes() throws Exception {
    ChartBarType type1 = new ChartBarType("name1", Color.BLACK);
    ChartBarType type2 = new ChartBarType("name2", Color.RED);
    ChartBarType type3 = new ChartBarType("name2", Color.GREEN);

    assertThat(type1.compareTo(type2)).isEqualTo(-1);
    assertThat(type2.compareTo(type1)).isEqualTo(1);
    assertThat(type2.compareTo(type3)).isEqualTo(0);

    assertThat(type2).isEqualTo(type3);
    assertThat(type1.equals(type3)).isFalse();
    assertThat(type1.equals(type2)).isFalse();

    assertThat(type2.hashCode()).isEqualTo(type3.hashCode());
    assertThat(type1.hashCode() == type2.hashCode()).isFalse();
    assertThat(type1.hashCode() == type3.hashCode()).isFalse();
  }

  @Test
  public void testChartBar() throws Exception {
    ChartRow row1 = new ChartRow("1", 0);
    ChartBarType type = new ChartBarType("name1", Color.BLACK);
    ChartBar bar1 = new ChartBar(row1, 1, 2, type, false, "label1");
    assertThat(bar1.getRow()).isEqualTo(row1);
    assertThat(bar1.getStart()).isEqualTo(1);
    assertThat(bar1.getStop()).isEqualTo(2);
    assertThat(bar1.getType()).isSameAs(type);
    assertThat(bar1.getLabel()).isEqualTo("label1");
  }

  @Test
  public void testVisitor() throws Exception {
    Chart chart = new Chart();
    ChartBarType type3 = chart.createType("name3", Color.GREEN);
    ChartBarType type2 = chart.createType("name2", Color.RED);
    ChartBarType type1 = chart.createType("name1", Color.BLACK);
    chart.addBar(1, 2, 3, type1, "label1");
    chart.addBar(2, 3, 4, type2, "label2");
    chart.addBar(2, 4, 5, type3, "label3");
    chart.addBar(3, 3, 4, type2, "label4");
    chart.addBar(3, 4, 5, type3, "label5");
    chart.addBar(3, 5, 6, type3, "label6");

    TestingChartVisitor visitor = new TestingChartVisitor();
    chart.accept(visitor);
    assertThat(visitor.beginChartCount).isEqualTo(1);
    assertThat(visitor.endChartCount).isEqualTo(1);
    assertThat(visitor.rowCount).isEqualTo(3);
    assertThat(visitor.barCount).isEqualTo(6);
    assertThat(visitor.columnCount).isEqualTo(0);
    assertThat(visitor.lineCount).isEqualTo(0);
  }

  @Test
  public void testHtmlChartVisitorFormatColor() {
    Locale defaultLocale = Locale.getDefault();

    Locale.setDefault(Locale.GERMANY);
    String black = HtmlChartVisitor.formatColor(Color.GRAY);
    String[] grayComponents = black.split(",");
    assertThat(grayComponents.length).isEqualTo(4);

    Locale.setDefault(defaultLocale);
  }

  private ProfileInfo createProfileInfo(Runnable runnable, int noOfRows) throws Exception {
    Scratch scratch = new Scratch();
    Path cacheDir = scratch.dir("/tmp");
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    Profiler profiler = Profiler.instance();
    try (OutputStream out = cacheFile.getOutputStream()) {
      profiler.start(
          ImmutableSet.copyOf(ProfilerTask.values()),
          out,
          Profiler.Format.BINARY_BAZEL_FORMAT,
          "basic test",
          "dummy_output_base",
          UUID.randomUUID(),
          false,
          BlazeClock.instance(),
          BlazeClock.instance().nanoTime(),
          /* enabledCpuUsageProfiling= */ false,
          /* slimProfile= */ false,
          /* enableJsonMetadata= */ false);

      // Write from multiple threads to generate multiple rows in the chart.
      for (int i = 0; i < noOfRows; i++) {
        Thread t = new Thread(runnable);
        t.start();
        t.join();
      }

      profiler.stop();
    }
    try (InputStream in = cacheFile.getInputStream()) {
      return ProfileInfo.loadProfile(in);
    }
  }

  private void task(final Profiler profiler, ProfilerTask task, String name) {
    try (SilentCloseable c = profiler.profile(task, name)) {
      Thread.sleep(100);
    } catch (InterruptedException e) {
      // ignore
    }
  }

  private static final class TestingChartVisitor implements ChartVisitor {
    private int rowCount;
    private int endChartCount;
    private int barCount;
    private int beginChartCount;
    private int columnCount;
    private int lineCount;

    @Override
    public void visit(Chart chart) {
      beginChartCount++;
    }

    @Override
    public void visit(ChartRow chartRow) {
      rowCount++;
    }

    @Override
    public void visit(ChartBar chartBar) {
      barCount++;
    }

    @Override
    public void visit(ChartColumn chartColumn) {
      columnCount++;
    }

    @Override
    public void visit(ChartLine chartLine) {
      lineCount++;
    }

    @Override
    public void endVisit(Chart chart) {
      endChartCount++;
    }
  }
}
