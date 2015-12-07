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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;

import com.google.devtools.build.lib.profiler.Profiler.ProfiledTaskKinds;
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
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Unit tests for the profiler chart generation.
 */
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
        Profiler.instance().startTask(ProfilerTask.ACTION, "action");
        Profiler.instance().completeTask(ProfilerTask.ACTION);
      }
    };
    int threads = 4; // there is one extra thread due due the event that finalizes the profiler
    ProfileInfo info = createProfileInfo(run, threads - 1);
    ChartCreator aggregatingCreator = new AggregatingChartCreator(info, true);
    Chart aggregatedChart = aggregatingCreator.create();
    assertEquals(threads, aggregatedChart.getRowCount());
    assertThat(aggregatedChart.getSortedRows().get(0).getBars()).hasSize(1);

    ChartCreator detailedCreator = new DetailedChartCreator(info);
    Chart detailedChart = detailedCreator.create();
    assertThat(detailedChart.getSortedTypes()).hasSize(COMMON_CHART_TYPES + DETAILED_CHART_TYPES);
    assertEquals(threads, detailedChart.getRowCount());
    assertThat(detailedChart.getSortedRows().get(0).getBars()).hasSize(1);
  }

  @Test
  public void testAggregatingChartCreator() throws Exception {
    Runnable run = new Runnable() {
      @Override
      public void run() {
        Profiler profiler = Profiler.instance();
        profiler.startTask(ProfilerTask.ACTION, "action"); // Stays
        task(profiler, ProfilerTask.REMOTE_EXECUTION, "remote execution"); // Removed
        task(profiler, ProfilerTask.ACTION_CHECK, "check"); // Removed
        task(profiler, ProfilerTask.ACTION_LOCK, "lock"); // Stays
        profiler.completeTask(ProfilerTask.ACTION);
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
    assertEquals(type1.getName(), types.get(0).getName());
    assertEquals(type1.getColor(), types.get(0).getColor());
    assertEquals(type2.getName(), types.get(1).getName());
    assertEquals(type2.getColor(), types.get(1).getColor());
    assertEquals(type3.getName(), types.get(2).getName());
    assertEquals(type3.getColor(), types.get(2).getColor());

    assertSame(type3, chart.lookUpType("name3"));
    assertSame(type2, chart.lookUpType("name2"));
    assertSame(type1, chart.lookUpType("name1"));

    assertSame(Chart.UNKNOWN_TYPE, chart.lookUpType("wergl"));
    types = chart.getSortedTypes();
    assertThat(types).hasSize(4);

    chart.addBar(1, 2, 3, type1, "label1");
    chart.addBar(2, 3, 4, type2, "label2");
    chart.addBar(2, 4, 5, type3, "label3");
    chart.addBar(3, 3, 4, type2, "label4");
    chart.addBar(3, 4, 5, type3, "label5");
    chart.addBar(3, 5, 6, type3, "label6");

    assertEquals(6, chart.getMaxStop());
    assertEquals(3, chart.getRowCount());

    List<ChartRow> rows = chart.getSortedRows();
    assertThat(rows).hasSize(3);
    assertThat(rows.get(0).getBars()).hasSize(1);
    assertThat(rows.get(1).getBars()).hasSize(2);
    assertThat(rows.get(2).getBars()).hasSize(3);

    ChartBar bar = rows.get(0).getBars().get(0);
    assertEquals(2, bar.getStart());
    assertEquals(3, bar.getStop());
    assertSame(type1, bar.getType());
    assertEquals("label1", bar.getLabel());
  }

  @Test
  public void testChartRows() throws Exception {
    ChartRow row1 = new ChartRow("1", 0);
    ChartRow row2 = new ChartRow("2", 1);
    ChartRow row3 = new ChartRow("3", 1);

    assertEquals("1", row1.getId());
    assertEquals(0, row1.getIndex());

    assertEquals(-1, row1.compareTo(row2));
    assertEquals(1, row2.compareTo(row1));
    assertEquals(0, row2.compareTo(row3));

    row1.addBar(new ChartBar(row1, 1, 2, new ChartBarType("name1", Color.BLACK), false, "label1"));
    row1.addBar(new ChartBar(row1, 2, 3, new ChartBarType("name2", Color.RED), false, "label2"));

    assertThat(row1.getBars()).hasSize(2);
    assertEquals("label1", row1.getBars().get(0).getLabel());
    assertEquals("label2", row1.getBars().get(1).getLabel());
  }

  @Test
  public void testChartBarTypes() throws Exception {
    ChartBarType type1 = new ChartBarType("name1", Color.BLACK);
    ChartBarType type2 = new ChartBarType("name2", Color.RED);
    ChartBarType type3 = new ChartBarType("name2", Color.GREEN);

    assertEquals(-1, type1.compareTo(type2));
    assertEquals(1, type2.compareTo(type1));
    assertEquals(0, type2.compareTo(type3));

    assertEquals(type3, type2);
    assertFalse(type1.equals(type3));
    assertFalse(type1.equals(type2));

    assertEquals(type3.hashCode(), type2.hashCode());
    assertFalse(type1.hashCode() == type2.hashCode());
    assertFalse(type1.hashCode() == type3.hashCode());
  }

  @Test
  public void testChartBar() throws Exception {
    ChartRow row1 = new ChartRow("1", 0);
    ChartBarType type = new ChartBarType("name1", Color.BLACK);
    ChartBar bar1 = new ChartBar(row1, 1, 2, type, false, "label1");
    assertEquals(row1, bar1.getRow());
    assertEquals(1, bar1.getStart());
    assertEquals(2, bar1.getStop());
    assertSame(type, bar1.getType());
    assertEquals("label1", bar1.getLabel());
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
    assertEquals(1, visitor.beginChartCount);
    assertEquals(1, visitor.endChartCount);
    assertEquals(3, visitor.rowCount);
    assertEquals(6, visitor.barCount);
    assertEquals(0, visitor.columnCount);
    assertEquals(0, visitor.lineCount);
  }

  private ProfileInfo createProfileInfo(Runnable runnable, int noOfRows) throws Exception {
    Scratch scratch = new Scratch();
    Path cacheDir = scratch.dir("/tmp");
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    Profiler profiler = Profiler.instance();
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "basic test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());

    // Write from multiple threads to generate multiple rows in the chart.
    for (int i = 0; i < noOfRows; i++) {
      Thread t = new Thread(runnable);
      t.start();
      t.join();
    }

    profiler.stop();
    return ProfileInfo.loadProfile(cacheFile);
  }

  private void task(final Profiler profiler, ProfilerTask task, String name) {
    profiler.startTask(task, name);
    try {
      Thread.sleep(100);
    } catch (InterruptedException e) {
      // ignore
    }
    profiler.completeTask(task);
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
    public void endVisit(Chart chart) {
      endChartCount++;
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
  }
}
