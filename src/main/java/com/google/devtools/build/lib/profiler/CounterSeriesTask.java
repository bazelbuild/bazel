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
package com.google.devtools.build.lib.profiler;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import javax.annotation.Nullable;

/**
 * Describes counter series to be logged into profile.
 *
 * @param laneName The lane name for the counter series. Series with the same lane name should be
 *     stacked when displaying.
 * @param seriesName The name for the counter series.
 * @param color The color for the counter series. If {@code null}, the profile viewer will pick a
 *     color automatically.
 */
@SkybridgeInterface
public record CounterSeriesTask(String laneName, String seriesName, @Nullable Color color) {
  /** The reserved color for rendering the bar chart. */
  public static final class Color {
    // Pick acceptable counter colors manually, unfortunately we have to pick from these
    // weird reserved names from
    // https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
    public static final Color THREAD_STATE_UNINTERRUPTIBLE =
        new Color("thread_state_uninterruptible");
    public static final Color THREAD_STATE_IOWAIT = new Color("thread_state_iowait");
    public static final Color THREAD_STATE_RUNNING = new Color("thread_state_running");
    public static final Color THREAD_STATE_RUNNABLE = new Color("thread_state_runnable");
    public static final Color THREAD_STATE_SLEEPING = new Color("thread_state_sleeping");
    public static final Color THREAD_STATE_UNKNOWN = new Color("thread_state_unknown");
    public static final Color BACKGROUND_MEMORY_DUMP = new Color("background_memory_dump");
    public static final Color LIGHT_MEMORY_DUMP = new Color("light_memory_dump");
    public static final Color DETAILED_MEMORY_DUMP = new Color("detailed_memory_dump");
    public static final Color VSYNC_HIGHLIGHT_COLOR = new Color("vsync_highlight_color");
    public static final Color GENERIC_WORK = new Color("generic_work");
    public static final Color GOOD = new Color("good");
    public static final Color BAD = new Color("bad");
    public static final Color TERRIBLE = new Color("terrible");
    public static final Color BLACK = new Color("black");
    public static final Color GREY = new Color("grey");
    public static final Color WHITE = new Color("white");
    public static final Color YELLOW = new Color("yellow");
    public static final Color OLIVE = new Color("olive");
    public static final Color RAIL_RESPONSE = new Color("rail_response");
    public static final Color RAIL_ANIMATION = new Color("rail_animation");
    public static final Color RAIL_IDLE = new Color("rail_idle");
    public static final Color RAIL_LOAD = new Color("rail_load");
    public static final Color STARTUP = new Color("startup");
    public static final Color HEAP_DUMP_STACK_FRAME = new Color("heap_dump_stack_frame");
    public static final Color HEAP_DUMP_OBJECT_TYPE = new Color("heap_dump_object_type");
    public static final Color HEAP_DUMP_CHILD_NODE_ARROW = new Color("heap_dump_child_node_arrow");
    public static final Color CQ_BUILD_RUNNING = new Color("cq_build_running");
    public static final Color CQ_BUILD_PASSED = new Color("cq_build_passed");
    public static final Color CQ_BUILD_FAILED = new Color("cq_build_failed");
    public static final Color CQ_BUILD_ABANDONED = new Color("cq_build_abandoned");
    public static final Color CQ_BUILD_ATTEMPT_RUNNIG = new Color("cq_build_attempt_runnig");
    public static final Color CQ_BUILD_ATTEMPT_PASSED = new Color("cq_build_attempt_passed");
    public static final Color CQ_BUILD_ATTEMPT_FAILED = new Color("cq_build_attempt_failed");

    private final String value;

    private Color(String value) {
      this.value = value;
    }

    public String value() {
      return value;
    }
  }
}
