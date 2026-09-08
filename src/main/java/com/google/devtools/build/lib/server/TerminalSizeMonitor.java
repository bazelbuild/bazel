// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.server;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;

/** Tracks terminal size updates for a single running command. */
public final class TerminalSizeMonitor {
  public static final TerminalSizeMonitor NOOP = new TerminalSizeMonitor(/* enabled= */ false);

  /** Listener for terminal size updates. */
  public interface Listener {
    void terminalSizeChanged(int columns, int rows);
  }

  private final Object lock = new Object();
  private final boolean enabled;
  private final List<Listener> listeners = new ArrayList<>();
  private int columns;
  private int rows;

  public TerminalSizeMonitor() {
    this(/* enabled= */ true);
  }

  private TerminalSizeMonitor(boolean enabled) {
    this.enabled = enabled;
  }

  public void addListener(Listener listener) {
    if (!enabled) {
      return;
    }
    int columnsSnapshot;
    int rowsSnapshot;
    synchronized (lock) {
      listeners.add(listener);
      columnsSnapshot = columns;
      rowsSnapshot = rows;
    }
    if (columnsSnapshot > 0) {
      listener.terminalSizeChanged(columnsSnapshot, rowsSnapshot);
    }
  }

  public void updateTerminalSize(int columns, int rows) {
    if (!enabled || columns <= 0) {
      return;
    }
    ImmutableList<Listener> listenersSnapshot;
    synchronized (lock) {
      if (this.columns == columns && this.rows == rows) {
        return;
      }
      this.columns = columns;
      this.rows = rows;
      listenersSnapshot = ImmutableList.copyOf(listeners);
    }
    for (Listener listener : listenersSnapshot) {
      listener.terminalSizeChanged(columns, rows);
    }
  }
}
