// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler.chart;

import com.google.common.base.Preconditions;

/**
 * The type of a bar in a Gantt Chart. A type consists of a name and a color.
 * Types are used to create the legend of a Gantt Chart.
 */
public class ChartBarType implements Comparable<ChartBarType> {

  /** The name of the type. */
  private final String name;

  /** The color of the type. */
  private final Color color;

  /**
   * Creates a {@link ChartBarType}.
   *
   * @param name the name of the type
   * @param color the color of the type
   */
  public ChartBarType(String name, Color color) {
    Preconditions.checkNotNull(name);
    Preconditions.checkNotNull(color);
    this.name = name;
    this.color = color;
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }

  /**
   * {@inheritDoc}
   *
   * <p>Equality of two types is defined by the equality of their names.
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    return name.equals(((ChartBarType) obj).name);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Compares types by their names.
   */
  @Override
  public int compareTo(ChartBarType o) {
    return name.compareTo(o.name);
  }

  public String getName() {
    return name;
  }

  public Color getColor() {
    return color;
  }
}
