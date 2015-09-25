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

/**
 * Represents a color in ARGB format, 8 bits per channel.
 */
public final class Color {
  public static final Color RED = new Color(0xff0000);
  public static final Color GREEN = new Color(0x00ff00);
  public static final Color GRAY = new Color(0x808080);
  public static final Color BLACK = new Color(0x000000);

  private final int argb;

  public Color(int rgb) {
    this.argb = rgb | 0xff000000;
  }

  public Color(int argb, boolean hasAlpha) {
    this.argb = argb;
  }

  public int getRed() {
    return (argb >> 16) & 0xFF;
  }

  public int getGreen() {
    return (argb >> 8) & 0xFF;
  }

  public int getBlue() {
    return argb & 0xFF;
  }

  public int getAlpha() {
    return (argb >> 24) & 0xFF;
  }
}

