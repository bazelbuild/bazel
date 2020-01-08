/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.testing.junit;

/**
 * Enumeration of the currently known Java ClassFile versions.
 *
 * <p>The minor version is stored in the 16 most significant bits, and the major version in the 16
 * least significant bits. These are the values that can be found in
 * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-4.html#jvms-4.1-200-B.2
 */
public final class JdkVersion {

  public static final int V1_1 = 3 << 16 | 45;
  public static final int V1_2 = 0 << 16 | 46;
  public static final int V1_3 = 0 << 16 | 47;
  public static final int V1_4 = 0 << 16 | 48;
  public static final int V1_5 = 0 << 16 | 49;
  public static final int V1_6 = 0 << 16 | 50;
  public static final int V1_7 = 0 << 16 | 51;
  public static final int V1_8 = 0 << 16 | 52;
  public static final int V9 = 0 << 16 | 53;
  public static final int V10 = 0 << 16 | 54;
  public static final int V11 = 0 << 16 | 55;
  public static final int V12 = 0 << 16 | 56;
  public static final int V13 = 0 << 16 | 57;
  public static final int V14 = 0 << 16 | 58;

  private JdkVersion() {}
}
