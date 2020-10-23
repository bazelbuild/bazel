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

package com.google.devtools.build.android.desugar.typehierarchy.testlib.pkga;

import com.google.devtools.build.android.desugar.typehierarchy.testlib.ClassZulu;
import com.google.devtools.build.android.desugar.typehierarchy.testlib.pkgb.ClassBravo;
import com.google.devtools.build.android.desugar.typehierarchy.testlib.pkgb.InterfaceBravo;

/**
 * @see {@link com.google.devtools.build.android.desugar.typehierarchy.TypeHierarchyTest} for type
 *     inheritance structure and dynamic-dispatchable method relationships.
 */
public class ClassAlpha extends ClassBravo implements InterfaceAlpha, InterfaceBravo {

  private static final String TAG = "ca";

  @Override
  public String getTag() {
    return TAG;
  }

  @Override
  public InterfaceAlpha defaultInstance(ClassZulu instance) {
    return instance == null ? this : instance;
  }
}
