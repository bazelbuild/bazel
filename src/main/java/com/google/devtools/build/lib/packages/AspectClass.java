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

package com.google.devtools.build.lib.packages;

/**
 *  A class of aspects.
 *
 *  <p>This interface serves as a factory for {@code AspectFactory}.
 *  {@code AspectFactory} type argument is a placeholder for
 *  a {@link com.google.devtools.build.lib.analysis.ConfiguredAspectFactory}, which is
 *  an analysis-phase class. All loading-phase code uses {@code AspectClass&lt;?&gt;},
 *  whereas analysis-phase code uses {@code AspectClass&lt;ConfiguredAspectFactory&gt;}.
 *  The latter is what all real implementations of this interface should implement.
 *
 */
public interface AspectClass {

  /**
   * Returns an aspect name.
   */
  String getName();

  AspectDefinition getDefinition(AspectParameters aspectParameters);
}
