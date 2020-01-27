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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * A graph that is both Dirtiable (values can be deleted) and Evaluable (values can be added). All
 * methods in this interface (as inherited from super-interfaces) should be thread-safe.
 *
 * <p>This class is not intended for direct use, and is only exposed as public for use in evaluation
 * implementations outside of this package.
 */
@ThreadSafe
public interface ProcessableGraph extends DeletableGraph, EvaluableGraph {}
