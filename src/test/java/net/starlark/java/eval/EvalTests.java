// Copyright 2020 The Bazel Authors. All rights reserved.
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
package net.starlark.java.eval;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** EvalTests tests the Starlark evaluator. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
  EvaluationTest.class,
  EvalUtilsTest.class,
  FunctionTest.class,
  ImmutableKeyTrackingDictTest.class,
  MethodLibraryTest.class,
  MutabilityTest.class,
  PrinterTest.class,
  StarlarkClassTest.class,
  StarlarkEvaluationTest.class,
  StarlarkFlagGuardingTest.class,
  StarlarkAnnotationsTest.class,
  StarlarkListTest.class,
  StarlarkMutableTest.class,
  StarlarkThreadDebuggingTest.class,
  StarlarkThreadTest.class,
})
public class EvalTests {}
