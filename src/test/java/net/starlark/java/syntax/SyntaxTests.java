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
package net.starlark.java.syntax;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** SyntaxTests tests the syntax package (Starlark frontend). */
@RunWith(Suite.class)
@Suite.SuiteClasses({
  FileLocationsTest.class,
  LexerTest.class,
  LocationTest.class,
  LValueBoundNamesTest.class,
  NodePrinterTest.class,
  NodeVisitorTest.class,
  ParserInputTest.class,
  ParserTest.class,
  ProgramTest.class,
  ResolverTest.class,
  StarlarkFileTest.class,
  StarlarkTypesTest.class,
})
public class SyntaxTests {}
