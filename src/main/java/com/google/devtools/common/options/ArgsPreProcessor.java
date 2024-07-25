// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import java.util.List;

/** Defines a preprocessing service for the "args" string list that is executed before parsing. */
@FunctionalInterface
interface ArgsPreProcessor {
  List<OptionsParserImpl.ArgAndFallbackData> preProcess(
      List<OptionsParserImpl.ArgAndFallbackData> args) throws OptionsParsingException;
}
