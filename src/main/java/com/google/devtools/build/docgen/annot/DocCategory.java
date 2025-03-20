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
package com.google.devtools.build.docgen.annot;

/** Sections of the API reference for the Bazel build language. */
public final class DocCategory {

  private DocCategory() {} // uninstantiable

  public static final String CONFIGURATION_FRAGMENT = "CONFIGURATION_FRAGMENT";

  // TODO(adonovan): be more rigorous about distinguishing providers
  // from provider instances (aka structs/infos).
  public static final String PROVIDER = "PROVIDER";

  public static final String BUILTIN = "BUILTIN";

  public static final String TOP_LEVEL_MODULE = "TOP_LEVEL_MODULE";
}
