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
package com.google.devtools.build.lib.analysis.stringtemplate;

/**
 * Interface to be implemented by callers of MakeVariableExpander which defines the expansion of
 * each "Make" variable.
 */
public interface TemplateContext {

  /**
   * Returns the expansion of the specified "Make" variable.
   *
   * @param name the variable to expand.
   * @return the expansion of the variable.
   * @throws ExpansionException if the variable "var" was not defined or
   *     there was any other error while expanding "var".
   */
  String lookupVariable(String name) throws ExpansionException;
}