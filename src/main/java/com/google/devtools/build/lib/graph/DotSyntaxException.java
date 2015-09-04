// Copyright 2014 Google Inc. All rights reserved.
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
// All Rights Reserved.

package com.google.devtools.build.lib.graph;

/**
 *  <p> A DotSyntaxException represents a syntax error encountered while
 *  parsing a dot-format fule.  Thrown by createFromDotFile if syntax errors
 *  are encountered.  May also be thrown by implementations of
 *  LabelDeserializer. </p>
 *
 *  <p> The 'file' and 'lineNumber' fields indicate location of syntax error,
 *  and are populated externally by Digraph.createFromDotFile(). </p>
 */
public class DotSyntaxException extends Exception {

  public DotSyntaxException(String message) {
    super(message);
  }
}
