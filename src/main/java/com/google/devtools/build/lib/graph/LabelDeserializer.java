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
 *  <p> An interface for specifying a graph label de-serialization
 *  function. </p>
 *
 *  <p> Implementations should provide a means of mapping from the string
 *  representation to an instance of the graph label type T. </p>
 *
 *  <p> e.g. to construct Digraph{Integer} from a String representation, the
 *  LabelDeserializer{Integer} implementation would return
 *  Integer.parseInt(rep). </p>
 */
public interface LabelDeserializer<T> {

  /**
   *  Returns an instance of the label object (of type T)
   *  corresponding to serialized representation 'rep'.
   *
   *  @throws DotSyntaxException if 'rep' is invalid.
   */
  T deserialize(String rep) throws DotSyntaxException;
}
