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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.syntax.BuildFileAST;

/** The result of parsing a BUILD file. */
public class AstParseResult {
  public final BuildFileAST ast;
  public final Iterable<Event> allEvents;
  public final Iterable<Postable> allPosts;

  public AstParseResult(BuildFileAST ast, StoredEventHandler astParsingEventHandler) {
    this.ast = ast;
    this.allPosts = astParsingEventHandler.getPosts();
    this.allEvents = astParsingEventHandler.getEvents();
  }
}
