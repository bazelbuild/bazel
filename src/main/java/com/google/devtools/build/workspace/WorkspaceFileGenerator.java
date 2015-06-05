// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.workspace;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.workspace.maven.Resolver;

import java.io.File;

/**
 * Generates a WORKSPACE file for Bazel from other types of dependency trackers.
 */
public class WorkspaceFileGenerator {

  public static void main(String[] args) {
    String directory;
    if (args.length == 1) {
      directory = args[0];
    } else {
      directory = System.getProperty("user.dir");
    }
    StoredEventHandler handler = new StoredEventHandler();
    Resolver connector = new Resolver(new File(directory), handler);
    connector.writeDependencies(System.out);
    if (handler.hasErrors()) {
      for (Event event : handler.getEvents()) {
        System.err.println(event);
      }
    }
  }

}
