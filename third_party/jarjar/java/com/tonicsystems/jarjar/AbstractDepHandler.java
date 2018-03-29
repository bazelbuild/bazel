/*
 * Copyright 2007 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.tonicsystems.jarjar;

import com.tonicsystems.jarjar.util.*;
import java.io.*;
import java.util.*;

public abstract class AbstractDepHandler implements DepHandler {
  protected final int level;
  private final Set<List<Object>> seenIt = new HashSet<List<Object>>();

  protected AbstractDepHandler(int level) {
    this.level = level;
  }

  public void handle(PathClass from, PathClass to) throws IOException {
    List<Object> pair;
    if (level == LEVEL_JAR) {
      pair = createPair(from.getClassPath(), to.getClassPath());
    } else {
      pair = createPair(from.getClassName(), to.getClassName());
    }
    if (!seenIt.contains(pair)) {
      seenIt.add(pair);
      handle(pair.get(0).toString(), pair.get(1).toString());
    }
  }

  protected abstract void handle(String from, String to) throws IOException;

  public void handleStart() throws IOException {}

  public void handleEnd() throws IOException {}

  private static List<Object> createPair(Object o1, Object o2) {
    List<Object> list = new ArrayList<Object>(2);
    list.add(o1);
    list.add(o2);
    return list;
  }
}
