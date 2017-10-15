/**
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
package com.tonicsystems.jarjar.dependencies;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nonnull;

public abstract class AbstractDependencyHandler implements DependencyHandler {

    protected final Level level;
    private final Set<Pair<String>> seen = new HashSet<Pair<String>>();

    protected AbstractDependencyHandler(Level level) {
        this.level = level;
    }

    @Override
    public void handle(Dependency from, Dependency to) throws IOException {
        Pair<String> pair;
        if (level == Level.JAR) {
            pair = new Pair<String>(from.getClassPath(), to.getClassPath());
        } else {
            pair = new Pair<String>(from.getClassName(), to.getClassName());
        }
        if (seen.add(pair))
            handle(pair.getLeft(), pair.getRight());
    }

    protected abstract void handle(@Nonnull String from, @Nonnull String to) throws IOException;

    @Override
    public void handleStart() throws IOException {
    }

    @Override
    public void handleEnd() throws IOException {
    }
}
