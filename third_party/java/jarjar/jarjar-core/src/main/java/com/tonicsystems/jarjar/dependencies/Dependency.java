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

import javax.annotation.Nonnull;

public class Dependency {

    private final String classPath;
    private final String className;

    public Dependency(@Nonnull String classPath, @Nonnull String className) {
        this.classPath = classPath;
        this.className = className;
    }

    @Nonnull
    public String getClassPath() {
        return classPath;
    }

    @Nonnull
    public String getClassName() {
        return className;
    }

    @Override
    public String toString() {
        return classPath + "!" + className;
    }
}
