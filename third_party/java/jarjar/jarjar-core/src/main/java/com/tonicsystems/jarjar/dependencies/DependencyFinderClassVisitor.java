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

import com.tonicsystems.jarjar.util.RuntimeIOException;
import java.io.IOException;
import java.util.Map;
import org.objectweb.asm.commons.Remapper;
import org.objectweb.asm.commons.ClassRemapper;

class DependencyFinderClassVisitor extends ClassRemapper {

    public DependencyFinderClassVisitor(Map<String, String> classToArchiveMap, String archiveName, DependencyHandler handler) throws IOException {
        super(null, new DependencyFinderRemapper(classToArchiveMap, archiveName, handler));
    }

    @Override
    public void visit(int version, int access, String name, String signature, String superName, String[] interfaces) {
        ((DependencyFinderRemapper) remapper).setClassName(name);
        super.visit(version, access, name, signature, superName, interfaces);
    }

    private static class DependencyFinderRemapper extends Remapper {

        private final Map<String, String> classes;
        private final String archiveName;
        private final DependencyHandler handler;
        private Dependency curPathClass;

        public DependencyFinderRemapper(Map<String, String> classes, String archiveName, DependencyHandler handler) throws IOException {
            this.classes = classes;
            this.archiveName = archiveName;
            this.handler = handler;
        }

        public void setClassName(String name) {
            curPathClass = new Dependency(archiveName, name);
        }

        @Override
        public String map(String key) {
            try {
                if (classes.containsKey(key)) {
                    String otherSource = classes.get(key);
                    if (!archiveName.equals(otherSource)) {
                        // TODO: some escape mechanism?
                        handler.handle(curPathClass, new Dependency(otherSource, key));
                    }
                }
            } catch (IOException e) {
                throw new RuntimeIOException(e);
            }
            return null;
        }
    }
}
