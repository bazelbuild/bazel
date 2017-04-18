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

import com.tonicsystems.jarjar.classpath.ClassPath;
import com.tonicsystems.jarjar.classpath.ClassPathArchive;
import com.tonicsystems.jarjar.classpath.ClassPathResource;
import com.tonicsystems.jarjar.util.RuntimeIOException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.objectweb.asm.ClassReader;

public class DependencyFinder {

    private File curDir = new File(System.getProperty("user.dir"));

    public void setCurrentDirectory(File curDir) {
        this.curDir = curDir;
    }

    public void run(DependencyHandler handler, ClassPath from, ClassPath to) throws IOException {
        try {
            ClassHeaderReader header = new ClassHeaderReader();
            Map<String, String> classToArchiveMap = new HashMap<String, String>();
            for (ClassPathArchive toArchive : to) {
                for (ClassPathResource toResource : toArchive) {
                    InputStream in = toResource.openStream();
                    try {
                        header.read(in);
                        classToArchiveMap.put(header.getClassName(), toArchive.getArchiveName());
                    } catch (Exception e) {
                        System.err.println("Error reading " + toResource.getName() + ": " + e.getMessage());
                    } finally {
                        in.close();
                    }
                }
            }

            handler.handleStart();
            for (ClassPathArchive fromArchive : from) {
                for (ClassPathResource fromResource : fromArchive) {
                    InputStream in = fromResource.openStream();
                    try {
                        new ClassReader(in).accept(new DependencyFinderClassVisitor(classToArchiveMap, fromArchive.getArchiveName(), handler),
                                ClassReader.SKIP_DEBUG);
                    } catch (Exception e) {
                        System.err.println("Error reading " + fromResource.getName() + ": " + e.getMessage());
                    } finally {
                        in.close();
                    }
                }
            }
            handler.handleEnd();
        } catch (RuntimeIOException e) {
            throw (IOException) e.getCause();
        }
    }
}
