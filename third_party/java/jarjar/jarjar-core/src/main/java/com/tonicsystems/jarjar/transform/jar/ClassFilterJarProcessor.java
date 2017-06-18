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
package com.tonicsystems.jarjar.transform.jar;

import com.tonicsystems.jarjar.transform.config.AbstractPattern;
import com.tonicsystems.jarjar.transform.config.ClassDelete;
import com.tonicsystems.jarjar.transform.config.ClassKeep;
import com.tonicsystems.jarjar.util.ClassNameUtils;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Filters classes by name.
 *
 * Keeps all classes specified by ClassKeep (default all classes).
 * Then removes all classes specified by ClassDelete (default no classes).
 * Ignores non-class resources.
 *
 * @see ClassNameUtils#isClass(String)
 * @author shevek
 */
public class ClassFilterJarProcessor extends AbstractFilterJarProcessor {

    // private static final Logger LOG = LoggerFactory.getLogger(ClassFilterJarProcessor.class);
    private final List<ClassKeep> keepPatterns = new ArrayList<ClassKeep>();
    private final List<ClassDelete> deletePatterns = new ArrayList<ClassDelete>();

    public void addClassKeep(@Nonnull ClassKeep pattern) {
        keepPatterns.add(pattern);
    }

    public void addClassDelete(@Nonnull ClassDelete pattern) {
        deletePatterns.add(pattern);
    }

    @CheckForNull
    protected <T extends AbstractPattern> T getMatchingPattern(@Nonnull List<? extends T> patterns, @Nonnull String name) {
        for (T pattern : patterns) {
            if (pattern.matches(name)) {
                // LOG.debug(pattern + " matches " + name);
                return pattern;
            }
        }
        // LOG.debug("No pattern matches " + name);
        return null;
    }

    @Override
    protected boolean isFiltered(@Nonnull String name) {
        if (!ClassNameUtils.isClass(name))
            return false;
        name = name.substring(0, name.length() - 6);
        // LOG.debug("Looking to include " + name);
        INCLUDE:
        {
            if (keepPatterns.isEmpty())
                break INCLUDE;
            if (getMatchingPattern(keepPatterns, name) != null)
                break INCLUDE;
            // We have include patterns, but none matched. Filter it.
            return true;
        }
        // LOG.debug("Looking to exclude " + name);
        return getMatchingPattern(deletePatterns, name) != null;
    }
}
