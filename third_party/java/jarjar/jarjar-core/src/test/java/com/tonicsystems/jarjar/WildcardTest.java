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
package com.tonicsystems.jarjar;

import com.tonicsystems.jarjar.transform.config.ResourceRename;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static org.junit.Assert.*;

public class WildcardTest {

    private static final Logger LOG = LoggerFactory.getLogger(WildcardTest.class);

    @Test
    public void testWildcards() {
        wildcard("net/sf/cglib/**", "foo/@1", "net/sf/cglib/proxy/Mixin$Generator",
                "foo/proxy/Mixin$Generator");
        wildcard("net/sf/cglib/**", "foo/@1", "net/sf/cglib/Bar", "foo/Bar");
        wildcard("net/sf/cglib/**", "foo/@1", "net/sf/cglib/Bar/Baz", "foo/Bar/Baz");
        wildcard("net/sf/cglib/**", "foo/@1", "net/sf/cglib/", "foo/");
        wildcard("net/sf/cglib/**", "foo/@1", "net/sf/cglib/!", null);
        wildcard("net/sf/cglib/*", "foo/@1", "net/sf/cglib/Bar", "foo/Bar");
        wildcard("net/sf/cglib/*/*", "foo/@2/@1", "net/sf/cglib/Bar/Baz", "foo/Baz/Bar");
    }

    private static void wildcard(String pattern, String result, String value, String expect) {
        ResourceRename wc = new ResourceRename(pattern, result);
        LOG.info("Compile: " + pattern + " -> " + wc);
        String actual = wc.replace(value);
        LOG.info("Replace: " + value + " -> " + actual + " (expected " + expect + ")");
        assertEquals(expect, actual);
    }
}
