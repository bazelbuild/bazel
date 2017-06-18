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

import com.tonicsystems.jarjar.transform.asm.PackageRemapper;
import com.tonicsystems.jarjar.transform.config.ClassRename;
import java.util.Collections;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class PackageRemapperTest {

    protected PackageRemapper remapper;

    @Before
    public void setUp() {
        ClassRename rule = new ClassRename("org.**", "foo.@1");
        remapper = new PackageRemapper(Collections.singletonList(rule));
    }

    @Test
    public void testMapValue() {
        assertUnchangedValue("[^\\s;/@&=,.?:+$]");
        assertUnchangedValue("[Ljava/lang/Object;");
        assertUnchangedValue("[Lorg/example/Object;");
        assertUnchangedValue("[Ljava.lang.Object;");
        assertUnchangedValue("[Lorg.example/Object;");
        assertUnchangedValue("[L;");
        assertUnchangedValue("[Lorg.example.Object;;");
        assertUnchangedValue("[Lorg.example.Obj ct;");
        assertUnchangedValue("org.example/Object");

        assertEquals("[Lfoo.example.Object;", remapper.mapValue("[Lorg.example.Object;"));
        assertEquals("foo.example.Object", remapper.mapValue("org.example.Object"));
        assertEquals("foo/example/Object", remapper.mapValue("org/example/Object"));
        assertEquals("foo/example.Object", remapper.mapValue("org/example.Object")); // path match

        assertEquals("foo.example.package-info", remapper.mapValue("org.example.package-info"));
        assertEquals("foo/example/package-info", remapper.mapValue("org/example/package-info"));
        assertEquals("foo/example.package-info", remapper.mapValue("org/example.package-info"));
    }

    private void assertUnchangedValue(String value) {
        assertEquals(value, remapper.mapValue(value));
    }
}
