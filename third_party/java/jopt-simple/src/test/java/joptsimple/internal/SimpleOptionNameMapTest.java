/*
 The MIT License

 Copyright (c) 2004-2015 Paul R. Holser, Jr.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package joptsimple.internal;

import java.util.List;
import java.util.Map;

import org.junit.Test;

import static java.util.Arrays.*;

import static org.junit.Assert.*;

public class SimpleOptionNameMapTest {
    private static final Integer VALUE = 1;
    private static final String KEY = "someKey";
    private static final String KEY2 = "someOtherKey";

    @Test
    public void putAndContains() {
        SimpleOptionNameMap<Integer> map = new SimpleOptionNameMap<>();
        assertFalse( map.contains( KEY ) );

        map.put( KEY, 1 );

        assertTrue( map.contains( KEY ) );
    }

    @Test
    public void get() {
        SimpleOptionNameMap<Integer> map = new SimpleOptionNameMap<>();
        assertNull( map.get( KEY ) );

        map.put( KEY, VALUE );

        assertEquals( VALUE, map.get( KEY ) );
    }

    @Test
    public void putAll() {
        SimpleOptionNameMap<Integer> map = new SimpleOptionNameMap<>();
        List<String> keys = asList( KEY, KEY2 );

        map.putAll( keys, VALUE );

        assertEquals( VALUE, map.get( KEY ) );
        assertEquals( VALUE, map.get( KEY2 ) );
    }

    @Test
    public void remove() {
        SimpleOptionNameMap<Integer> map = new SimpleOptionNameMap<>();
        map.put( KEY, 1 );

        map.remove( KEY );

        assertFalse( map.contains( KEY ) );
    }

    @Test
    public void toJavaUtilMap() {
        SimpleOptionNameMap<Integer> map = new SimpleOptionNameMap<>();
        map.put( KEY, VALUE );

        Map<String, Integer> javaUtilMap = map.toJavaUtilMap();

        assertEquals( VALUE, javaUtilMap.get( KEY ) );
        assertEquals( 1, javaUtilMap.size() );
    }
}
