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

import java.util.HashMap;
import java.util.Map;

/**
 * <p>An {@code OptionNameMap} which wraps and behaves like {@code HashMap}.</p>
 */
public class SimpleOptionNameMap<V> implements OptionNameMap<V> {
    private final Map<String, V> map = new HashMap<>();

    @Override
    public boolean contains( String key ) {
        return map.containsKey( key );
    }

    @Override
    public V get( String key ) {
        return map.get( key );
    }

    @Override
    public void put( String key, V newValue ) {
        map.put( key, newValue );
    }

    @Override
    public void putAll( Iterable<String> keys, V newValue ) {
        for ( String each : keys )
            map.put( each, newValue );
    }

    @Override
    public void remove( String key ) {
        map.remove( key );
    }

    @Override
    public Map<String, V> toJavaUtilMap() {
        return new HashMap<>( map );
    }
}
