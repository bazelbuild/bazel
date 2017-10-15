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
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public final class Classes {
    private static final Map<Class<?>, Class<?>> WRAPPERS = new HashMap<>( 13 );

    static {
        WRAPPERS.put( boolean.class, Boolean.class );
        WRAPPERS.put( byte.class, Byte.class );
        WRAPPERS.put( char.class, Character.class );
        WRAPPERS.put( double.class, Double.class );
        WRAPPERS.put( float.class, Float.class );
        WRAPPERS.put( int.class, Integer.class );
        WRAPPERS.put( long.class, Long.class );
        WRAPPERS.put( short.class, Short.class );
        WRAPPERS.put( void.class, Void.class );
    }

    private Classes() {
        throw new UnsupportedOperationException();
    }

    /**
     * Gives the "short version" of the given class name.  Somewhat naive to inner classes.
     *
     * @param className class name to chew on
     * @return the short name of the class
     */
    public static String shortNameOf( String className ) {
        return className.substring( className.lastIndexOf( '.' ) + 1 );
    }

    /**
     * Gives the primitive wrapper class for the given class. If the given class is not
     * {@linkplain Class#isPrimitive() primitive}, returns the class itself.
     *
     * @param <T> generic class type
     * @param clazz the class to check
     * @return primitive wrapper type if {@code clazz} is primitive, otherwise {@code clazz}
     */
    @SuppressWarnings( "unchecked" )
    public static <T> Class<T> wrapperOf( Class<T> clazz ) {
        return clazz.isPrimitive() ? (Class<T>) WRAPPERS.get( clazz ) : clazz;
    }
}
