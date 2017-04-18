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

package joptsimple;

import java.util.Collections;
import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class EmptyOptionSetTest {
    private OptionSet empty;

    @Before
    public void setUp() {
        empty = new OptionSet( Collections.<String, AbstractOptionSpec<?>> emptyMap() );
        empty.add( new NonOptionArgumentSpec<>() );
    }

    @Test
    public void valueOf() {
        assertNull( empty.valueOf( "a" ) );
    }

    @Test
    public void valuesOf() {
        assertEquals( emptyList(), empty.valuesOf( "a" ) );
    }

    @Test
    public void hasArgument() {
        assertFalse( empty.hasArgument( "a" ) );
    }

    @Test
    public void hasOptions() {
        assertFalse( empty.hasOptions() );
    }

    @Test( expected = NullPointerException.class )
    public void valueOfWithNullString() {
        empty.valueOf( (String) null );
    }

    @Test( expected = NullPointerException.class )
    public void valueOfWithNullOptionSpec() {
        empty.valueOf( (OptionSpec<?>) null );
    }

    @Test( expected = NullPointerException.class )
    public void valuesOfWithNullString() {
        empty.valuesOf( (String) null );
    }

    @Test( expected = NullPointerException.class )
    public void valuesOfWithNullOptionSpec() {
        empty.valuesOf( (OptionSpec<?>) null );
    }
}
