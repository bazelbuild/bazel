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

import static java.util.Arrays.*;
import static java.util.Collections.*;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class InterleavedArgumentsTest {
    @Test
    public void onlyAppearingToHaveOptionArguments() {
        OptionParser parser = new OptionParser( "c" );

        OptionSet options = parser.parse( "-c", "a", "-c", "b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( emptyList(), options.valuesOf( "c" ) );
        assertEquals( asList( "a", "b", "c", "d" ), options.nonOptionArguments() );
    }

    @Test
    public void onlyAppearingToHaveOptionArgumentsButPosixlyCorrect() {
        OptionParser parser = new OptionParser( "+c" );

        OptionSet options = parser.parse( "-c", "a", "-c", "b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( emptyList(), options.valuesOf( "c" ) );
        assertEquals(
            asList( "a", "-c", "b", "-c", "c", "-c", "d" ),
            options.nonOptionArguments() );
    }

    @Test
    public void requiredArgument() {
        OptionParser parser = new OptionParser( "c:" );

        OptionSet options = parser.parse( "-c", "a", "-c", "b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( asList( "a", "b", "c", "d" ), options.valuesOf( "c" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void requiredArgumentAndPosixlyCorrect() {
        OptionParser parser = new OptionParser( "+c:" );

        OptionSet options = parser.parse( "-c", "a", "-c", "b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( asList( "a", "b", "c", "d" ), options.valuesOf( "c" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void optionalArgument() {
        OptionParser parser = new OptionParser( "c::" );

        OptionSet options = parser.parse( "-c", "a", "-c", "b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( asList( "a", "b", "c", "d" ), options.valuesOf( "c" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void optionalArgumentAndPosixlyCorrect() {
        OptionParser parser = new OptionParser( "+c::" );

        OptionSet options = parser.parse( "-c", "a", "-c", "b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( emptyList(), options.valuesOf( "c" ) );
        assertEquals( asList( "a", "-c", "b", "-c", "c", "-c", "d" ), options.nonOptionArguments() );
    }

    @Test
    public void leadingNonOptionCausesPosixlyCorrectToIgnoreRemainder() {
        OptionParser parser = new OptionParser( "+c:" );
        String[] args = { "boo", "-c", "a", "-c", "b", "-c", "c", "-c", "d" };

        OptionSet options = parser.parse( args );

        assertFalse( options.has( "c" ) );
        assertEquals( emptyList(), options.valuesOf( "c" ) );
        assertEquals( asList( args ), options.nonOptionArguments() );
    }

    @Test
    public void optionalAbuttedArgumentVersusPosixlyCorrect() {
        OptionParser parser = new OptionParser( "+c::" );

        OptionSet options = parser.parse( "-ca", "-cb", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( asList( "a", "b" ), options.valuesOf( "c" ) );
        assertEquals( asList( "c", "-c", "d" ), options.nonOptionArguments() );
    }

    @Test
    public void optionalKeyValuePairArgumentVersusPosixlyCorrect() {
        OptionParser parser = new OptionParser( "+c::" );

        OptionSet options = parser.parse( "-c=a", "-c=b", "-c", "c", "-c", "d" );

        assertTrue( options.has( "c" ) );
        assertEquals( asList( "a", "b" ), options.valuesOf( "c" ) );
        assertEquals( asList( "c", "-c", "d" ), options.nonOptionArguments() );
    }
}
