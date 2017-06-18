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

import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ClusterWhereManyOptionsAcceptArgumentsTest extends AbstractOptionParserFixture {
    @Before
    public final void initializeParser() {
        parser.accepts( "f" );
        parser.accepts( "o" ).withOptionalArg();
        parser.accepts( "x" ).withRequiredArg();
    }

    @Test
    public void foxPermutation() {
        OptionSet options = parser.parse( "-fox" );

        assertTrue( options.has( "f" ) );
        assertTrue( options.has( "o" ) );
        assertFalse( options.has( "x" ) );

        assertEquals( singletonList( "x" ), options.valuesOf( "o" ) );
    }

    @Test
    public void fxoPermutation() {
        OptionSet options = parser.parse( "-fxo" );

        assertTrue( options.has( "f" ) );
        assertFalse( options.has( "o" ) );
        assertTrue( options.has( "x" ) );

        assertEquals( singletonList( "o" ), options.valuesOf( "x" ) );
    }

    @Test
    public void ofxPermutation() {
        OptionSet options = parser.parse( "-ofx" );

        assertFalse( options.has( "f" ) );
        assertTrue( options.has( "o" ) );
        assertFalse( options.has( "x" ) );

        assertEquals( singletonList( "fx" ), options.valuesOf( "o" ) );
    }

    @Test
    public void oxfPermutation() {
        OptionSet options = parser.parse( "-oxf" );

        assertFalse( options.has( "f" ) );
        assertTrue( options.has( "o" ) );
        assertFalse( options.has( "x" ) );

        assertEquals( singletonList( "xf" ), options.valuesOf( "o" ) );
    }

    @Test
    public void xofPermutation() {
        OptionSet options = parser.parse( "-xof" );

        assertFalse( options.has( "f" ) );
        assertFalse( options.has( "o" ) );
        assertTrue( options.has( "x" ) );

        assertEquals( singletonList( "of" ), options.valuesOf( "x" ) );
    }

    @Test
    public void xfoPermutation() {
        OptionSet options = parser.parse( "-xfo" );

        assertFalse( options.has( "f" ) );
        assertFalse( options.has( "o" ) );
        assertTrue( options.has( "x" ) );

        assertEquals( singletonList( "fo" ), options.valuesOf( "x" ) );
    }

    @Test
    public void foxPermutationWithFollowingArg() {
        OptionSet options = parser.parse( "-fox", "bar" );

        assertTrue( options.has( "f" ) );
        assertTrue( options.has( "o" ) );
        assertFalse( options.has( "x" ) );

        assertEquals( singletonList( "x" ), options.valuesOf( "o" ) );
        assertEquals( singletonList( "bar" ), options.nonOptionArguments() );
    }

    @Test
    public void fxoPermutationWithFollowingArg() {
        OptionSet options = parser.parse( "-fxo", "bar" );

        assertTrue( options.has( "f" ) );
        assertFalse( options.has( "o" ) );
        assertTrue( options.has( "x" ) );

        assertEquals( singletonList( "o" ), options.valuesOf( "x" ) );
        assertEquals( singletonList( "bar" ), options.nonOptionArguments() );
    }

    @Test
    public void ofxPermutationWithFollowingArg() {
        OptionSet options = parser.parse( "-ofx", "bar" );

        assertFalse( options.has( "f" ) );
        assertTrue( options.has( "o" ) );
        assertFalse( options.has( "x" ) );

        assertEquals( singletonList( "fx" ), options.valuesOf( "o" ) );
        assertEquals( singletonList( "bar" ), options.nonOptionArguments() );
    }

    @Test
    public void oxfPermutationWithFollowingArg() {
        OptionSet options = parser.parse( "-oxf", "bar" );

        assertFalse( options.has( "f" ) );
        assertTrue( options.has( "o" ) );
        assertFalse( options.has( "x" ) );

        assertEquals( singletonList( "xf" ), options.valuesOf( "o" ) );
        assertEquals( singletonList( "bar" ), options.nonOptionArguments() );
    }

    @Test
    public void xofPermutationWithFollowingArg() {
        OptionSet options = parser.parse( "-xof", "bar" );

        assertFalse( options.has( "f" ) );
        assertFalse( options.has( "o" ) );
        assertTrue( options.has( "x" ) );

        assertEquals( singletonList( "of" ), options.valuesOf( "x" ) );
        assertEquals( singletonList( "bar" ), options.nonOptionArguments() );
    }

    @Test
    public void xfoPermutationWithFollowingArg() {
        OptionSet options = parser.parse( "-xfo", "bar" );

        assertFalse( options.has( "f" ) );
        assertFalse( options.has( "o" ) );
        assertTrue( options.has( "x" ) );

        assertEquals( singletonList( "fo" ), options.valuesOf( "x" ) );
        assertEquals( singletonList( "bar" ), options.nonOptionArguments() );
    }
}
