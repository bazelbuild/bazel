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

import java.math.BigDecimal;
import java.util.List;

import static java.math.BigDecimal.*;
import static java.util.Arrays.*;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class HandlingDefaultValuesForOptionArgumentsTest extends AbstractOptionParserFixture {
    @Test
    public void requiredArgOptionWithDefaultGivesArgIfArgSpecifiedOnCommandLine() {
        OptionSpec<Integer> optionA =
            parser.accepts( "a" ).withRequiredArg().ofType( Integer.class ).defaultsTo( 2 );

        OptionSet options = parser.parse( "-a", "1" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.has( optionA ) );
        assertTrue( options.hasArgument( "a" ) );
        assertTrue( options.hasArgument( optionA ) );
        Integer expectedArgument = 1;
        assertEquals( expectedArgument, options.valueOf( "a" ) );
        assertEquals( expectedArgument, options.valueOf( optionA ) );
        assertEquals( expectedArgument, optionA.value( options ) );
        assertEquals( asList( expectedArgument ), options.valuesOf( "a" ) );
        assertEquals( asList( expectedArgument ), options.valuesOf( optionA ) );
        assertEquals( asList( expectedArgument ), optionA.values( options ) );
    }

    @Test( expected = OptionMissingRequiredArgumentException.class )
    public void requiredArgOptionWithDefaultStillsFailToParseIfArgNotSpecifiedOnCommandLine() {
        parser.accepts( "a" ).withRequiredArg().defaultsTo( "boo" );

        parser.parse( "-a" );
    }

    @Test
    public void optionalArgOptionWithDefaultGivesDefaultIfArgNotSpecifiedOnCommandLine() {
        OptionSpec<Long> optionA =
            parser.accepts( "a" ).withOptionalArg().ofType( Long.class ).defaultsTo( -1L );

        OptionSet options = parser.parse( "-a" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.has( optionA ) );
        assertFalse( options.hasArgument( "a" ) );
        assertFalse( options.hasArgument( optionA ) );
        Long expectedArgument = -1L;
        assertEquals( expectedArgument, options.valueOf( "a" ) );
        assertEquals( expectedArgument, options.valueOf( optionA ) );
        assertEquals( expectedArgument, optionA.value( options ) );
        assertEquals( asList( expectedArgument ), options.valuesOf( "a" ) );
        assertEquals( asList( expectedArgument ), options.valuesOf( optionA ) );
        assertEquals( asList( expectedArgument ), optionA.values( options ) );
    }

    @Test
    public void optionalArgOptionWithDefaultGivesArgIfSpecifiedOnCommandLine() {
        OptionSpec<Long> optionA =
            parser.accepts( "a" ).withOptionalArg().ofType( Long.class ).defaultsTo( -1L );

        OptionSet options = parser.parse( "-a", "2" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.has( optionA ) );
        assertTrue( options.hasArgument( "a" ) );
        assertTrue( options.hasArgument( optionA ) );
        Long expectedArgument = 2L;
        assertEquals( expectedArgument, options.valueOf( "a" ) );
        assertEquals( expectedArgument, options.valueOf( optionA ) );
        assertEquals( expectedArgument, optionA.value( options ) );
        assertEquals( asList( expectedArgument ), options.valuesOf( "a" ) );
        assertEquals( asList( expectedArgument ), options.valuesOf( optionA ) );
        assertEquals( asList( expectedArgument ), optionA.values( options ) );
    }

    @Test
    public void requiredArgOptionWithDefaultGivesDefaultIfOptionNotOnCommandLine() {
        OptionSpec<BigDecimal> optionA =
            parser.accepts( "a" ).withRequiredArg().ofType( BigDecimal.class ).defaultsTo( TEN );

        OptionSet options = parser.parse();

        assertFalse( options.has( "a" ) );
        assertFalse( options.has( optionA ) );
        assertFalse( options.hasArgument( "a" ) );
        assertFalse( options.hasArgument( optionA ) );
        assertEquals( TEN, options.valueOf( "a" ) );
        assertEquals( TEN, options.valueOf( optionA ) );
        assertEquals( TEN, optionA.value( options ) );
        assertEquals( asList( TEN ), options.valuesOf( "a" ) );
        assertEquals( asList( TEN ), options.valuesOf( optionA ) );
        assertEquals( asList( TEN ), optionA.values( options ) );
    }

    @Test
    public void optionalArgOptionWithDefaultGivesDefaultIfOptionNotOnCommandLine() {
        OptionSpec<BigDecimal> optionA =
            parser.accepts( "a" ).withOptionalArg().ofType( BigDecimal.class ).defaultsTo( TEN );

        OptionSet options = parser.parse();

        assertFalse( options.has( "a" ) );
        assertFalse( options.has( optionA ) );
        assertFalse( options.hasArgument( "a" ) );
        assertFalse( options.hasArgument( optionA ) );
        assertEquals( TEN, options.valueOf( "a" ) );
        assertEquals( TEN, options.valueOf( optionA ) );
        assertEquals( TEN, optionA.value( options ) );
        assertEquals( asList( TEN ), options.valuesOf( "a" ) );
        assertEquals( asList( TEN ), options.valuesOf( optionA ) );
        assertEquals( asList( TEN ), optionA.values( options ) );
    }

    @Test
    public void allowsListOfDefaults() {
        OptionSpec<Integer> optionC =
            parser.accepts( "c" ).withOptionalArg().ofType( Integer.class ).defaultsTo( 1, 2, 3 );

        OptionSet options = parser.parse();

        List<Integer> expected = asList( 1, 2, 3 );
        assertEquals( expected, optionC.values( options ) );
        assertEquals( expected, options.valuesOf( optionC ) );
    }

    @Test
    public void specifiedOptionArgumentsTrumpsListOfDefaults() {
        OptionSpec<Integer> optionC =
            parser.accepts( "c" ).withRequiredArg().ofType( Integer.class )
                .defaultsTo( 1, 2, 3 ).withValuesSeparatedBy( ',' );

        OptionSet options = parser.parse( "-c", "4", "-c", "5", "-c", "6,7,8" );

        List<Integer> expected = asList( 4, 5, 6, 7, 8 );
        assertEquals( expected, optionC.values( options ) );
        assertEquals( expected, options.valuesOf( optionC ) );
    }

    @Test
    public void withCompileTimeArraySpecifyingDefaults() {
        OptionSpec<Integer> optionD =
            parser.accepts( "d" ).withRequiredArg().ofType( Integer.class ).defaultsTo( new Integer[] { 1, 2, 3 } );

        OptionSet options = parser.parse();

        List<Integer> expected = asList( 1, 2, 3 );
        assertEquals( expected, optionD.values( options ) );
        assertEquals( expected, options.valuesOf( optionD ) );
    }
}
