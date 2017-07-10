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

import java.util.List;

import org.junit.Test;

import static java.lang.Short.*;
import static java.util.Arrays.*;
import static java.util.Collections.*;
import static joptsimple.ExceptionMatchers.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class TypesafeOptionArgumentRetrievalTest extends AbstractOptionParserFixture {
    @Test
    public void retrievalOfTypedRequiredArgumentsInATypesafeManner() {
        OptionSpec<Integer> optionA = parser.accepts( "a" ).withRequiredArg().ofType( Integer.class );

        OptionSet options = parser.parse( "-a", "1" );

        assertTrue( options.has( optionA ) );
        Integer valueFromOption = optionA.value( options );
        assertEquals( Integer.valueOf( 1 ), valueFromOption );
        Integer valueFromOptionSet = options.valueOf( optionA );
        assertEquals( valueFromOption, valueFromOptionSet );

        List<Integer> valuesFromOption = optionA.values( options );
        assertEquals( asList( 1 ), valuesFromOption );
        List<Integer> valuesFromOptionSet = options.valuesOf( optionA );
        assertEquals( valuesFromOption, valuesFromOptionSet );
    }

    @Test
    public void retrievalOfTypedOptionalArgumentsInATypesafeManner() {
        OptionSpec<Double> optionB = parser.accepts( "b" ).withOptionalArg().ofType( Double.class );

        OptionSet options = parser.parse( "-b", "3.14D" );

        assertTrue( options.has( optionB ) );
        assertEquals( Double.valueOf( 3.14D ), optionB.value( options ) );
        assertEquals( asList( 3.14D ), optionB.values( options ) );
    }

    @Test
    public void retrievalOfUntypedRequiredArgumentsInATypesafeManner() {
        OptionSpec<String> optionC = parser.accepts( "c" ).withRequiredArg();

        OptionSet options = parser.parse( "-c", "foo", "-c", "bar" );

        assertTrue( options.has( optionC ) );
        assertEquals( asList( "foo", "bar" ), optionC.values( options ) );
    }

    @Test
    public void retrievalOfUntypedOptionalArgumentsInATypesafeManner() {
        OptionSpec<String> optionD = parser.accepts( "d" ).withRequiredArg();

        OptionSet options = parser.parse( "-d", "foo", "-d", "bar", "-d", "baz" );

        assertTrue( options.has( optionD ) );
        List<String> valuesFromOption = optionD.values( options );
        assertEquals( asList( "foo", "bar", "baz" ), valuesFromOption );
        List<String> valuesFromOptionSet = options.valuesOf( optionD );
        assertEquals( valuesFromOption, valuesFromOptionSet );
    }

    @Test
    public void retrievalWithVoidOption() {
        OptionSpec<Void> optionE = parser.accepts( "e" );

        OptionSet options = parser.parse( "-e" );

        assertTrue( options.has( optionE ) );
        assertEquals( emptyList(), options.valuesOf( optionE ) );
    }

    @Test
    public void primitiveBooleanAllowedAsTypeSpecifier() {
        OptionSpec<Boolean> optionA = parser.accepts( "a" ).withRequiredArg().ofType( boolean.class );

        OptionSet options = parser.parse( "-a", "false" );

        assertTrue( options.has( optionA ) );
        assertEquals( asList( false ), options.valuesOf( optionA ) );
    }

    @Test
    public void primitiveByteAllowedAsTypeSpecifier() {
        OptionSpec<Byte> optionB = parser.accepts( "b" ).withOptionalArg().ofType( byte.class );

        OptionSet options = parser.parse( "-b", "3" );

        assertTrue( options.has( optionB ) );
        assertEquals( asList( Byte.valueOf( "3" ) ), options.valuesOf( optionB ) );
    }

    @Test( expected = IllegalArgumentException.class )
    public void primitiveCharAllowedAsTypeSpecifier() {
        parser.accepts( "c" ).withRequiredArg().ofType( char.class );
    }

    @Test
    public void primitiveDoubleAllowedAsTypeSpecifier() {
        OptionSpec<Double> optionD = parser.accepts( "d" ).withOptionalArg().ofType( double.class );

        OptionSet options = parser.parse( "-d", "3.1" );

        assertTrue( options.has( optionD ) );
        assertEquals( asList( 3.1D ), options.valuesOf( optionD ) );
    }

    @Test
    public void primitiveFloatAllowedAsTypeSpecifier() {
        OptionSpec<Float> optionE = parser.accepts( "e" ).withRequiredArg().ofType( float.class );

        OptionSet options = parser.parse( "-e", "2.09" );

        assertTrue( options.has( optionE ) );
        assertEquals( asList( 2.09F ), options.valuesOf( optionE ) );
    }

    @Test
    public void primitiveIntAllowedAsTypeSpecifier() {
        OptionSpec<Integer> optionF = parser.accepts( "F" ).withRequiredArg().ofType( int.class );

        OptionSet options = parser.parse( "-F", "91" );

        assertTrue( options.has( optionF ) );
        assertEquals( asList( 91 ), options.valuesOf( optionF ) );
    }

    @Test
    public void primitiveLongAllowedAsTypeSpecifier() {
        OptionSpec<Long> optionG = parser.accepts( "g" ).withOptionalArg().ofType( long.class );

        OptionSet options = parser.parse("-g", "12");

        assertTrue( options.has( optionG ) );
        assertEquals( asList( 12L ), options.valuesOf( optionG ) );
    }

    @Test
    public void primitiveShortAllowedAsTypeSpecifier() {
        OptionSpec<Short> optionH = parser.accepts( "H" ).withRequiredArg().ofType( short.class );

        OptionSet options = parser.parse( "-H", "8" );

        assertTrue( options.has( optionH ) );
        assertEquals( asList( Short.valueOf( "8" ) ), options.valuesOf( optionH ) );
    }

    @Test
    public void cannotFoolHasWithAnOptionNotIssuedFromBuilder() {
        parser.accepts( "e" );

        OptionSet options = parser.parse( "-e" );

        assertFalse( options.has( new FakeOptionSpec<Void>( "e" ) ) );
    }

    @Test
    public void cannotFoolHasArgumentWithAnOptionNotIssuedFromBuilder() {
        parser.accepts( "f" ).withRequiredArg();
        OptionSpec<String> fakeOptionF = new FakeOptionSpec<>( "f" );

        OptionSet options = parser.parse( "-f", "boo" );

        assertFalse( options.hasArgument( fakeOptionF ) );
    }

    @Test
    public void cannotFoolValueOfWithAnOptionNotIssuedFromBuilder() {
        parser.accepts( "g" ).withRequiredArg();

        OptionSet options = parser.parse( "-g", "foo" );

        assertNull( options.valueOf( new FakeOptionSpec<String>( "g" ) ) );
    }

    @Test
    public void cannotFoolValuesOfWithAnOptionNotIssuedFromBuilder() {
        parser.accepts( "h" ).withRequiredArg();

        OptionSet options = parser.parse( "-h", "foo", "-h", "bar" );

        assertEquals( emptyList(), options.valuesOf( new FakeOptionSpec<String>( "h" ) ) );
    }

    @Test( expected = ClassCastException.class )
    public void canSubvertTypeSafetyIfYouUseAnOptionSpecAsTheWrongType() {
        ArgumentAcceptingOptionSpec<String> optionI = parser.accepts( "i" ).withRequiredArg();
        optionI.ofType( Integer.class );

        OptionSet options = parser.parse( "-i", "2" );

        @SuppressWarnings( "unused" )
        String value = optionI.value( options );
    }

    @Test( expected = ClassCastException.class )
    public void canSubvertTypeSafetyIfYouGiveAnOptionSpecToOptionSetAsTheWrongType() {
        ArgumentAcceptingOptionSpec<String> optionJ = parser.accepts( "j" ).withRequiredArg();
        optionJ.ofType( Integer.class );

        OptionSet options = parser.parse( "-j", "3" );

        @SuppressWarnings( "unused" )
        String value = options.valuesOf( optionJ ).get( 0 );
    }

    @Test
    public void canUseBooleanType() {
        OptionSpec<Boolean> optionK = parser.accepts( "k" ).withRequiredArg().ofType( Boolean.class );

        OptionSet options = parser.parse( "-k", "true" );

        assertTrue( optionK.value( options ) );
    }

    @Test
    public void usesConverterIfProvided() {
        OptionSpec<Short> optionL = parser.accepts( "L" ).withRequiredArg().withValuesConvertedBy(
            new ValueConverter<Short>() {
                public Short convert( String value ) {
                    return parseShort( value );
                }

                public Class<Short> valueType() {
                    return Short.class;
                }

                public String valuePattern() {
                    return null;
                }
            } );

        OptionSet options = parser.parse( "-L", "34" );

        assertEquals( new Short( (short) 34 ), optionL.value( options ) );
    }

    @Test
    public void wrapsValueConversionExceptionsRaisedByConverter() {
        OptionSpec<Character> optionM = parser.accepts( "m" ).withRequiredArg().withValuesConvertedBy(
            new ValueConverter<Character>() {
                public Character convert( String value ) {
                    throw new ValueConversionException( value );
                }

                public Class<Character> valueType() {
                    return Character.class;
                }

                public String valuePattern() {
                    return null;
                }
            } );

        OptionSet options = parser.parse( "-m", "a" );

        thrown.expect( OptionArgumentConversionException.class );
        thrown.expect( withCauseOfType( ValueConversionException.class ) );

        optionM.value( options );
    }

    private static class FakeOptionSpec<V> implements OptionSpec<V> {
        private final String option;

        FakeOptionSpec( String option ) {
            this.option = option;
        }

        public List<String> options() {
            return asList( option );
        }

        public V value( OptionSet detectedOptions ) {
            return detectedOptions.valueOf( this );
        }

        public List<V> values( OptionSet detectedOptions ) {
            return detectedOptions.valuesOf( this );
        }

        public boolean isForHelp() {
            return false;
        }
    }
}
