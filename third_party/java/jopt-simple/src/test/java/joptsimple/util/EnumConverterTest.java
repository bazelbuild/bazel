/*
 The MIT License

 Copyright (c) 2004-2014 Paul R. Holser, Jr.

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

package joptsimple.util;

import joptsimple.ValueConversionException;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.rules.ExpectedException.none;

/**
 * @author <a href="mailto:christian.ohr@gmail.com">Christian Ohr</a>
 */
public class EnumConverterTest {
    @Rule
    public final ExpectedException thrown = none();

    private EnumConverter<TestEnum> converter;

    @Before
    public void setUp() {
        converter = new TestEnumConverter();
    }

    @Test
    public void convertsEnumValuesToEnum() {
        assertEquals( TestEnum.A, converter.convert( "A" ) );
    }

    @Test
    public void rejectsNonEnumeratedValues() {
        thrown.expect( ValueConversionException.class );

        converter.convert( "Z" );
    }

    @Test
    public void answersCorrectValueType() {
        assertSame( TestEnum.class, converter.valueType() );
    }

    @Test
    public void givesDefaultValuePattern() {
        assertEquals( "[A,B,C,D]", converter.valuePattern() );
    }

    @Test
    public void givesCustomValuePattern() {
        converter.setDelimiters( "(|)" );

        assertEquals( "(A|B|C|D)", converter.valuePattern() );
    }

    @Test
    public void ignoresCase() {
        assertEquals( TestEnum.A, converter.convert( "a" ) );
    }

    private static class TestEnumConverter extends EnumConverter<TestEnum> {
        TestEnumConverter() {
            super( TestEnum.class );
        }
    }

    private static enum TestEnum {
        A, B, C, D
    }
}
