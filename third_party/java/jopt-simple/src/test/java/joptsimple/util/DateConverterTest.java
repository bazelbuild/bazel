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

package joptsimple.util;

import java.text.DateFormat;
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.text.SimpleDateFormat;
import java.util.Date;

import static java.text.DateFormat.*;

import joptsimple.ValueConversionException;
import joptsimple.ValueConverter;
import org.joda.time.LocalDate;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static joptsimple.util.DateConverter.*;
import static org.hamcrest.CoreMatchers.*;
import static org.junit.Assert.*;
import static org.junit.rules.ExpectedException.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class DateConverterTest {
    @Rule public final ExpectedException thrown = none();

    private DateFormat notASimpleDateFormat;
    private SimpleDateFormat monthDayYear;

    @Before
    public void setUp() {
        notASimpleDateFormat = new DateFormat() {
            private static final long serialVersionUID = -1L;

            {
                setNumberFormat( NumberFormat.getInstance() );
            }

            @Override
            public StringBuffer format( Date date, StringBuffer toAppendTo, FieldPosition fieldPosition ) {
                return null;
            }

            @Override
            public Date parse( String source, ParsePosition pos ) {
                return null;
            }
        };

        monthDayYear = new SimpleDateFormat( "MM/dd/yyyy" );
    }

    @Test( expected = NullPointerException.class )
    public void rejectsNullDateFormatter() {
        new DateConverter( null );
    }

    @Test
    public void shouldConvertValuesToDatesUsingADateFormat() {
        ValueConverter<Date> converter = new DateConverter( monthDayYear );

        assertEquals( new LocalDate( 2009, 1, 24 ).toDate(), converter.convert( "01/24/2009" ) );
    }

    @Test
    public void rejectsNonParsableValues() {
        thrown.expect( ValueConversionException.class );

        new DateConverter( getDateInstance() ).convert( "@(#*^" );
    }

    @Test
    public void rejectsValuesThatDoNotEntirelyMatch() {
        thrown.expect( ValueConversionException.class );

        new DateConverter( monthDayYear ).convert( "12/25/09 00:00:00" );
    }

    @Test
    public void shouldCreateSimpleDateFormatConverter() {
        assertEquals( new LocalDate( 2009, 7, 4 ).toDate(), datePattern( "MM/dd/yyyy" ).convert( "07/04/2009" ) );
    }

    @Test
    public void rejectsNullDatePattern() {
        thrown.expect( NullPointerException.class );

        datePattern( null );
    }

    @Test
    public void shouldRaiseExceptionThatContainsDatePatternAndValue() {
        thrown.expect( ValueConversionException.class );
        thrown.expectMessage( "qwe" );
        thrown.expectMessage( monthDayYear.toPattern() );

        new DateConverter( monthDayYear ).convert( "qwe" );
    }

    @Test
    public void shouldRaiseExceptionThatContainsValueOnlyIfNotASimpleDateFormat() {
        thrown.expect( ValueConversionException.class );
        thrown.expectMessage( "asdf" );
        thrown.expectMessage( not( containsString( notASimpleDateFormat.toString() ) ) );

        new DateConverter( notASimpleDateFormat ).convert( "asdf" );
    }

    @Test
    public void shouldAnswerCorrectValueType() {
        assertSame( Date.class, new DateConverter( monthDayYear ).valueType() );
    }

    @Test
    public void shouldGiveNoValuePatternIfFormatterNotASimpleDateFormat() {
        assertEquals( "", new DateConverter( notASimpleDateFormat ).valuePattern() );
    }

    @Test
    public void shouldGiveValuePatternIfFormatterIsASimpleDateFormat() {
        assertEquals( monthDayYear.toPattern(), datePattern( monthDayYear.toPattern() ).valuePattern() );
    }
}
