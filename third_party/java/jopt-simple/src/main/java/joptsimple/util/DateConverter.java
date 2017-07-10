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
import java.text.ParsePosition;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import joptsimple.ValueConversionException;
import joptsimple.ValueConverter;
import joptsimple.internal.Messages;

/**
 * Converts values to {@link Date}s using a {@link DateFormat} object.
 *
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class DateConverter implements ValueConverter<Date> {
    private final DateFormat formatter;

    /**
     * Creates a converter that uses the given date formatter/parser.
     *
     * @param formatter the formatter/parser to use
     * @throws NullPointerException if {@code formatter} is {@code null}
     */
    public DateConverter( DateFormat formatter ) {
        if ( formatter == null )
            throw new NullPointerException( "illegal null formatter" );

        this.formatter = formatter;
    }

    /**
     * Creates a converter that uses a {@link SimpleDateFormat} with the given date/time pattern.  The date formatter
     * created is not {@link SimpleDateFormat#setLenient(boolean) lenient}.
     *
     * @param pattern expected date/time pattern
     * @return the new converter
     * @throws NullPointerException if {@code pattern} is {@code null}
     * @throws IllegalArgumentException if {@code pattern} is invalid
     */
    public static DateConverter datePattern( String pattern ) {
        SimpleDateFormat formatter = new SimpleDateFormat( pattern );
        formatter.setLenient( false );

        return new DateConverter( formatter );
    }

    public Date convert( String value ) {
        ParsePosition position = new ParsePosition( 0 );

        Date date = formatter.parse( value, position );
        if ( position.getIndex() != value.length() )
            throw new ValueConversionException( message( value ) );

        return date;
    }

    public Class<Date> valueType() {
        return Date.class;
    }

    public String valuePattern() {
        return formatter instanceof SimpleDateFormat
            ? ( (SimpleDateFormat) formatter ).toPattern()
            : "";
    }

    private String message( String value ) {
        String key;
        Object[] arguments;

        if ( formatter instanceof SimpleDateFormat ) {
            key = "with.pattern.message";
            arguments = new Object[] { value, ( (SimpleDateFormat) formatter ).toPattern() };
        } else {
            key = "without.pattern.message";
            arguments = new Object[] { value };
        }

        return Messages.message(
            Locale.getDefault(),
            "joptsimple.ExceptionMessages",
            DateConverter.class,
            key,
            arguments );
    }
}
