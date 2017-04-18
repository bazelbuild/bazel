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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.Locale;

import static java.util.Arrays.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
@RunWith( Parameterized.class )
public class OptionExceptionLocalizedMessageTest {
    private final OptionException subject;
    private final String expectedMessage;

    public OptionExceptionLocalizedMessageTest( OptionException subject, String expectedMessage ) {
        this.subject = subject;
        this.expectedMessage = expectedMessage;
    }

    @Parameterized.Parameters
    public static Collection<?> exceptionsAndMessages() {
        return asList(
            new Object[] { new IllegalOptionSpecificationException( "," ), "illegal option specification exception" },
            new Object[] { new MultipleArgumentsForOptionException(
                new RequiredArgumentOptionSpec<>( asList( "b", "c" ), "d" ) ),
                "multiple arguments for option exception" },
            new Object[] { new OptionArgumentConversionException(
                new RequiredArgumentOptionSpec<>( asList( "c", "number" ), "x" ), "d", null ),
                "option argument conversion exception" },
            new Object[] { new OptionMissingRequiredArgumentException(
                new RequiredArgumentOptionSpec<>( asList( "e", "honest" ), "" ) ),
                "option missing required argument exception" },
            new Object[] { new UnrecognizedOptionException( "f" ), "unrecognized option exception" },
            new Object[] { new MissingRequiredOptionsException(
                Arrays.<AbstractOptionSpec<?>> asList(
                    new NoArgumentOptionSpec( "g" ), new NoArgumentOptionSpec( "h" ) ) ),
                "missing required option exception" },
            new Object[] { new MissingRequiredOptionsException(
                    Arrays.<AbstractOptionSpec<?>> asList(
                        new RequiredArgumentOptionSpec<>( asList( "p", "place" ), "spot" ),
                        new RequiredArgumentOptionSpec<>( asList( "d", "data-dir" ), "dir" ) ) ),
                    "missing required option exception" },
            new Object[] { new UnconfiguredOptionException( asList( "i", "j" ) ),
                "unconfigured option exception" }
        );
    }

    @Test
    public void givesCorrectExceptionMessage() {
        assertEquals( expectedMessage, subject.localizedMessage( new Locale( "xx", "YY" ) ) );
    }
}
