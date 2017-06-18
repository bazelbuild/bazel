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

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static com.google.common.collect.Lists.newArrayList;
import static java.util.Arrays.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
@RunWith( Parameterized.class )
public class OptionExceptionMessageTest {
    private final OptionException subject;
    private final String expectedMessage;

    public OptionExceptionMessageTest( OptionException subject, String expectedMessage ) {
        this.subject = subject;
        this.expectedMessage = expectedMessage;
    }

    @Parameterized.Parameters
    public static Collection<?> exceptionsAndMessages() {
        return asList(
            new Object[] { new IllegalOptionSpecificationException( "," ), ", is not a legal option character" },
            new Object[] { new MultipleArgumentsForOptionException(
                new RequiredArgumentOptionSpec<>( asList( "b", "c" ), "d" ) ),
                "Found multiple arguments for option b/c, but you asked for only one" },
            new Object[] { new OptionArgumentConversionException(
                new RequiredArgumentOptionSpec<>( asList( "c", "number" ), "x" ), "d", null ),
                "Cannot parse argument 'd' of option c/number" },
            new Object[] { new OptionMissingRequiredArgumentException(
                new RequiredArgumentOptionSpec<>( asList( "e", "honest" ), "" ) ),
                "Option e/honest requires an argument" },
            new Object[] { new UnrecognizedOptionException( "f" ), "f is not a recognized option" },
            new Object[] { new MissingRequiredOptionsException(
                Arrays.<AbstractOptionSpec<?>> asList(
                    new NoArgumentOptionSpec( "g" ), new NoArgumentOptionSpec( "h" ) ) ),
                "Missing required option(s) [g, h]" },
            new Object[] { new MissingRequiredOptionsException(
                    Arrays.<AbstractOptionSpec<?>> asList(
                        new RequiredArgumentOptionSpec<>( asList( "p", "place" ), "spot" ),
                        new RequiredArgumentOptionSpec<>( asList( "d", "data-dir" ), "dir" ) ) ),
                    "Missing required option(s) [p/place, d/data-dir]" },
            new Object[] { new UnconfiguredOptionException( asList( "i", "j" ) ),
                "Option(s) [i, j] not configured on this parser" },
            new Object[] {
                    new UnavailableOptionException(
                            newArrayList( new NoArgumentOptionSpec( "a" ), new NoArgumentOptionSpec( "b" ) ) ),
                    "Option(s) [a, b] are unavailable given other options on the command line"
            }
        );
    }

    @Test
    public void givesCorrectExceptionMessage() {
        assertEquals( expectedMessage, subject.getLocalizedMessage() );
        assertEquals( expectedMessage, subject.getMessage() );
        assertEquals( subject.getClass().getName() + ": " + expectedMessage, subject.toString() );
    }
}
