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

import java.util.Collection;

import static com.google.common.collect.Lists.newArrayList;
import static java.util.Arrays.*;

import joptsimple.util.KeyValuePair;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.hamcrest.CoreMatchers.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
@RunWith( Parameterized.class )
public class ToStringTest {
    private final Object subject;
    private final String[] substrings;

    public ToStringTest( Object subject, String[] substrings ) {
        this.subject = subject;
        this.substrings = substrings.clone();
    }

    @Parameterized.Parameters
    public static Collection<?> objectsAndStringRepresentations() {
        return asList( new Object[][] {
            { KeyValuePair.valueOf( "key=value" ), new String[] { "key", "=", "value" } },
            { new UnrecognizedOptionException( "a" ), new String[] { "a" } },
            { new NoArgumentOptionSpec( asList( "a", "b" ), "" ), new String[] { "[a, b]" } },
            { new UnavailableOptionException(
                    newArrayList( new NoArgumentOptionSpec( "a" ), new NoArgumentOptionSpec( "b" ) ) ),
              new String[] { "[a, b]" } },
        } );
    }

    @Test
    public void givesUsefulStringRepresentations() {
        String stringRepresentation = subject.toString();

        for ( String each : substrings )
            assertThat( stringRepresentation, containsString( each ) );
    }
}
