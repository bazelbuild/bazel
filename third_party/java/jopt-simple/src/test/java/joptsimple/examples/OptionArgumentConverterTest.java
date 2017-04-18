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

package joptsimple.examples;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.joda.time.LocalDate;
import org.junit.Test;

import static joptsimple.util.DateConverter.*;
import static joptsimple.util.RegexMatcher.*;
import static org.junit.Assert.*;

public class OptionArgumentConverterTest {
    @Test
    public void usesConvertersOnOptionArgumentsWhenTold() {
        OptionParser parser = new OptionParser();
        parser.accepts( "birthdate" ).withRequiredArg().withValuesConvertedBy( datePattern( "MM/dd/yy" ) );
        parser.accepts( "ssn" ).withRequiredArg().withValuesConvertedBy( regex( "\\d{3}-\\d{2}-\\d{4}" ));

        OptionSet options = parser.parse( "--birthdate", "02/24/05", "--ssn", "123-45-6789" );

        assertEquals( new LocalDate( 2005, 2, 24 ).toDate(), options.valueOf( "birthdate" ) );
        assertEquals( "123-45-6789", options.valueOf( "ssn" ) );
    }
}
