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

package joptsimple.internal;

import org.junit.Test;

import static joptsimple.internal.Strings.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class RowsTest {
    @Test
    public void optionsAndDescriptionsWithinOverallWidth() {
        Rows rows = new Rows( 40, 2 );
        rows.add( "left one", "right one" );
        rows.add( "another left one", "another right one" );

        assertRows( rows,
                "left one          right one        ",
                "another left one  another right one" );
    }

    @Test
    public void someOptionsExceedOverallWidth() {
        Rows rows = new Rows( 40, 2 );
        rows.add( "left one is pretty freaking long to be over here", "right one" );
        rows.add( "another left one also has length that is quite excessive", "another right one" );

        assertRows( rows,
                "left one is pretty  right one        ",
                "  freaking long to                   ",
                "  be over here                       ",
                "another left one    another right one",
                "  also has length                    ",
                "  that is quite                      ",
                "  excessive                          " );
    }

    @Test
    public void someDescriptionsExceedOverallWidth() {
        Rows rows = new Rows( 40, 2 );
        rows.add( "left one", "right one for the time we have chosen" );
        rows.add( "another left one", "another right one could be used here instead" );

        assertRows( rows,
            "left one          right one for the    ",
            "                    time we have chosen",
            "another left one  another right one    ",
            "                    could be used here ",
            "                    instead            " );
    }

    private void assertRows( Rows rows, String... expected ) {
        rows.fitToWidth();

        assertEquals( join( expected, LINE_SEPARATOR ) + LINE_SEPARATOR, rows.render() );
    }
}
