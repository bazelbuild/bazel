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

import java.util.Collections;

import static java.util.Collections.*;

import org.junit.Test;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class OptionParserArgumentExceptionTest extends AbstractOptionParserFixture {
    @Test( expected = NullPointerException.class )
    public void createWithNullOptionSpec() {
        new OptionParser( null );
    }

    @Test( expected = NullPointerException.class )
    public void parseNull() {
        parser.parse( (String[]) null );
    }

    @Test( expected = NullPointerException.class )
    public void nullOptionToAccepts() {
        parser.accepts( null );
    }

    @Test( expected = NullPointerException.class )
    public void nullOptionToAcceptsWithDescription() {
        parser.accepts( null, "a weird option" );
    }

    @Test( expected = NullPointerException.class )
    public void nullOptionListToAcceptsAll() {
        parser.acceptsAll( null );
    }

    @Test( expected = IllegalArgumentException.class )
    public void emptyOptionListToAcceptsAll() {
        parser.acceptsAll( Collections.<String> emptyList() );
    }

    @Test( expected = NullPointerException.class )
    public void optionListContainingNullToAcceptsAll() {
        parser.acceptsAll( singletonList( (String) null ) );
    }

    @Test( expected = NullPointerException.class )
    public void nullOptionListToAcceptsAllWithDescription() {
        parser.acceptsAll( null, "some weird options" );
    }

    @Test( expected = NullPointerException.class )
    public void optionListContainingNullToAcceptsAllWithDescription() {
        parser.acceptsAll( singletonList( (String) null ), "some weird options" );
    }
}
