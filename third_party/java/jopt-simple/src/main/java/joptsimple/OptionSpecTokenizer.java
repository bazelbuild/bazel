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

import java.util.NoSuchElementException;

import static joptsimple.ParserRules.*;

/**
 * Tokenizes a short option specification string.
 *
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
class OptionSpecTokenizer {
    private static final char POSIXLY_CORRECT_MARKER = '+';
    private static final char HELP_MARKER = '*';

    private String specification;
    private int index;

    OptionSpecTokenizer( String specification ) {
        if ( specification == null )
            throw new NullPointerException( "null option specification" );

        this.specification = specification;
    }

    boolean hasMore() {
        return index < specification.length();
    }

    AbstractOptionSpec<?> next() {
        if ( !hasMore() )
            throw new NoSuchElementException();


        String optionCandidate = String.valueOf( specification.charAt( index ) );
        index++;

        AbstractOptionSpec<?> spec;
        if ( RESERVED_FOR_EXTENSIONS.equals( optionCandidate ) ) {
            spec = handleReservedForExtensionsToken();

            if ( spec != null )
                return spec;
        }

        ensureLegalOption( optionCandidate );

        if ( hasMore() ) {
            boolean forHelp = false;
            if ( specification.charAt( index ) == HELP_MARKER ) {
                forHelp = true;
                ++index;
            }
            spec = hasMore() && specification.charAt( index ) == ':'
                ? handleArgumentAcceptingOption( optionCandidate )
                : new NoArgumentOptionSpec( optionCandidate );
            if ( forHelp )
                spec.forHelp();
        } else
            spec = new NoArgumentOptionSpec( optionCandidate );

        return spec;
    }

    void configure( OptionParser parser ) {
        adjustForPosixlyCorrect( parser );

        while ( hasMore() )
            parser.recognize( next() );
    }

    private void adjustForPosixlyCorrect( OptionParser parser ) {
        if ( POSIXLY_CORRECT_MARKER == specification.charAt( 0 ) ) {
            parser.posixlyCorrect( true );
            specification = specification.substring( 1 );
        }
    }

    private AbstractOptionSpec<?> handleReservedForExtensionsToken() {
        if ( !hasMore() )
            return new NoArgumentOptionSpec( RESERVED_FOR_EXTENSIONS );

        if ( specification.charAt( index ) == ';' ) {
            ++index;
            return new AlternativeLongOptionSpec();
        }

        return null;
    }

    private AbstractOptionSpec<?> handleArgumentAcceptingOption( String candidate ) {
        index++;

        if ( hasMore() && specification.charAt( index ) == ':' ) {
            index++;
            return new OptionalArgumentOptionSpec<String>( candidate );
        }

        return new RequiredArgumentOptionSpec<String>( candidate );
    }
}
