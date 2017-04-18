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

import org.infinitest.toolkit.Block;
import org.junit.Test;

import static org.infinitest.toolkit.Assertions.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ArgumentAcceptingOptionSpecTest {
    @Test( expected = NullPointerException.class )
    public void requiredArgOfNullType() {
        new RequiredArgumentOptionSpec<Void>( "a" ).ofType( null );
    }

    @Test( expected = NullPointerException.class )
    public void optionalArgOfNullType() {
        new OptionalArgumentOptionSpec<Void>( "verbose" ).ofType( null );
    }

    @Test( expected = IllegalArgumentException.class )
    public void requiredArgOfNonValueType() {
        new RequiredArgumentOptionSpec<Void>( "threshold" ).ofType( Object.class );
    }

    @Test( expected = IllegalArgumentException.class )
    public void optionalArgOfNonValueType() {
        new OptionalArgumentOptionSpec<Void>( "max" ).ofType( Object.class );
    }

    @Test
    public void requiredArgOfValueTypeBasedOnValueOf() {
        assertNoException( new Block() {
            public void execute() {
                new RequiredArgumentOptionSpec<Void>( "threshold" ).ofType( ValueOfHaver.class );
            }
        } );
    }

    @Test
    public void optionalArgOfValueTypeBasedOnValueOf() {
        assertNoException( new Block() {
            public void execute() {
                new OptionalArgumentOptionSpec<Void>( "abc" ).ofType( ValueOfHaver.class );
            }
        } );
    }

    @Test
    public void requiredArgOfValueTypeBasedOnCtor() {
        assertNoException( new Block() {
            public void execute() {
                new RequiredArgumentOptionSpec<Void>( "threshold" ).ofType( Ctor.class );
            }
        } );
    }

    @Test
    public void optionalArgOfValueTypeBasedOnCtor() {
        final OptionalArgumentOptionSpec<Ctor> spec = new OptionalArgumentOptionSpec<>( "abc" );

        assertNoException( new Block() {
            public void execute() {
                spec.ofType( Ctor.class );
                assertEquals( "foo", spec.convert( "foo" ).getS() );
            }
        } );
    }

    @Test( expected = IllegalArgumentException.class )
    public void rejectsUnicodeZeroAsCharValueSeparatorForRequiredArgument() {
        new RequiredArgumentOptionSpec<Void>( "a" ).withValuesSeparatedBy( '\u0000' );
    }

    @Test( expected = IllegalArgumentException.class )
    public void rejectsUnicodeZeroAsCharValueSeparatorForOptionalArgument() {
        new OptionalArgumentOptionSpec<Void>( "b" ).withValuesSeparatedBy( '\u0000' );
    }

    @Test( expected = IllegalArgumentException.class )
    public void rejectsUnicodeZeroInStringValueSeparatorForRequiredArgument() {
        new RequiredArgumentOptionSpec<Void>( "c" ).withValuesSeparatedBy( "::\u0000::" );
    }

    @Test( expected = IllegalArgumentException.class )
    public void rejectsUnicodeZeroInStringValueSeparatorForOptionalArgument() {
        new OptionalArgumentOptionSpec<Void>( "d" ).withValuesSeparatedBy( "::::\u0000" );
    }

    @Test( expected = NullPointerException.class )
    public void rejectsNullConverter() {
        new RequiredArgumentOptionSpec<Void>( "c" ).withValuesConvertedBy( null );
    }

    @Test( expected = NullPointerException.class )
    public void rejectsNullDefaultValue() {
        new RequiredArgumentOptionSpec<Integer>( "d" ).defaultsTo( null );
    }

    @Test( expected = NullPointerException.class )
    public void rejectsNullDefaultValueRemainder() {
        new RequiredArgumentOptionSpec<Integer>( "d" ).defaultsTo( 2, (Integer[]) null );
    }

    @Test( expected = NullPointerException.class )
    public void rejectsNullInDefaultValueRemainder() {
        new RequiredArgumentOptionSpec<Integer>( "d" ).defaultsTo( 2, 3, null );
    }
}
